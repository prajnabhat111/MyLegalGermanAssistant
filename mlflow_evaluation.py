"""
Free-tier-friendly RAG evaluation.

Per question this makes:
  - 1 Gemini call   (generation)
  - 1 Groq call     (one judge call that scores 5 criteria at once)
Retrieval metrics (Precision/Recall/MRR/NDCG/Hit-rate @K) are computed locally for free.

Every API result is cached on disk, so if a quota runs out you just re-run
the script later and it continues where it stopped. Nothing is logged to
MLflow until ALL questions are done, so you never get half-finished runs.

Set LIMIT = 5 first to smoke-test the whole pipeline cheaply.
"""
import hashlib
import json
import math
import os
import time

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq

from ingest import load_data
from rag_helper import RAGBaseQdrant

load_dotenv(override=True)

# ============================================================
# Configuration
# ============================================================
DATASET_FILE = "validation_dataset_combined.json"
EXPERIMENT = "Multilingual-RAG"
PROMPT_NAME = "rag-prompt"
PROMPT_VERSION = 1
RUN_NAME = f"eval_prompt_v{PROMPT_VERSION}_gemini"

GEN_MODEL = "gemini-3.5-flash"
GEN_MIN_INTERVAL = 7.0      # seconds between Gemini calls (free tier ~10 requests/min)
# thinking_budget=0 -> no hidden "thinking" tokens: faster, cheaper, fairer prompt comparison
GEN_CONFIG = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_budget=0)
)

JUDGE_MODEL = "qwen/qwen3.8-27b"   # fallback if quota is hit: "llama-3.1-8b-instant"
JUDGE_MIN_INTERVAL = 2.5    # seconds between Groq calls
JUDGE_DOCS = 4              # how many retrieved docs the judge sees
JUDGE_CHARS_PER_DOC = 700   # truncation keeps judge token usage low

K_VALUES = (5)        # set to match your retriever's top_k
LIMIT = None                # e.g. 5 for a quick test run

CACHE_FILE = f"eval_cache_{PROMPT_NAME}_v{PROMPT_VERSION}.json"

mlflow.set_tracking_uri("http://127.0.0.1:5000")

SCORE_KEYS = [
    "context_relevance",
    "answer_relevance",
    "faithfulness",
    "coherence",
    "correctness",
]


class QuotaExhausted(RuntimeError):
    pass


# ============================================================
# Small helpers
# ============================================================
class Throttle:
    def __init__(self, interval):
        self.interval = interval
        self.last = 0.0

    def wait(self):
        delay = self.interval - (time.time() - self.last)
        if delay > 0:
            time.sleep(delay)
        self.last = time.time()


gen_throttle = Throttle(GEN_MIN_INTERVAL)
judge_throttle = Throttle(JUDGE_MIN_INTERVAL)


def doc_text(doc):
    if isinstance(doc, (tuple, list)):  # (doc, score) pairs
        doc = doc[0]
    if isinstance(doc, str):
        return doc
    return getattr(doc, "page_content", None) or str(doc)


def norm(text):
    return " ".join(text.split()).lower()


def build_chunk_lookup():
    """Map chunk text -> chunk index, using the same chunks the golden dataset was built from."""
    return {norm(doc_text(c)): i for i, c in enumerate(load_data())}


def to_chunk_id(text, lookup):
    n = norm(text)
    if n in lookup:
        return lookup[n]
    head = n[:200]
    for key, idx in lookup.items():  # fallback: partial match
        if head and (head in key or key[:200] in n):
            return idx
    return None


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"gen": {}, "judge": {}}


def save_cache(cache):
    tmp = CACHE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)
    os.replace(tmp, CACHE_FILE)


def judge_key(question, answer):
    return f"{question}||{hashlib.md5(answer.encode('utf-8')).hexdigest()}"


# ============================================================
# Retrieval metrics (local, no API calls)
# ============================================================
def retrieval_metrics(retrieved_ids, relevant_ids):
    rel = set(relevant_ids)
    out = {}
    for k in K_VALUES:
        top = retrieved_ids[:k]
        hits = sum(1 for i in top if i in rel)
        dcg = sum(1 / math.log2(r + 2) for r, i in enumerate(top) if i in rel)
        idcg = sum(1 / math.log2(r + 2) for r in range(min(len(rel), k)))
        out[f"precision_at_{k}"] = hits / k
        out[f"recall_at_{k}"] = len(set(top) & rel) / len(rel)
        out[f"ndcg_at_{k}"] = dcg / idcg if idcg else 0.0
        out[f"hit_rate_at_{k}"] = float(hits > 0)
    out["mrr"] = 0.0
    for rank, i in enumerate(retrieved_ids, start=1):
        if i in rel:
            out["mrr"] = 1 / rank
            break
    return out


# ============================================================
# Gemini generation (rate-limited, retries, stops on daily quota)
# ============================================================
def generate_answer(gemini, prompt):
    for attempt in range(5):
        gen_throttle.wait()
        start = time.perf_counter()
        try:
            resp = gemini.models.generate_content(
                model=GEN_MODEL, contents=prompt, config=GEN_CONFIG
            )
            return (resp.text or "").strip(), time.perf_counter() - start
        except Exception as e:
            msg = str(e).lower().replace(" ", "").replace("_", "")
            if "429" in msg or "resourceexhausted" in msg:
                if "perday" in msg:
                    raise QuotaExhausted("Gemini DAILY quota exhausted.") from e
                wait = 20 * (attempt + 1)
                print(f"    Gemini per-minute limit, waiting {wait}s...")
                time.sleep(wait)
                continue
            if "503" in msg or "unavailable" in msg or "overloaded" in msg:
                time.sleep(5 * (attempt + 1))
                continue
            raise
    raise RuntimeError("Gemini failed after 5 attempts.")


def run_generation(rag, gemini, rows, cache, lookup):
    for i, row in enumerate(rows, start=1):
        q = row["question"]
        if q in cache["gen"]:
            continue

        t0 = time.perf_counter()
        docs = rag.retrieve_docs(q)
        retrieval_s = time.perf_counter() - t0

        prompt = rag.format_prompt(q, docs)
        answer, generation_s = generate_answer(gemini, prompt)

        texts = [doc_text(d) for d in docs]
        cache["gen"][q] = {
            "answer": answer,
            "retrieved_ids": [to_chunk_id(t, lookup) for t in texts],
            "judge_context": "\n\n".join(
                f"[Doc {n + 1}] {t[:JUDGE_CHARS_PER_DOC]}"
                for n, t in enumerate(texts[:JUDGE_DOCS])
            ),
            "retrieval_s": retrieval_s,
            "generation_s": generation_s,
        }
        save_cache(cache)
        print(f"[gen {i}/{len(rows)}] {generation_s:.1f}s  {q[:70]}")


# ============================================================
# LLM judge: ONE call scores all criteria
# ============================================================
JUDGE_PROMPT = """You are a strict evaluator of a RAG system. Score each criterion from 1 (very poor) to 5 (excellent).

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

REFERENCE ANSWER (ground truth):
{ground_truth}

Criteria:
- context_relevance: does the retrieved context contain the information needed to answer the question?
- answer_relevance: does the answer directly address the question (ignore correctness)?
- faithfulness: is every claim in the answer supported by the retrieved context? (5 = fully grounded, 1 = mostly invented). Saying the information is unavailable when the context lacks it IS faithful.
- coherence: is the answer clear, well structured and logically consistent?
- correctness: does the answer agree with the reference answer? If the reference says the information is unavailable, a correct answer states that it cannot be answered from the documents.

Return ONLY JSON:
{{"context_relevance": int, "answer_relevance": int, "faithfulness": int, "coherence": int, "correctness": int, "comment": "max 20 words"}}"""



def judge_answer(groq_client, row, gen):
    prompt = JUDGE_PROMPT.format(
        question=row["question"],
        context=gen["judge_context"] or "(nothing retrieved)",
        answer=gen["answer"] or "(empty answer)",
        ground_truth=row["ground_truth"],
    )

    last_error = None

    for attempt in range(4):
        judge_throttle.wait()

        try:
            resp = groq_client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )

            data = json.loads(resp.choices[0].message.content)
            scores = {
                k: min(5, max(1, int(data[k])))
                for k in SCORE_KEYS
            }

            return {
                **scores,
                "comment": str(data.get("comment", ""))[:200],
            }

        except Exception as e:
            last_error = e
            msg = str(e).lower()

            print(
                f"[Judge] Attempt {attempt + 1}/4 failed: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )

            if "tokens per day" in msg or "tpd" in msg:
                raise QuotaExhausted(
                    "Groq DAILY token quota exhausted."
                ) from e

            if "429" in msg or "rate limit" in msg:
                time.sleep(15 * (attempt + 1))
            else:
                time.sleep(3)

    raise RuntimeError(
        f"Judge failed after 4 attempts. "
        f"Last error: {type(last_error).__name__}: {last_error}"
    ) from last_error


def run_judging(groq_client, rows, cache):
    for i, row in enumerate(rows, start=1):
        gen = cache["gen"][row["question"]]
        key = judge_key(row["question"], gen["answer"])
        if key in cache["judge"]:
            continue
        cache["judge"][key] = judge_answer(groq_client, row, gen)
        save_cache(cache)
        print(f"[judge {i}/{len(rows)}] {cache['judge'][key]}")


# ============================================================
# Aggregate + log to MLflow
# ============================================================
def build_results(rows, cache):
    records = []
    for row in rows:
        q = row["question"]
        gen = cache["gen"][q]
        jdg = cache["judge"][judge_key(q, gen["answer"])]
        rel = row["source_chunks"]

        rec = {
            "question": q,
            "question_type": row["question_type"],
            "ground_truth": row["ground_truth"],
            "answer": gen["answer"],
            "relevant_chunks": json.dumps(rel),
            "retrieved_chunks": json.dumps(gen["retrieved_ids"]),
            "retrieval_s": gen["retrieval_s"],
            "generation_s": gen["generation_s"],
            "total_s": gen["retrieval_s"] + gen["generation_s"],
            **{k: jdg[k] for k in SCORE_KEYS},
            "judge_comment": jdg["comment"],
        }
        # Unanswerable questions have no relevant chunks -> retrieval metrics undefined (NaN)
        if rel:
            rec.update(retrieval_metrics(gen["retrieved_ids"], rel))
        records.append(rec)
    return pd.DataFrame(records)


def log_to_mlflow(df, cache):
    metrics = {}

    # 1. Retrieval effectiveness (answerable questions only)
    retrieval_cols = [
        c for c in df.columns
        if c.startswith(("precision_at_", "recall_at_", "ndcg_at_", "hit_rate_at_")) or c == "mrr"
    ]
    for c in retrieval_cols:
        if df[c].notna().any():
            metrics[f"retrieval/{c}"] = df[c].mean()
    metrics["retrieval/n_questions_scored"] = int(df["mrr"].notna().sum()) if "mrr" in df else 0

    # 2-3. Relevance + coherence (+ faithfulness, correctness), scale 1-5
    for c in SCORE_KEYS:
        metrics[f"quality/{c}"] = df[c].mean()

    # 4. Prompt performance: one 0-1 number for comparing prompt versions
    norm_scores = (df[["faithfulness", "correctness", "answer_relevance", "coherence"]] - 1) / 4
    metrics["prompt/quality_score"] = norm_scores.mean(axis=1).mean()
    for qt, grp in df.groupby("question_type"):
        metrics[f"by_type/{qt}/correctness"] = grp["correctness"].mean()
        metrics[f"by_type/{qt}/faithfulness"] = grp["faithfulness"].mean()

    # 5. Latency
    for col in ("retrieval_s", "generation_s", "total_s"):
        name = col[:-2]
        metrics[f"latency/{name}_mean_s"] = df[col].mean()
        metrics[f"latency/{name}_p50_s"] = float(np.percentile(df[col], 50))
        metrics[f"latency/{name}_p95_s"] = float(np.percentile(df[col], 95))

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_params({
            "prompt_name": PROMPT_NAME,
            "prompt_version": PROMPT_VERSION,
            "generation_model": GEN_MODEL,
            "judge_model": JUDGE_MODEL,
            "k_values": str(K_VALUES),
            "n_questions": len(df),
            "dataset": DATASET_FILE,
        })
        mlflow.set_tag("prompt", f"{PROMPT_NAME}:v{PROMPT_VERSION}")
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if pd.notna(v)})

        df.to_csv("eval_results.csv", index=False)
        mlflow.log_artifact("eval_results.csv")
        if hasattr(mlflow, "log_table"):
            mlflow.log_table(data=df, artifact_file="eval_results_table.json")

    return metrics


# ============================================================
# Main
# ============================================================
def main():
    raw = pd.read_json(DATASET_FILE)
    rows = []
    for _, r in raw.iterrows():
        inp = r["inputs"]
        chunks = r["source_chunks"] if isinstance(r["source_chunks"], list) else []
        rows.append({
            "question": inp["question"] if isinstance(inp, dict) else inp,
            "ground_truth": r["ground_truth"],
            "question_type": r["question_type"],
            "source_chunks": [int(x) for x in chunks],
        })
    if LIMIT:
        rows = rows[:LIMIT]
    print(f"Evaluating {len(rows)} questions.")

    rag = RAGBaseQdrant(prompt_name=PROMPT_NAME, prompt_version=PROMPT_VERSION)
    gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    lookup = build_chunk_lookup()
    cache = load_cache()

    try:
        run_generation(rag, gemini, rows, cache, lookup)
        run_judging(groq_client, rows, cache)
    except QuotaExhausted as e:
        print(f"\n{e}\nProgress is saved in {CACHE_FILE}. Re-run later; finished calls are not repeated.")
        return

    df = build_results(rows, cache)

    all_ids = [i for ids in df["retrieved_chunks"].map(json.loads) for i in ids]
    unmatched = sum(1 for i in all_ids if i is None)
    if unmatched:
        print(f"WARNING: {unmatched}/{len(all_ids)} retrieved docs could not be matched to a chunk "
              "index. Retrieval metrics will be too low if the Qdrant index was built from different chunks.")

    metrics = log_to_mlflow(df, cache)

    print("\n===== RESULTS =====")
    for k in sorted(metrics):
        print(f"{k:45s} {metrics[k]:.3f}")


if __name__ == "__main__":
    main()