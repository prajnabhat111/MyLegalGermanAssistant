import json
import os
from collections import Counter

import mlflow
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from legal_assistant.ingest import load_data


# ============================================================
# Configuration
# ============================================================

load_dotenv(override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=GROQ_API_KEY)

MODEL_ID = "openai/gpt-oss-120b"

OUTPUT_FILE = "validation_dataset_v1.json"

# Keep this small enough to stay comfortably below Groq's TPM limit.
BATCH_SIZE = 5

# Number of candidate questions generated from each batch.
QUESTIONS_PER_BATCH = 3

# Final number of golden questions.
FINAL_QUESTION_COUNT = 15

TEMPERATURE = 0.2


# ============================================================
# MLflow
# ============================================================

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("RAG Golden Dataset")


# ============================================================
# JSON schemas
# ============================================================

BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "ground_truth": {"type": "string"},
                    "question_type": {
                        "type": "string",
                        "enum": [
                            "direct",
                            "major_topic",
                            "multi_chunk",
                            "cross_section",
                            "unanswerable",
                            "partially_answerable",
                            "edge_case",
                            "specific_detail",
                            "paraphrased",
                            "ambiguous",
                        ],
                    },
                    "source_chunks": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                },
                "required": [
                    "question",
                    "ground_truth",
                    "question_type",
                    "source_chunks",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}


FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "ground_truth": {"type": "string"},
                    "question_type": {
                        "type": "string",
                        "enum": [
                            "direct",
                            "major_topic",
                            "multi_chunk",
                            "cross_section",
                            "unanswerable",
                            "partially_answerable",
                            "edge_case",
                            "specific_detail",
                            "paraphrased",
                            "ambiguous",
                        ],
                    },
                    "source_chunks": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                },
                "required": [
                    "question",
                    "ground_truth",
                    "question_type",
                    "source_chunks",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}


# ============================================================
# Generate candidates from one batch
# ============================================================

def generate_batch_questions(batch, batch_number):
    context = ""

    for chunk_idx, chunk in batch:
        context += f"""
================ CHUNK {chunk_idx} ================
{chunk}
"""

    prompt = f"""
You are creating a high-quality golden evaluation dataset for a
Retrieval-Augmented Generation (RAG) system.

You are currently looking at batch {batch_number} of document chunks.

Generate approximately {QUESTIONS_PER_BATCH} strong candidate evaluation
questions from these chunks.

Important requirements:

1. Use ONLY information contained in the provided chunks.
2. Do NOT invent facts.
3. Do NOT create one question per chunk.
4. Prefer questions that test meaningful RAG behavior.
5. Include different question types when appropriate.
6. Some questions should require information from multiple chunks.
7. Some questions may test specific details.
8. Some questions should be paraphrased rather than copying wording.
9. If the provided chunks cannot answer a question, you may create an
   unanswerable question. For an unanswerable question:
   - source_chunks must be []
   - ground_truth should clearly state that the information is not
     available in the provided document.
10. For partially answerable questions, source_chunks should identify
    the chunks containing the available information.
11. A multi_chunk question must genuinely require information from
    multiple chunks.
12. Avoid trivial questions such as "What is this document about?"
    unless that is genuinely an important evaluation case.

The goal is QUALITY and DIVERSITY, not quantity.

Document chunks:

{context}
"""

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=TEMPERATURE,
        max_tokens=1200,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "batch_questions",
                "strict": True,
                "schema": BATCH_SCHEMA,
            },
        },
    )

    content = response.choices[0].message.content

    return json.loads(content)["questions"]


# ============================================================
# Final question selection
# ============================================================

def select_final_questions(candidates):
    candidate_text = ""

    for idx, item in enumerate(candidates):
        candidate_text += f"""
================ CANDIDATE {idx} ================

Question:
{item["question"]}

Ground truth:
{item["ground_truth"]}

Type:
{item["question_type"]}

Source chunks:
{item["source_chunks"]}
"""

    prompt = f"""
You are the final curator of a golden evaluation dataset for a RAG system.

Below are candidate questions generated from different parts of a document.

Select exactly {FINAL_QUESTION_COUNT} questions.

Your goal is to create a small but high-quality evaluation set with
excellent coverage and diversity.

Prefer a balanced mixture of:

- direct retrieval
- major topics
- multi-chunk reasoning
- cross-section questions
- specific details
- paraphrased questions
- edge cases
- partially answerable questions
- unanswerable questions

Do not select duplicate or nearly duplicate questions.

Important:

- Keep the strongest questions.
- Prefer questions that test different RAG behaviors.
- Preserve the original ground truth.
- Preserve source chunk IDs.
- Do not invent information.
- An unanswerable question must have source_chunks=[].
- A multi_chunk question should have at least two source chunks.
- Return exactly {FINAL_QUESTION_COUNT} questions if enough candidates
  are available.

Candidates:

{candidate_text}
"""

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.1,
        max_tokens=2500,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "final_questions",
                "strict": True,
                "schema": FINAL_SCHEMA,
            },
        },
    )

    content = response.choices[0].message.content

    return json.loads(content)["questions"]


# ============================================================
# Main
# ============================================================

with mlflow.start_run(run_name="golden_dataset_v1"):

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    document_chunks = load_data()

    print(f"Generated {len(document_chunks)} chunks from web page.")

    mlflow.log_param("generation_model", MODEL_ID)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("questions_per_batch", QUESTIONS_PER_BATCH)
    mlflow.log_param("target_question_count", FINAL_QUESTION_COUNT)
    mlflow.log_param("temperature", TEMPERATURE)

    # --------------------------------------------------------
    # Create batches
    # --------------------------------------------------------

    indexed_chunks = list(enumerate(document_chunks))

    batches = [
        indexed_chunks[i:i + BATCH_SIZE]
        for i in range(0, len(indexed_chunks), BATCH_SIZE)
    ]

    print(f"Created {len(batches)} batches.")

    mlflow.log_metric("num_chunks", len(document_chunks))
    mlflow.log_metric("num_batches", len(batches))

    # --------------------------------------------------------
    # Generate candidate questions
    # --------------------------------------------------------

    candidates = []

    for batch_number, batch in enumerate(batches, start=1):

        print(
            f"Generating questions for batch "
            f"{batch_number}/{len(batches)}..."
        )

        try:
            batch_questions = generate_batch_questions(
                batch,
                batch_number,
            )

            candidates.extend(batch_questions)

            print(
                f"  Generated {len(batch_questions)} candidates "
                f"(total: {len(candidates)})"
            )

        except Exception as e:
            print(
                f"  Error generating batch {batch_number}: {e}"
            )

    if not candidates:
        raise RuntimeError("No candidate questions were generated.")

    print(f"\nGenerated {len(candidates)} candidate questions.")

    mlflow.log_metric(
        "num_candidate_questions",
        len(candidates),
    )

    # --------------------------------------------------------
    # Final selection
    # --------------------------------------------------------

    print(
        f"Selecting final {FINAL_QUESTION_COUNT} questions..."
    )

    final_questions = select_final_questions(candidates)

    # Make sure we don't accidentally exceed our target.
    final_questions = final_questions[:FINAL_QUESTION_COUNT]

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    eval_data = []

    for item in final_questions:
        eval_data.append(
            {
                "inputs": {
                    "question": item["question"]
                },
                "ground_truth": item["ground_truth"],
                "question_type": item["question_type"],
                "source_chunks": item["source_chunks"],
            }
        )

    eval_df = pd.DataFrame(eval_data)

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    eval_df.to_json(
        OUTPUT_FILE,
        orient="records",
        indent=2,
    )

    print(
        f"\nGolden dataset created: {OUTPUT_FILE}"
    )

    print(
        f"Final question count: {len(eval_df)}"
    )

    # --------------------------------------------------------
    # MLflow dataset tracking
    # --------------------------------------------------------

    dataset = mlflow.data.from_pandas(
        eval_df,
        source=OUTPUT_FILE,
        name="rag_golden_dataset",
    )

    mlflow.log_input(
        dataset,
        context="evaluation",
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mlflow.log_metric(
        "num_questions_generated",
        len(eval_df),
    )

    type_counts = Counter(
        eval_df["question_type"]
    )

    for question_type, count in type_counts.items():
        mlflow.log_metric(
            f"questions_{question_type}",
            count,
        )

    # --------------------------------------------------------
    # Artifact
    # --------------------------------------------------------

    mlflow.log_artifact(
        OUTPUT_FILE
    )

    # --------------------------------------------------------
    # Tags
    # --------------------------------------------------------

    mlflow.set_tag(
        "dataset_version",
        "v1",
    )

    mlflow.set_tag(
        "dataset_purpose",
        "RAG evaluation",
    )

    # --------------------------------------------------------
    # Print run information
    # --------------------------------------------------------

    run = mlflow.active_run()

    print("\nMLflow run:")
    print(f"  Experiment: RAG Golden Dataset")
    print(f"  Run ID: {run.info.run_id}")

    print("\nQuestion type distribution:")

    for question_type, count in type_counts.items():
        print(f"  {question_type}: {count}")
