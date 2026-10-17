import json
import tempfile

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

SOURCE_EXPERIMENT = "RAG Validation Dataset"
NEW_EXPERIMENT = "RAG Validation Dataset Combined"
ARTIFACT_NAME = "validation_dataset_v1.json"
OUTPUT_FILE = "validation_dataset_combined.json"

# Paste the two run IDs here (MLflow UI -> click a run -> Run ID).
# Leave empty to auto-pick every run that finished with questions.
RUN_IDS = []

client = MlflowClient()

if not RUN_IDS:
    exp = client.get_experiment_by_name(SOURCE_EXPERIMENT)
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="metrics.num_questions_generated > 0",
        order_by=["start_time ASC"],
    )
    RUN_IDS = [r.info.run_id for r in runs]

print("Using runs:", RUN_IDS)

rows = []
for run_id in RUN_IDS:
    path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=ARTIFACT_NAME,
        dst_path=tempfile.mkdtemp(),
    )
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    for item in items:
        item["source_run_id"] = run_id
    rows.extend(items)
    print(f"  {run_id}: {len(items)} questions")

# Remove duplicate questions (case/whitespace-insensitive).
seen, unique = set(), []
for item in rows:
    key = " ".join(item["inputs"]["question"].lower().split())
    if key not in seen:
        seen.add(key)
        unique.append(item)

print(f"Total {len(rows)} -> {len(unique)} after dedup")

df = pd.DataFrame(unique)
df.to_json(OUTPUT_FILE, orient="records", indent=2, force_ascii=False)

mlflow.set_experiment(NEW_EXPERIMENT)
with mlflow.start_run(run_name="golden_dataset_combined"):
    dataset = mlflow.data.from_pandas(
        df, source=OUTPUT_FILE, name="rag_golden_dataset_combined"
    )
    mlflow.log_input(dataset, context="evaluation")
    mlflow.log_artifact(OUTPUT_FILE)
    mlflow.log_metric("num_questions_generated", len(df))
    for qtype, n in df["question_type"].value_counts().items():
        mlflow.log_metric(f"questions_{qtype}", int(n))
    mlflow.set_tag("dataset_version", "v1_combined")
    mlflow.set_tag("source_runs", ",".join(RUN_IDS))

print(f"Saved {len(df)} questions to {OUTPUT_FILE} and logged to '{NEW_EXPERIMENT}'")