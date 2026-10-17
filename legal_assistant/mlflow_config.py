from dataclasses import dataclass
from typing import ClassVar


# MLflow configuration constants
@dataclass(frozen=True)
class MLflowConfig:
    TRACKING_URI: ClassVar[str] = (
        "http://127.0.0.1:5000"  # URI for the MLflow tracking server
    )


# Prompt registry configuration constants
@dataclass(frozen=True)
class PromptConfig:
    PROMPT_NAME: ClassVar[str] = (
        "rag-prompt"  # Name for the registered system prompt in MLflow
    )
    PROMPT_ALIAS: ClassVar[str] = "production"
    TAGS: ClassVar[dict] = {
        "author": "prajnabhat1199@gmail.com",
        "task": "RAG chat prompt",
        "language": "en",
    }


# Model registry configuration constants
@dataclass(frozen=True)
class ModelRegistryConfig:
    REGISTERED_MODEL_NAME: ClassVar[str] = (
        "Multilingual-RAG"  # Name for the logged agent model in MLflow
    )
    EXPERIMENT_NAME: ClassVar[str] = "Multilingual-RAG"  # Name of the MLflow experiment
    MODEL_ALIAS: ClassVar[str] = "champion"  # Alias for the registered model version


# Model evaluation configuration constants
@dataclass(frozen=True)
class ModelEvaluationConfig:
    EVALUATION_DATASET_NAME: ClassVar[str] = (
        "validation_set"  # Name for the evaluation dataset
    )