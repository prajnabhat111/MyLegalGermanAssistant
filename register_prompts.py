import mlflow

from mlflow_config import (MLflowConfig, ModelRegistryConfig,
                                  PromptConfig)
from prompt_template import USER_PROMPT_TEMPLATE

# ── MLflow setup ────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(MLflowConfig.TRACKING_URI)
mlflow.set_experiment(ModelRegistryConfig.EXPERIMENT_NAME)

# ── Register the chat prompt ────────────────────────────────────────────────
# Creates a new version of the prompt in the Prompt Registry.
registered_prompt = mlflow.genai.register_prompt(
    name=PromptConfig.PROMPT_NAME,
    template=USER_PROMPT_TEMPLATE,
    tags=PromptConfig.TAGS,
)

# ── Assign a stable alias ──────────────────────────────────────────────────
# Points the alias (e.g. "production") to the newly registered version
mlflow.genai.set_prompt_alias(
    name=PromptConfig.PROMPT_NAME,
    alias=PromptConfig.PROMPT_ALIAS,
    version=registered_prompt.version,
)
