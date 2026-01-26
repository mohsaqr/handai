"""
Handai Core Module
Business logic without Streamlit dependencies
"""

from .providers import LLMProvider, ProviderConfig, PROVIDER_CONFIGS
from .templates import DATASET_TEMPLATES
from .llm_client import get_client, call_llm_with_retry, fetch_local_models, fetch_openrouter_models
from .processing import TransformProcessor, GenerateProcessor, DocumentProcessor, build_results_dataframe
from .document_reader import (
    read_document, read_uploaded_file, get_files_from_folder,
    SUPPORTED_EXTENSIONS, get_supported_extensions
)
from .document_templates import (
    DOCUMENT_TEMPLATES, MASTER_SYSTEM_PROMPT,
    get_template_names, get_template, get_template_prompt, get_template_columns,
    get_template_description
)

__all__ = [
    "LLMProvider",
    "ProviderConfig",
    "PROVIDER_CONFIGS",
    "DATASET_TEMPLATES",
    "get_client",
    "call_llm_with_retry",
    "fetch_local_models",
    "fetch_openrouter_models",
    "TransformProcessor",
    "GenerateProcessor",
    "DocumentProcessor",
    "build_results_dataframe",
    # Document reader
    "read_document",
    "read_uploaded_file",
    "get_files_from_folder",
    "SUPPORTED_EXTENSIONS",
    "get_supported_extensions",
    # Document templates
    "DOCUMENT_TEMPLATES",
    "MASTER_SYSTEM_PROMPT",
    "get_template_names",
    "get_template",
    "get_template_prompt",
    "get_template_columns",
    "get_template_description",
]
