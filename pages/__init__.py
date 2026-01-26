"""
Handai Pages Module
Multi-page application pages
"""

# Import page modules
from . import home
from . import transform
from . import generate
from . import process_documents
from . import history
from . import settings

__all__ = [
    "home",
    "transform",
    "generate",
    "process_documents",
    "history",
    "settings",
]
