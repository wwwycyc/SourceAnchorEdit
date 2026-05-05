"""Dataset adapters that convert raw datasets into the standard sample format."""

from .base import DatasetAdapter
from .piebench import PIEBenchAdapter

__all__ = ["DatasetAdapter", "PIEBenchAdapter"]
