"""Parsica Recall: a deterministic retrieval benchmark for agent memory systems.

The judge never calls a model, the window it scores is set by the size of the store and
not by the system under test, and every number can be re-derived from the per-question
records a run writes. The specification of record is docs/spec.md.

    parsica-recall run --adapter bm25 --verify        # or: python -m parsica_recall run ...

To plug in a memory system, implement BaseMemoryAdapter (docs/adapting.md), or score a
results file that any system can write (parsica-recall score).
"""

from .adapters import BaseMemoryAdapter
from .recall_common import HARNESS_VERSION as __version__
from .recall_common import SPEC_VERSION

__all__ = ["BaseMemoryAdapter", "SPEC_VERSION", "__version__"]
