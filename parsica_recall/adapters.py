"""Memory system adapters for the Parsica Recall benchmark.

Adapter contract v2 (spec section 4). Five required methods, four optional:

    setup(workspace, mode)   mode is "file" or "live"
    ingest(record)           file mode only; record is a dict with "id", "text" and
                             any of "source", "kind", "ts", "meta"
    save()
    search(query, limit)     returns a list of hits, best first; a hit is
                             (id, text, score), (id, text) or a bare text string
    teardown()

    list_ids()               optional: every record id in the store (live mode)
    record_ts(ids)           optional: {id: ts} for those ids, read from the store
                             (live mode; the version tier needs it)
    memory_count()           optional: stored memories, -1 if unknown
    version()                optional: dict with version, engine_class, models, network

In live mode (parsica-recall run --live) the store already holds its records:
setup(workspace, "live") opens it read-only, ingest() is never called, and
list_ids() is what gives the run its N, band and chance line. Live mode takes your
own adapter (--adapter custom); docs/live_http_adapter.md shows the shape of one.

The adapters registered here are the reference adapters, which are yardsticks and not
memory systems (reference_adapters.py). A memory system is plugged in with --adapter
custom --custom-adapter your_module.YourAdapter, or scored from a results file with no
adapter at all (parsica-recall score; docs/adapting.md).

Every v1 adapter keeps working: an adapter whose setup() takes only a workspace
and whose ingest() takes (text, source, category) is called the v1 way, and bare
text hits get their id from the record-id hash of the text.
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseMemoryAdapter(ABC):
    """Interface any memory system implements to run the benchmark."""

    @abstractmethod
    def setup(self, workspace: str, mode: str = "file") -> None:
        """Open the store. File mode gets a fresh, empty workspace directory."""

    @abstractmethod
    def ingest(self, record: Dict[str, Any]) -> None:
        """Add one record. Keep record["id"] and return it from search()."""

    @abstractmethod
    def save(self) -> None:
        """Persist any pending writes."""

    @abstractmethod
    def search(self, query: str, limit: int) -> List[Any]:
        """Return up to `limit` hits, best first: (id, text, score) tuples, or text."""

    @abstractmethod
    def teardown(self) -> None:
        """Release connections, temp files and the like."""

    def list_ids(self) -> Optional[List[str]]:
        """Optional: every record id the store can return. None means unknown."""
        return None

    def record_ts(self, ids: List[str]) -> Optional[Dict[str, str]]:
        """Optional, live mode: the store's own timestamp for each of these ids, as
        {id: ts}; leave out ids it has no time for. A read, never a write. None
        means the store cannot say, and the version tier is then n/a."""
        return None

    def memory_count(self) -> int:
        """Optional: number of stored memories. -1 if unknown."""
        return -1

    def version(self) -> Dict[str, Any]:
        """Optional: what the engine is. engine_class is one of lexical, embedding,
        hybrid, llm_assisted (or baseline for the reference baselines). Add
        "abstain_threshold" to have abstention scored from your scores."""
        return {"version": "unknown", "engine_class": "unknown", "models": [], "network": None}


# ---------------------------------------------------------------------------
# Calling an adapter of either contract
# ---------------------------------------------------------------------------

def _params(fn) -> Optional[List[inspect.Parameter]]:
    try:
        return list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return None


def _positional(params: List[inspect.Parameter]) -> List[inspect.Parameter]:
    return [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]


def is_v1_ingest(adapter: BaseMemoryAdapter) -> bool:
    """v1 ingest is (text, source="", category="general"); v2 takes one record."""
    params = _params(adapter.ingest)
    if params is None:
        return False
    names = {p.name for p in params}
    return len(_positional(params)) >= 2 or bool(names & {"text", "source", "category"})


def call_setup(adapter: BaseMemoryAdapter, workspace: str, mode: str) -> None:
    """v1 setup takes only the workspace; v2 also takes the mode."""
    params = _params(adapter.setup)
    if params is None:
        adapter.setup(workspace, mode)
        return
    takes_mode = (len(_positional(params)) >= 2
                  or any(p.kind == p.VAR_POSITIONAL for p in params)
                  or any(p.name == "mode" for p in params))
    if takes_mode:
        adapter.setup(workspace, mode)
    else:
        adapter.setup(workspace)


def call_ingest(adapter: BaseMemoryAdapter, record: Dict[str, Any], v1: bool) -> None:
    if v1:
        adapter.ingest(record["text"], source=record["id"], category="benchmark")
    else:
        adapter.ingest(record)


def describe_version(adapter: BaseMemoryAdapter) -> Dict[str, Any]:
    """version() as a dict, whatever the adapter returned."""
    try:
        v = adapter.version() if hasattr(adapter, "version") else None
    except Exception as exc:  # an adapter bug should not sink the run
        return {"version": "unknown", "engine_class": "unknown", "models": [], "network": None,
                "error": f"version() raised {type(exc).__name__}"}
    if isinstance(v, dict):
        out = {"version": "unknown", "engine_class": "unknown", "models": [], "network": None}
        out.update(v)
        out["version"] = str(out.get("version"))
        return out
    return {"version": str(v) if v is not None else "unknown", "engine_class": "unknown",
            "models": [], "network": None}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _reference(name: str):
    def make():
        from . import reference_adapters
        return getattr(reference_adapters, name)()
    return make


# The registered adapters are the six reference adapters, all file mode. A memory system
# of your own goes through --adapter custom, in file mode and in live mode alike.
ADAPTERS = {
    'empty': _reference('EmptyAdapter'),
    'random': _reference('RandomAdapter'),
    'echo': _reference('EchoAdapter'),
    'bm25': _reference('BM25Adapter'),
    'bm25stop': _reference('BM25StopAdapter'),
    'oracle': _reference('OracleAdapter'),
}


def get_adapter(name: str) -> BaseMemoryAdapter:
    """Instantiate an adapter by name. Raises ValueError if unknown."""
    factory = ADAPTERS.get(name)
    if factory is None:
        raise ValueError(
            f"Unknown adapter '{name}'. "
            f"Available: {', '.join(ADAPTERS.keys())}, or 'custom' with "
            f"--custom-adapter my_module.MyAdapter."
        )
    return factory()


def get_custom_adapter(dotted_path: str) -> BaseMemoryAdapter:
    """Import a custom adapter from a dotted path like 'my_module.MyAdapter'. The folder
    the command is run from is searched too, after everything else, so my_module.py next
    to you is found however the command was started."""
    module_path, _, class_name = dotted_path.rpartition('.')
    if not module_path:
        raise ValueError(
            f"Custom adapter must be a dotted path like 'my_module.MyAdapter', "
            f"got '{dotted_path}'"
        )
    here = os.getcwd()
    if "" not in sys.path and here not in sys.path:
        sys.path.append(here)  # last, so a file here never shadows the standard library
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not issubclass(cls, BaseMemoryAdapter):
        raise TypeError(f"{dotted_path} is not a BaseMemoryAdapter subclass")
    return cls()
