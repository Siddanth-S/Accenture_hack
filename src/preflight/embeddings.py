"""Optional sentence embeddings — the upgrade from lexical overlap to meaning.

Retry / reformulation detection in the session accumulator asks one question:
"is this the same question the user already asked?". Two prompts that mean the
same thing but share almost no words —

    "What's the late fee on a missed EMI?"
    "If I don't pay my instalment on time, how much extra do they charge?"

— are a retry, and they are exactly what a determined user (or a failing model
loop) produces. The lexical fingerprint in `session.py` matches on shared
tokens, so it catches "re-asked with the same words" and misses "re-asked in
different words". That is the same paraphrase blind spot this project critiques
in fixed-signature guardrails, sitting in our cost/retry signal.

An embedding model closes it: encode each prompt to a vector, compare by cosine.
Paraphrases land near each other in the space regardless of surface wording.

Kept optional and opt-in, exactly like the NLI model (see [[nli.py]]). The
prototype's promise is that it runs with zero model downloads, so the default is
the documented lexical fallback and a model loads only when an operator sets
`PREFLIGHT_EMBED_MODEL`. When absent, the degraded list says so; it never
silently pretends to semantic matching it did not do.

    PREFLIGHT_EMBED_MODEL=default   # use the bundled default MiniLM encoder
    PREFLIGHT_EMBED_MODEL=sentence-transformers/all-mpnet-base-v2   # or name one

The interface is deliberately tiny — `encode(text) -> list[float]` returning an
L2-normalised vector — because that is all the accumulator needs. Vectors are
plain Python floats so they pickle into the Redis-backed session state without
dragging the model along.
"""

from __future__ import annotations

import os
import sys

# Small, fast, CPU-friendly. 384-dim, ~80 MB. Accurate enough for paraphrase
# matching, light enough to load on a laptop mid-demo.
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_ENABLE_VALUES = {"1", "on", "true", "yes", "default"}


class EmbeddingModel:
    """Thin wrapper over a sentence-transformers encoder.

    Returns L2-normalised vectors, so cosine similarity is a plain dot product
    and the accumulator needs no numpy at compare time. A small per-instance
    cache means re-encoding an identical prompt (a literal retry) is free.
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name)
        self._cache: dict[str, list[float]] = {}

    def encode(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        vec = self._model.encode(text, normalize_embeddings=True)
        out = [float(x) for x in vec]
        if len(self._cache) < 10_000:
            self._cache[text] = out
        return out


def load_embedder(model_name: str | None = None):
    """Load the embedding model if opted in, else return None (lexical fallback).

    Resolution order mirrors `load_nli`:
      - explicit `model_name` argument, if given;
      - else `PREFLIGHT_EMBED_MODEL` env var ("default"/"on"/"1" -> bundled
        model, anything else -> that model name);
      - else None (the default — no download, lexical retry matching).

    A configured-but-unloadable model degrades to None with a loud warning,
    exactly like the NLI and Redis paths: losing a capability silently is worse
    than saying so.
    """
    raw = model_name or os.getenv("PREFLIGHT_EMBED_MODEL")
    if not raw:
        return None
    name = DEFAULT_EMBED_MODEL if raw.lower() in _ENABLE_VALUES else raw
    try:
        model = EmbeddingModel(name)
        print(f"[preflight] embedding model loaded: {name} — retry detection now "
              f"matches meaning, not just shared words.", file=sys.stderr)
        return model
    except Exception as exc:
        print(f"[preflight] PREFLIGHT_EMBED_MODEL set ({name!r}) but the model "
              f"could not be loaded ({exc!r}); falling back to lexical retry "
              f"matching.", file=sys.stderr)
        return None
