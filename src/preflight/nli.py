"""Optional NLI model — the upgrade from lexical overlap to entailment.

Without this, groundedness runs on lexical + numeric overlap: it can tell that
a claim's *words* appear in a source, but not whether the source actually
*entails* the claim, and it cannot distinguish "the source is silent on this"
(neutral / UNVERIFIABLE) from "the source says the opposite" (CONTRADICTED).
A trained natural-language-inference model can. That distinction is the entire
difference between a checker an ops team trusts and one they mute in week one.

Kept optional and opt-in, on purpose. The prototype's promise is that it runs
with zero model downloads; forcing a ~70 MB model on every `pip install` would
break that. So the engine defaults to the documented lexical fallback and only
loads a model when an operator sets `PREFLIGHT_NLI_MODEL` — the same
explicit-capability contract the rest of the system follows. When the model is
absent the degraded list says so; it never silently pretends to model-grade
evidence.

    PREFLIGHT_NLI_MODEL=default   # use the bundled default cross-encoder
    PREFLIGHT_NLI_MODEL=cross-encoder/nli-deberta-v3-small   # or name one

The interface is deliberately tiny — `entailment(premise, hypothesis) -> float`
— because that is all `claims.verify_claim` and the semantic-entropy / bias
detectors call. `scores()` and `contradiction()` expose the full 3-way
distribution for callers that can use it (verify_claim does, to separate
CONTRADICTED from merely UNSUPPORTED).
"""

from __future__ import annotations

import os
import sys

# A small, CPU-friendly 3-way NLI cross-encoder. Accurate enough to demonstrate
# the mechanism, light enough to load on a laptop in the demo.
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"

_ENABLE_VALUES = {"1", "on", "true", "yes", "default"}


class NLIModel:
    """Thin wrapper over a HuggingFace cross-encoder NLI model.

    Returns calibrated probabilities over {contradiction, entailment, neutral}.
    A per-instance cache means the O(N^2) pair comparisons in semantic-entropy
    clustering do not re-run the same forward pass twice.
    """

    def __init__(self, model_name: str) -> None:
        import numpy as np
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self._np = np
        self._model = CrossEncoder(model_name)
        # Resolve label positions from the model config rather than assuming an
        # order — different NLI checkpoints permute {entail, contra, neutral}.
        id2label = {int(k): v.lower()
                    for k, v in self._model.model.config.id2label.items()}
        self._idx = {v: k for k, v in id2label.items()}
        if not {"entailment", "contradiction", "neutral"} <= set(self._idx):
            raise ValueError(f"{model_name} is not a 3-way NLI model: {id2label}")
        self._cache: dict[tuple[str, str], tuple[float, float, float]] = {}

    def _distribution(self, premise: str, hypothesis: str) -> tuple[float, float, float]:
        key = (premise, hypothesis)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        logits = self._np.asarray(self._model.predict([(premise, hypothesis)]))[0]
        e = self._np.exp(logits - logits.max())
        probs = e / e.sum()
        out = (
            float(probs[self._idx["entailment"]]),
            float(probs[self._idx["contradiction"]]),
            float(probs[self._idx["neutral"]]),
        )
        if len(self._cache) < 10_000:
            self._cache[key] = out
        return out

    def entailment(self, premise: str, hypothesis: str) -> float:
        """P(premise entails hypothesis) in [0, 1]."""
        return self._distribution(premise, hypothesis)[0]

    def contradiction(self, premise: str, hypothesis: str) -> float:
        """P(premise contradicts hypothesis) in [0, 1]."""
        return self._distribution(premise, hypothesis)[1]

    def scores(self, premise: str, hypothesis: str) -> dict[str, float]:
        e, c, n = self._distribution(premise, hypothesis)
        return {"entailment": e, "contradiction": c, "neutral": n}


def load_nli(model_name: str | None = None):
    """Load the NLI model if opted in, else return None (lexical fallback).

    Resolution order:
      - explicit `model_name` argument, if given;
      - else `PREFLIGHT_NLI_MODEL` env var ("default"/"on"/"1" -> bundled model,
        anything else -> that model name);
      - else None (the default — no download, lexical fallback).

    A configured-but-unloadable model degrades to None with a loud warning,
    exactly like the Redis backend: losing a capability silently is worse than
    saying so.
    """
    raw = model_name or os.getenv("PREFLIGHT_NLI_MODEL")
    if not raw:
        return None
    name = DEFAULT_NLI_MODEL if raw.lower() in _ENABLE_VALUES else raw
    try:
        model = NLIModel(name)
        print(f"[preflight] NLI model loaded: {name} — groundedness now runs "
              f"entailment, not lexical overlap.", file=sys.stderr)
        return model
    except Exception as exc:
        print(f"[preflight] PREFLIGHT_NLI_MODEL set ({name!r}) but the model "
              f"could not be loaded ({exc!r}); falling back to lexical "
              f"groundedness.", file=sys.stderr)
        return None
