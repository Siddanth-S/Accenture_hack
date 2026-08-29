"""Optional model-based prompt-injection classifier.

The signature + intent heuristics in `detectors/core.py` are precise and cost
nothing, but an external benchmark (`eval/benchmark.py` over
deepset/prompt-injections) shows what they cannot do: on attacks phrased outside
their patterns — and on any non-English attack — recall collapses. That is the
honest limitation of a fixed rule set, and it is exactly why `detect_injection`
exposes `set_injection_classifier()`: a trained classifier blends in (by max) and
carries the paraphrased and multilingual attacks the rules miss.

Kept optional and opt-in, like the NLI and embedding models (see [[nli.py]],
[[embeddings.py]]). The prototype runs with zero downloads; a model loads only
when an operator sets `PREFLIGHT_INJECTION_MODEL`. When absent, `detect_injection`
runs heuristics alone and the detector name omits `+model`, so the degraded
capability is stated, never hidden.

    PREFLIGHT_INJECTION_MODEL=default   # bundled ProtectAI deberta-v3 classifier
    PREFLIGHT_INJECTION_MODEL=some-org/my-injection-model   # or name one

The interface is exactly what `set_injection_classifier` wants: a
`callable(text) -> float` returning P(injection) in [0, 1].
"""
from __future__ import annotations

import os
import sys

# A widely-used, permissively-licensed binary prompt-injection classifier.
# ~440 MB, CPU-runnable, multilingual — good enough to demonstrate the lift over
# heuristics on independent data.
DEFAULT_INJECTION_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"

_ENABLE_VALUES = {"1", "on", "true", "yes", "default"}


class InjectionClassifier:
    """Thin wrapper over a HuggingFace text-classification pipeline.

    Returns P(injection). Resolves the INJECTION label from the model config
    rather than assuming label order, and caches per-text so a benchmark sweep
    does not re-run the same forward pass."""

    def __init__(self, model_name: str) -> None:
        from transformers import pipeline

        self.name = model_name
        self._pipe = pipeline("text-classification", model=model_name,
                              truncation=True, max_length=512)
        id2label = {int(k): v.upper()
                    for k, v in self._pipe.model.config.id2label.items()}
        if not any("INJ" in v for v in id2label.values()):
            raise ValueError(f"{model_name} has no INJECTION label: {id2label}")
        self._cache: dict[str, float] = {}

    def __call__(self, text: str) -> float:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        out = self._pipe(text)[0]  # {"label": ..., "score": ...}
        label = str(out["label"]).upper()
        p = float(out["score"])
        # Binary head: convert to P(injection) regardless of which class won.
        prob = p if "INJ" in label else 1.0 - p
        if len(self._cache) < 20_000:
            self._cache[text] = prob
        return prob


def load_injection_classifier(model_name: str | None = None):
    """Load the injection classifier if opted in, else None (heuristics only).

    Resolution mirrors `load_nli` / `load_embedder`:
      - explicit `model_name`, if given;
      - else `PREFLIGHT_INJECTION_MODEL` ("default"/"on"/"1" -> bundled model,
        else that name);
      - else None.

    A configured-but-unloadable model degrades to None with a loud warning —
    the same explicit-degradation contract as everything else here."""
    raw = model_name or os.getenv("PREFLIGHT_INJECTION_MODEL")
    if not raw:
        return None
    name = DEFAULT_INJECTION_MODEL if raw.lower() in _ENABLE_VALUES else raw
    try:
        clf = InjectionClassifier(name)
        print(f"[preflight] injection classifier loaded: {name} — the input gate "
              f"now blends heuristics with a trained model.", file=sys.stderr)
        return clf
    except Exception as exc:
        print(f"[preflight] PREFLIGHT_INJECTION_MODEL set ({name!r}) but the model "
              f"could not be loaded ({exc!r}); falling back to signature+intent "
              f"heuristics only.", file=sys.stderr)
        return None
