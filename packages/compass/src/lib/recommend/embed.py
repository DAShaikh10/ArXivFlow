"""
SPECTER2 query embedder — the model path the real free-text perturbation needs.

The corpus vectors in Chroma were produced by papervec with the SPECTER2 **proximity** adapter over
"Title[SEP]Abstract" (paper-to-paper). SPECTER2 is trained so a free-text query embedded with the
**adhoc-query** adapter lands in that same space, which is exactly the free-text search door the
recommender plan describes. This module re-embeds (perturbed) query text so `perturb.py` can measure how
a typo'd or word-dropped query moves the ranking — the realism the model-free Gaussian-noise probe can't
reach.

Kept out of the default dependency set: `adapters` (adapter-transformers) is only pulled in via the
`perturb-text` group, and this module is imported lazily by `perturb.py`. Mirrors papervec's loading
(`packages/papervec/src/lib/allenai/specter.py`) so the embeddings are consistent with the corpus.

`@author`: DAShaikh10
"""

import os
from typing import List, Optional

import numpy as np

from src.utils import logger

BASE_MODEL = os.getenv("SPECTER2_BASE_MODEL", "allenai/specter2_base")
# HF adapter ids: "allenai/specter2_adhoc_query" is the free-text query adapter; "allenai/specter2" is the
# proximity (paper-to-paper) adapter that built the corpus. `perturb.py` uses the query adapter by default.
ADAPTERS = {
    "adhoc_query": os.getenv("SPECTER2_QUERY_ADAPTER", "allenai/specter2_adhoc_query"),
    "proximity": os.getenv("SPECTER2_PROX_ADAPTER", "allenai/specter2"),
}
MAX_SEQ_LENGTH = int(os.getenv("SPECTER2_MAX_SEQ_LENGTH", "512"))


class Specter2QueryEmbedder:
    """
    Encodes text into the corpus embedding space via a chosen SPECTER2 adapter.
    """

    def __init__(self, adapter: str = "adhoc_query") -> None:
        if adapter not in ADAPTERS:
            raise ValueError(f"Unknown adapter {adapter!r} (expected one of {sorted(ADAPTERS)})")
        self.adapter = adapter
        self._model = None
        self._tokenizer = None
        self._device = None

    def load(self) -> None:
        """Load tokenizer + adapter-enabled base model onto the best available device (CUDA/MPS/CPU)."""

        try:
            import torch  # pylint: disable=import-outside-toplevel
            from adapters import AutoAdapterModel  # pylint: disable=import-outside-toplevel
            from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel
        except ImportError as exc:  # pragma: no cover - guidance path
            raise ImportError(
                "Real text perturbation needs the 'perturb-text' dependency group. "
                "Install it with: uv sync --group perturb-text"
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self._device = torch.device(device)
        logger.info("Loading SPECTER2 base + '%s' adapter on %s", self.adapter, device)

        self._tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoAdapterModel.from_pretrained(BASE_MODEL)
        model.load_adapter(ADAPTERS[self.adapter], source="hf", load_as=self.adapter, set_active=True)
        model.set_active_adapters(self.adapter)  # explicit: guard against the "none activated" path
        model.to(self._device)
        model.eval()
        self._model = model

    def encode(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        L2-normalized [CLS] embeddings for `texts`, matching the corpus's cosine space. (N, D) float32.
        """

        import torch  # pylint: disable=import-outside-toplevel

        if self._model is None:
            self.load()

        vectors: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self._tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt", max_length=MAX_SEQ_LENGTH
            )
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with torch.no_grad():
                cls = self._model(**inputs).last_hidden_state[:, 0, :]  # [CLS], as papervec does
            vectors.append(cls.cpu().numpy().astype(np.float32))

        matrix = np.vstack(vectors)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return matrix / norms


_EMBEDDER: Optional[Specter2QueryEmbedder] = None


# pylint: disable=global-statement


def get_embedder(adapter: str = "adhoc_query") -> Specter2QueryEmbedder:
    """
    Process-wide singleton so the model is built at most once across a perturbation sweep.
    """

    global _EMBEDDER
    if _EMBEDDER is None or _EMBEDDER.adapter != adapter:
        _EMBEDDER = Specter2QueryEmbedder(adapter)
    return _EMBEDDER


# pylint: enable=global-statement
