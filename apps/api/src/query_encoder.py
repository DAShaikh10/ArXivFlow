"""
SPECTER2 ad-hoc query encoder — the query-time half of Semantic Search v2.

`@author`: DAShaikh10
"""

import threading
from typing import List

import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

from . import config


class Specter2QueryEncoder:
    """
    Lazily-loaded SPECTER2 ad-hoc query encoder.

    The heavy model is loaded on first `embed()` (guarded so concurrent first requests load it once) and
    cached thereafter. torch/transformers/adapters are imported inside the load so the dependency is only
    materialised when dense search is actually used.
    """

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        """
        True once the model is loaded and dense search can serve without a cold-start stall.
        """

        return self._model is not None

    def warmup(self) -> None:
        """
        Force the (lazy) model load now — call at startup so the first user's dense search doesn't pay
        the one-time download + torch-init stall in-band. Safe to run in a background thread: the load
        is lock-guarded, so a dense request arriving mid-warmup simply waits on the same load.
        """

        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        """
        Load tokenizer + adapter-enabled model onto the best device, exactly once.
        """

        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            device = torch.device(
                "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            )

            tokenizer = AutoTokenizer.from_pretrained(config.SPECTER2_BASE_MODEL)
            model = AutoAdapterModel.from_pretrained(config.SPECTER2_BASE_MODEL)
            # The ad-hoc query adapter lives in the same space as the proximity doc vectors in Chroma.
            model.load_adapter(
                config.SPECTER2_QUERY_ADAPTER,
                source="hf",
                load_as="specter2_adhoc_query",
                set_active=True,
            )
            model.to(device)
            model.eval()

            self._device = device
            self._tokenizer = tokenizer
            self._model = model

    def embed(self, query: str) -> List[float]:
        """
        Embed a raw query string into a SPECTER2 ad-hoc query vector.

        Raw text only (no [SEP], no abstract) with [CLS] pooling — the ad-hoc query formatting. The
        Chroma collection is a cosine space, so the returned raw vector is compared directly against the
        stored proximity document vectors.
        """

        self._ensure_loaded()

        inputs = self._tokenizer(
            [query],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=config.SPECTER2_MAX_SEQ_LENGTH,
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            # [CLS] token (sequence index 0) — the representation SPECTER2 uses for matching.
            embedding = outputs.last_hidden_state[:, 0, :]

        return embedding[0].cpu().tolist()


# Module-level singleton, mirroring `store` — one encoder shared across requests.
query_encoder = Specter2QueryEncoder()
