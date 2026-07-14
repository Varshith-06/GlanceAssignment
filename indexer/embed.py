"""CLIP-family embedding wrapper (FashionCLIP by default).

Exposes exactly two functions on a loaded model: embed_images / embed_text,
both L2-normalised so cosine similarity == dot product. The class is model-name
agnostic so the ablation can instantiate vanilla CLIP through the same code path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class ClipEmbedder:
    def __init__(self, model_name: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @staticmethod
    def _l2(x: torch.Tensor) -> np.ndarray:
        return (x / x.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def embed_images(self, paths: list[str | Path], batch_size: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(paths), batch_size):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
            inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
            out.append(self._l2(self.model.get_image_features(**inputs)))
        return np.concatenate(out) if out else np.empty((0, self.model.config.projection_dim), np.float32)

    @torch.no_grad()
    def embed_text(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
        return self._l2(self.model.get_text_features(**inputs))

    def zero_shot_scores(self, embs: np.ndarray, prompts: dict[str, str],
                         temperature: float = 100.0) -> list[dict[str, float]]:
        """Softmax over prompt bank per embedding (image OR text). Temperature
        100 is CLIP's standard for images; the query side uses a lower one so
        the max-prob is a usable confidence signal (100 saturates to ~1.0)."""
        labels = list(prompts.keys())
        text_embs = self.embed_text([prompts[k] for k in labels])
        logits = temperature * embs @ text_embs.T
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        return [dict(zip(labels, map(float, row))) for row in probs]
