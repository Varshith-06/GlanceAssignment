"""Structured attribute extraction — the substance of Part A.

Per image we produce the record the reranker later scores against:
  garments  : [{category, color, color_lab, score}]   <- detector + LAB k-means
  formality : argmax label + full zero-shot probabilities
  environment: argmax label + full zero-shot probabilities
  caption   : short synthesised description (debugging / grids)

Why a detector + per-crop colour instead of whole-image colour: binding.
"red tie, white shirt" is only checkable if colour is measured on the tie's
pixels and the shirt's pixels separately. Whole-image dominant colour would
re-introduce exactly the bag-of-concepts failure we are trying to fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_config, nearest_palette_color, rgb_to_lab

# Fashionpedia detector labels -> canonical garment vocabulary shared with the
# query parser. Apparel parts (sleeve, collar, pocket...) are dropped: they are
# noise for retrieval and would pollute colour binding.
CATEGORY_MAP = {
    "shirt, blouse": "shirt",
    "top, t-shirt, sweatshirt": "t-shirt",
    "sweater": "sweater",
    "cardigan": "cardigan",
    "jacket": "jacket",
    "vest": "vest",
    "pants": "pants",
    "shorts": "shorts",
    "skirt": "skirt",
    "coat": "coat",
    "dress": "dress",
    "jumpsuit": "jumpsuit",
    "cape": "cape",
    "glasses": "glasses",
    "hat": "hat",
    "tie": "tie",
    "belt": "belt",
    "tights, stockings": "tights",
    "sock": "socks",
    "shoe": "shoes",
    "bag, wallet": "bag",
    "scarf": "scarf",
    "umbrella": "umbrella",
    "glove": "gloves",
    "watch": "watch",
}


class GarmentDetector:
    """YOLOS fine-tuned on Fashionpedia. Returns canonical-category boxes."""

    def __init__(self, model_name: str, threshold: float):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name).eval()
        self.threshold = threshold

    @torch.no_grad()
    def detect(self, image: Image.Image) -> list[dict]:
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        target_size = torch.tensor([image.size[::-1]])
        res = self.processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=target_size)[0]

        detections = []
        for score, label_id, box in zip(res["scores"], res["labels"], res["boxes"]):
            raw = self.model.config.id2label[int(label_id)]
            category = CATEGORY_MAP.get(raw)
            if category is None:  # apparel part or accessory we don't index
                continue
            detections.append({
                "category": category,
                "score": float(score),
                "box": [float(v) for v in box],  # xmin, ymin, xmax, ymax
            })
        # Keep the single best box per category — duplicates add no binding info.
        best: dict[str, dict] = {}
        for d in detections:
            if d["category"] not in best or d["score"] > best[d["category"]]["score"]:
                best[d["category"]] = d
        return list(best.values())


def dominant_color(image: Image.Image, box: list[float]) -> tuple[str, list[float]]:
    """K-means (k=3) over the garment crop in LAB; largest cluster's centroid
    is mapped to the nearest named palette colour. The crop is shrunk 15% per
    side to cut background/skin bleed at the box edges."""
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * 0.15, (y1 - y0) * 0.15
    crop = image.crop((x0 + dx, y0 + dy, x1 - dx, y1 - dy))
    if crop.width < 4 or crop.height < 4:
        crop = image.crop((x0, y0, x1, y1))
    crop = crop.resize((48, 48))

    lab = rgb_to_lab(np.asarray(crop.convert("RGB"))).reshape(-1, 3)
    km = KMeans(n_clusters=3, n_init=4, random_state=0).fit(lab)
    counts = np.bincount(km.labels_, minlength=3)
    centroid = km.cluster_centers_[int(np.argmax(counts))]
    name, _ = nearest_palette_color(centroid)
    return name, [round(float(v), 2) for v in centroid]


def make_caption(garments: list[dict], formality: str, environment: str) -> str:
    if not garments:
        return f"a person in a {formality} outfit, {environment} setting"
    parts = [f"{g['color']} {g['category']}" for g in garments]
    return f"a person wearing {', '.join(parts)} ({formality}, {environment})"


class AttributeExtractor:
    """Bundles detector + zero-shot heads. Embedder is injected (not created
    here) so index build loads FashionCLIP exactly once."""

    def __init__(self, embedder):
        cfg = load_config()
        self.cfg = cfg
        self.detector = GarmentDetector(cfg["models"]["detector"],
                                        cfg["index"]["detector_threshold"])
        self.embedder = embedder

    def extract(self, path: str | Path, image_emb: np.ndarray) -> dict:
        image = Image.open(path).convert("RGB")

        garments = []
        for det in self.detector.detect(image):
            color, lab = dominant_color(image, det["box"])
            garments.append({"category": det["category"], "color": color,
                             "color_lab": lab, "score": round(det["score"], 3)})

        emb = image_emb.reshape(1, -1)
        f_scores = self.embedder.zero_shot_scores(emb, self.cfg["formality_prompts"])[0]
        e_scores = self.embedder.zero_shot_scores(emb, self.cfg["environment_prompts"])[0]
        formality = max(f_scores, key=f_scores.get)
        environment = max(e_scores, key=e_scores.get)

        return {
            "image_id": Path(path).stem,
            "path": str(path),
            "garments": garments,
            "formality": formality,
            "formality_scores": {k: round(v, 4) for k, v in f_scores.items()},
            "environment": environment,
            "environment_scores": {k: round(v, 4) for k, v in e_scores.items()},
            "caption": make_caption(garments, formality, environment),
        }
