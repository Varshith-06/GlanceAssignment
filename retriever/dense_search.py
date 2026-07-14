"""Stage 1 — dense recall. FashionCLIP text embedding vs Chroma, top-N.

Deliberately dumb: its only job is to cast a wide semantic net so the reranker
has good candidates to work with. All precision logic lives downstream.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_config
from indexer.embed import ClipEmbedder


class DenseSearcher:
    def __init__(self, embedder: ClipEmbedder | None = None):
        cfg = load_config()
        self.cfg = cfg
        self.embedder = embedder or ClipEmbedder(cfg["models"]["fashion_clip"])
        client = chromadb.PersistentClient(path=cfg["paths"]["chroma_dir"])
        self.collection = client.get_collection(cfg["paths"]["collection_name"])

    def search(self, query: str, n: int | None = None,
               query_emb=None) -> list[dict]:
        """Returns candidates: {image_id, cosine, record} — record is the full
        structured attribute dict written at index time. Pass query_emb to
        reuse an embedding computed elsewhere (avoids encoding twice)."""
        n = n or self.cfg["search"]["top_n"]
        emb = query_emb if query_emb is not None else self.embedder.embed_text([query])[0]
        res = self.collection.query(query_embeddings=[emb.tolist()], n_results=n,
                                    include=["metadatas", "distances"])
        candidates = []
        for image_id, meta, dist in zip(res["ids"][0], res["metadatas"][0],
                                        res["distances"][0]):
            candidates.append({
                "image_id": image_id,
                "cosine": 1.0 - float(dist),  # chroma cosine distance -> similarity
                "record": json.loads(meta["record_json"]),
            })
        return candidates
