"""Build the search index: embeddings + structured attributes -> ChromaDB.

Orchestration only — embedding lives in embed.py, attribute logic in
attributes.py. Idempotent: images already present in metadata.jsonl are
skipped, and Chroma writes are upserts keyed by image_id.

Run:  python indexer/build_index.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import chromadb
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_config
from indexer.attributes import AttributeExtractor
from indexer.download_data import subsample
from indexer.embed import ClipEmbedder


def load_existing(metadata_file: Path) -> dict[str, dict]:
    if not metadata_file.exists():
        return {}
    with open(metadata_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return {r["image_id"]: r for r in records}


def main() -> None:
    cfg = load_config()
    paths = subsample()

    metadata_file = Path(cfg["paths"]["metadata_file"])
    existing = load_existing(metadata_file)
    todo = [p for p in paths if p.stem not in existing]
    print(f"{len(existing)} images already indexed, {len(todo)} to process")

    client = chromadb.PersistentClient(path=cfg["paths"]["chroma_dir"])
    collection = client.get_or_create_collection(
        cfg["paths"]["collection_name"], metadata={"hnsw:space": "cosine"})

    if todo:
        embedder = ClipEmbedder(cfg["models"]["fashion_clip"])
        extractor = AttributeExtractor(embedder)

        # Reuse full-pool embeddings cached by download_data's diversity
        # probes — identical model and preprocessing, so re-encoding here
        # would only burn CPU.
        cache_path = Path(cfg["paths"]["raw_dir"]).parent / "pool_fashionclip.npz"
        cached = {}
        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            cached = {Path(n).stem: e for n, e in zip(data["names"], data["embs"])}
        fresh = [p for p in todo if p.stem not in cached]
        print(f"Embedding {len(fresh)} images with FashionCLIP "
              f"({len(todo) - len(fresh)} from cache)...")
        fresh_embs = dict(zip(
            (p.stem for p in fresh),
            embedder.embed_images(fresh, batch_size=cfg["index"]["batch_size"])))
        embs = np.stack([cached[p.stem] if p.stem in cached else fresh_embs[p.stem]
                         for p in todo])

        with open(metadata_file, "a", encoding="utf-8") as meta_out:
            batch_ids, batch_embs, batch_metas = [], [], []
            for path, emb in tqdm(zip(todo, embs), total=len(todo),
                                  desc="Extracting attributes"):
                record = extractor.extract(path, emb)
                meta_out.write(json.dumps(record) + "\n")
                batch_ids.append(record["image_id"])
                batch_embs.append(emb.tolist())
                # Chroma metadata must be flat scalars; full record rides along
                # as JSON and is what the reranker consumes.
                batch_metas.append({
                    "path": record["path"],
                    "formality": record["formality"],
                    "environment": record["environment"],
                    "record_json": json.dumps(record),
                })
                if len(batch_ids) >= 64:
                    collection.upsert(ids=batch_ids, embeddings=batch_embs,
                                      metadatas=batch_metas)
                    batch_ids, batch_embs, batch_metas = [], [], []
            if batch_ids:
                collection.upsert(ids=batch_ids, embeddings=batch_embs,
                                  metadatas=batch_metas)

    print(f"Index complete: {collection.count()} vectors in "
          f"'{cfg['paths']['collection_name']}' at {cfg['paths']['chroma_dir']}")


if __name__ == "__main__":
    main()
