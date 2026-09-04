import chromadb

import os
import json
import uuid
import numpy as np
import chromadb
from pathlib import Path
from rank_bm25 import BM25Okapi
from backend.services.embedding_service import get_embeddings

CHROMA_PATH = Path(__file__).resolve().parent.parent.parent / "rag_docs" / "chroma_db"
COLLECTION_NAME = "medexplain_knowledge_base"

_client = None
_collection = None
_bm25 = None
_bm25_chunks = []

def _init_chromadb():
    global _client, _collection
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

def ingest_documents(chunks: list[dict]):
    # _init_chromadb()
    global _bm25, _bm25_chunks

    ids, embeddings, documents, metadatas = [], [], [], []

    for chunk in chunks:
        chunk_id = f"{chunk['source_filename']}_{chunk['page_number']}_{uuid.uuid4().hex[:8]}"
        embedding = get_embeddings([chunk["text"]])[0].tolist()

        ids.append(chunk_id)
        embeddings.append(embedding)
        documents.append(chunk["text"])
        metadatas.append({
            "source": chunk["source_filename"],
            "page": chunk["page_number"],
            "chunk_index": chunk.get("chunk_index", 0)
        })

    _collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    _rebuild_bm25()
    return len(ids)


def _rebuild_bm25():
    global _bm25, _bm25_chunks
    results = _collection.get(include=["documents", "metadatas"])
    _bm25_chunks = [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(results["documents"], results["metadatas"])
    ]
    tokenized = [chunk["text"].lower().split() for chunk in _bm25_chunks]
    _bm25 = BM25Okapi(tokenized) if tokenized else None


def _dense_query(query: str, top_k: int = 5) -> list[dict]:
    # _init_chromadb()
    query_embedding = get_embeddings([query])[0].tolist()
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        output.append({
            "text": doc,
            "source": meta["source"],
            "page": meta["page"],
            "dense_score": float(1 - dist)
        })
    return output


def _bm25_query(query: str, top_k: int = 5) -> list[dict]:
    if _bm25 is None:
        _rebuild_bm25()
    if _bm25 is None:
        return []

    tokenized_query = query.lower().split()
    scores = _bm25.get_scores(tokenized_query)

    max_score = scores.max() if scores.max() > 0 else 1
    normalized = scores / max_score

    top_indices = np.argsort(normalized)[::-1][:top_k]
    output = []
    for idx in top_indices:
        chunk = _bm25_chunks[idx]
        output.append({
            "text": chunk["text"],
            "source": chunk["metadata"]["source"],
            "page": chunk["metadata"]["page"],
            "bm25_score": float(normalized[idx])
        })
    return output


def query(query_text: str, top_k: int = 3) -> list[dict]:
    dense_results = _dense_query(query_text, top_k=top_k * 2)
    bm25_results = _bm25_query(query_text, top_k=top_k * 2)

    merged = {}

    for r in dense_results:
        key = r["text"]
        merged[key] = {
            "text": r["text"],
            "source": r["source"],
            "page": r["page"],
            "dense_score": r["dense_score"],
            "bm25_score": 0.0
        }

    for r in bm25_results:
        key = r["text"]
        if key in merged:
            merged[key]["bm25_score"] = r["bm25_score"]
        else:
            merged[key] = {
                "text": r["text"],
                "source": r["source"],
                "page": r["page"],
                "dense_score": 0.0,
                "bm25_score": r["bm25_score"]
            }

    for key in merged:
        merged[key]["relevance_score"] = round(
            (merged[key]["dense_score"] * 0.6) + (merged[key]["bm25_score"] * 0.4), 4
        )

    ranked = sorted(merged.values(), key=lambda x: x["relevance_score"], reverse=True)

    return [
        {
            "text": r["text"],
            "source": r["source"],
            "page": r["page"],
            "relevance_score": r["relevance_score"]
        }
        for r in ranked[:top_k]
    ]


def list_documents() -> list[str]:
    # _init_chromadb()
    results = _collection.get(include=["metadatas"])
    sources = list({meta["source"] for meta in results["metadatas"]})
    return sources
