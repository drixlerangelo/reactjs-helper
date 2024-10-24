import os
from typing import Dict
from .base import BaseVS as VectorStore
from .faiss import FaissVS as FaissVectorStore
from .qdrant import QdrantVS as QdrantVectorStore


def load_vector_store(embeddings) -> VectorStore:
    vector_stores: Dict[str, VectorStore] = {
        'faiss': FaissVectorStore,
        'qdrant': QdrantVectorStore,
    }
    engine = os.environ['VECTORSTORE_TYPE']
    engine = engine if engine in vector_stores.keys() else 'faiss'
    return vector_stores[engine](embeddings)
