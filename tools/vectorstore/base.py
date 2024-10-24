from typing import List
from langchain_core.vectorstores.base import Embeddings
from langchain_core.documents.base import Document


class BaseVS:
    embeddings: Embeddings

    def __init__(self, embeddings) -> None:
        self.embeddings = embeddings

    def save(self, documents: List[Document]):
        pass

    def search(self, query: str):
        pass
