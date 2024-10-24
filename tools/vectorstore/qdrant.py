import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from .base import BaseVS


class QdrantVS(BaseVS):
    vector_store: QdrantVectorStore

    def __init__(self, embeddings):
        super().__init__(embeddings)
        client = QdrantClient(url=os.environ['VECTORSTORE_URL'])
        collection = 'reactjs'

        if client.collection_exists(collection) is False:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    # 768 is the dimension of HuggingFaceEmbeddings
                    # 1536 is the dimension of OpenAIEmbeddings
                    size=768,
                    distance=Distance.COSINE,
                ),
            )

        self.vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            url=os.environ['VECTORSTORE_URL'],
            collection_name=collection,
        )

    def save(self, documents):
        # TODO: prevent duplicate documents
        self.vector_store.add_documents(documents)

    def search(self, query):
        return self.vector_store.similarity_search(query)
