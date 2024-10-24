from langchain_community.vectorstores import FAISS
from .base import BaseVS


class FaissVS(BaseVS):
    def __init__(self, embeddings):
        super().__init__(embeddings)

    def save(self, documents):
        # TODO: prevent duplicate documents
        FAISS.from_documents(
            documents,
            self.embeddings,
        ).save_local('reactjs')

    def search(self, query):
        vector_store: FAISS = \
            FAISS.load_local(
                'reactjs',
                embedding=self.embeddings,
                allow_dangerous_deserialization=True,
            )
        return vector_store.similarity_search(query)
