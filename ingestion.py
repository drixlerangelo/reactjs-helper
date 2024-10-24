import json
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from tools.vectorstore import load_vector_store

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_kwargs={'device': 'cuda'},
    cache_folder='./.cache'
)


def ingest_docs():
    # 1. Gathering the sources
    file = open('sources.json', 'r')
    sources = json.load(file)
    file.close()

    # 2. Load the sources
    loader = WebBaseLoader(sources)
    raw_documents = loader.load()

    # 3. Determine and split by chunks
    # Based on last testing:
    # Total length: 842067
    # Largest length: 76477
    # Smallest length: 3250
    # Average length: 19137.886363636364
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=128
    )
    documents = text_splitter.split_documents(raw_documents)

    # 4. Save to a vector store
    vector_store = load_vector_store(embeddings)
    vector_store.save(documents)

    # 5. Test the vector store
    results = vector_store.search('What is React.js?')
    for index, result in enumerate(results):
        print(f'Source {index + 1}:\n \
            \nTitle: {result.metadata.get('title')} \
            \nContent: {result.page_content}\n')


if __name__ == '__main__':
    ingest_docs()
