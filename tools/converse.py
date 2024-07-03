from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.ollama import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.chains.conversational_retrieval.base \
    import ConversationalRetrievalChain
from core import settings
from typing import List, Dict, Tuple


def run(query: str, history: List[Tuple[str, str]] = []) -> Dict:
    if settings.LLM_TYPE == 'ollama':
        embeddings = HuggingFaceEmbeddings(
            model_kwargs={'device': 'cuda'},
            cache_folder='./.cache'
        )
        chat_model = Ollama(
            model=settings.LLM_NAME,
            base_url=settings.LLM_URL,
            temperature=0,
        )
    elif settings.LLM_TYPE == 'openai':
        embeddings = HuggingFaceEmbeddings(
            model_kwargs={'device': 'cpu'},
            cache_folder='./.cache'
        )
        chat_model = ChatOpenAI(
            openai_api_key=settings.LLM_KEY,
            model_name=settings.LLM_NAME,
            temperature=0,
        )
    else:
        embeddings = HuggingFaceEmbeddings(
            model_kwargs={'device': 'cpu'},
            cache_folder='./.cache'
        )
        chat_model = ChatGoogleGenerativeAI(
            google_api_key=settings.LLM_KEY,
            model=settings.LLM_NAME,
            convert_system_message_to_human=True,
            temperature=0,
        )

    vectorstore = FAISS.load_local(
        'vectorstore',
        embeddings,
        allow_dangerous_deserialization=True,
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 10}
        ),
        return_source_documents=True,
    )

    result = qa_chain.invoke({
        "question": query,
        "chat_history": history,
    })

    sources = [doc.metadata['source'] for doc in result['source_documents']]
    sources = list(set(sources))

    return {
        'question': query,
        'answer': result['answer'],
        'metadata': {
            'sources': sources,
        }
    }
