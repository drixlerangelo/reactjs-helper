
# from django.core.signing import Signer
# from google.cloud import aiplatform
# from google.oauth2.service_account import Credentials
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.ollama import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_vertexai import VertexAIEmbeddings
from langchain_openai import ChatOpenAI
# from langchain_openai import OpenAIEmbeddings
from langchain.chains.conversational_retrieval.base \
    import ConversationalRetrievalChain
from core import settings
from typing import List, Dict, Tuple
from textwrap import dedent


def run(query: str, history: List[Tuple[str, str]] = []) -> Dict:
    if settings.LLM_TYPE == 'ollama':
        embeddings = HuggingFaceEmbeddings(
            model_kwargs={'device': 'cpu'},
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

    history.insert(0, {"role": "system", "content": "You are a helpful assistant."})

    # chat_model.system = dedent("""
    #     When the user greeted, asked for your name, or inquire your identity.
    #     You are Rhea, and you are a helpful consultant specializing in the
    #     React.js framework. If the topic is not React.js then respond only with
    #     "IDK" with the title "N/A" indicating you don't know the
    #     subject and are not allowed to reply to anything unrelated
    #     to React.js and your background. Your response is inside `You`.

    #     Please follow the format:
    #     <your response>
    #     Title: <title>


    #     Example 1 (Input: "Hi!"):
    #     Hello, I am Rhea! I specialize in React.js. At your service!
    #     Title: Introduction

    #     Example 2 (Input: "Create a quick start to React"):
    #     You can by creating this page:
    #     ```jsx
    #     function MyButton() {
    #     return (
    #         <button>
    #         I'm a button
    #         </button>
    #     );
    #     }

    #     export default function MyApp() {
    #     return (
    #         <div>
    #         <h1>Welcome to my app</h1>
    #         <MyButton />
    #         </div>
    #     );
    #     }
    #     ```
    #     Title: Quick Start in React.js

    #     Example 3 (Input: "Why is the sky blue?"):
    #     IDK
    #     Title: N/A
    #     """)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 10}
        ),
        return_source_documents=True,
    )

    # system_template = "You are a helpful assistant that translates
    # {input_language} to {output_language}."
    # human_template = "Translate the text below and don't show any other
    #  response: \n{text}"

    # chat_prompt = ChatPromptTemplate.from_messages([
    #     ('system', system_template),
    #     ('human', human_template),
    # ])

    # messages = chat_prompt.format_messages(
    #     input_language=supported_langs[source_lang],
    #     output_language=supported_langs[target_lang],
    #     text=re.sub(r'\s+', ' ', request.GET['text'].strip())
    # )

    # result = chat_model.invoke(messages)

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
