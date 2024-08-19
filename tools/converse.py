import re
import string
import random
# from django.core.signing import Signer
# from google.cloud import aiplatform
# from google.oauth2.service_account import Credentials
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.ollama import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_vertexai import VertexAIEmbeddings
from langchain_openai import ChatOpenAI
# from langchain_openai import OpenAIEmbeddings
from langchain.chains import (
        create_history_aware_retriever,
        create_retrieval_chain,
    )
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from core import settings
from typing import List, Dict, Tuple
from textwrap import dedent


def run(query: str, history: List[Tuple[str, str]] = []) -> Dict:
    """
    query: str
    This is the human input.

    history: List[Tuple[str, str]
    Past conversation of human and AI. Illustrated as:
    [
        ("human", "what's 5 + 2"),
        ("ai", "5 + 2 is 7"),
    ]
    """

    system_msg = dedent("""
        When the user greeted, asked for your name, or inquire your identity.
        You are Rhea, and you are a helpful consultant specializing in the
        React.js framework. If the topic is not React.js then respond only with
        "IDK" with the title "N/A" indicating you don't know the
        subject and are not allowed to reply to anything unrelated
        to React.js and your background. Your response is inside `You`.

        Please follow the format:
        <your response>
        Title: <title>


        Example 1 (Human: "Hi!"):
        AI: Hello, I am Rhea! I specialize in React.js. At your service!
        Title: Introduction

        Example 2 (Human: "Create a quick start to React"):
        AI: You can by creating this page:
        ```jsx
        function MyButton() {
        return (
            <button>
            I'm a button
            </button>
        );
        }

        export default function MyApp() {
        return (
            <div>
            <h1>Welcome to my app</h1>
            <MyButton />
            </div>
        );
        }
        ```
        Title: Quick Start in React.js

        Example 3 (Human: "Why is the sky blue?"):
        AI: IDK
        Title: N/A
        """)

    if settings.LLM_TYPE == 'ollama':
        embeddings = HuggingFaceEmbeddings(
            model_kwargs={'device': 'cuda'},
            cache_folder='./.cache'
        )
        chat_model = Ollama(
            model=settings.LLM_NAME,
            base_url=settings.LLM_URL,
            temperature=0,
            system=system_msg,
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
            system=system_msg,
        )

    vectorstore = FAISS.load_local(
        'vectorstore',
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever()

    # Contextualize question
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        chat_model, retriever, contextualize_q_prompt
    )

    # Answer question
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use "
        "the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, just say that you "
        "don't know. Use three sentences maximum and keep the answer "
        "concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    # Below we use create_stuff_documents_chain to feed all retrieved context
    # into the LLM. Note that we can also use StuffDocumentsChain and other
    # instances of BaseCombineDocumentsChain.
    question_answer_chain = create_stuff_documents_chain(chat_model, qa_prompt)
    rag_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )

    # Usage:
    messages = []
    messages.append(SystemMessage(content=system_msg))
    for actor, message in history:
        if actor == 'ai':
            messages.append(AIMessage(content=message))
        else:
            messages.append(HumanMessage(content=message))
    chat_history = ChatPromptTemplate(messages)
    result = rag_chain.invoke({"input": query, "chat_history": chat_history})

    sources = [doc.metadata['source'] for doc in result['context']]
    sources = list(set(sources))
    key = ''.join(random.SystemRandom().choice(
                    string.ascii_uppercase + string.digits
                ) for _ in range(10))

    try:
        matches = re.search('(.*)\\sTitle: (.*)$', result['answer'])
        answer = matches.group(1)
        title = matches.group(2)

        return {
            'question': query,
            'answer': answer,
            'metadata': {
                'sources': sources,
                'title': title,
                'key': key,
            }
        }
    except AttributeError:
        return {
            'question': query,
            'answer': result['answer'],
            'metadata': {
                'sources': sources,
                'key': key,
            }
        }
