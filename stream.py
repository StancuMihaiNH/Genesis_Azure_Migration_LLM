import json
import random
from typing import AsyncIterable, Optional

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel

from common.key_vault_manager import KeyVaultManager
from enums.model import OpenAIModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from nh.stream_document_qa_api import get_sourced_documents
from src.helper import get_history_question, get_prompt
from utils.constants import KeyVaultSecretKeys
from utils.prompt import *
from langchain_anthropic import ChatAnthropic
from utils.logging_config  import setup_logging

setup_logging()

key_vault_manager = KeyVaultManager.getInstance()
OPENAI_API_KEY_LIST = key_vault_manager.get_secret(
    KeyVaultSecretKeys.OPENAI_API_KEY_LIST
).split(",")
ANTHROPIC_API_KEY_LIST = key_vault_manager.get_secret(
    KeyVaultSecretKeys.ANTHROPIC_KEY_LIST
).split(",")

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


class Item(BaseModel):
    messages: Optional[list] = None
    tags: Optional[list] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    files: Optional[list] = None


def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


async def handle_gpt_4model(callback, chat_history, session_user_question, user_prompt):
    llm = ChatOpenAI(
        model=OpenAIModel.GPT_4.value,
        max_tokens=4000,
        streaming=True,
        verbose=True,
        callbacks=[callback],
        api_key=random.choice(OPENAI_API_KEY_LIST),
    )
    try:
        chain = user_prompt | llm | StrOutputParser()
        async for msg in chain.astream(
            {"chat_history": chat_history, "user_question": session_user_question}
        ):
            print(msg) 
            yield msg + "\n"
    except Exception as e:
        print(e)
        yield "There is some Error to generate response. Please Contact AI-CoE with Full Screen Screenshot"


async def handle_claude_model(
    callback, chat_history, session_user_question, user_prompt
):
    api_key = random.choice(ANTHROPIC_API_KEY_LIST)
    llm_anthropic = ChatAnthropic(
        temperature=0.3,
        max_tokens=4000,
        model_name="claude-3-opus-20240229",
        streaming=True,
        verbose=True,
        callbacks=[callback],
        api_key=api_key,
    )
    try:
        chain = user_prompt | llm_anthropic | StrOutputParser()
        async for msg in chain.astream(
            {"chat_history": chat_history, "user_question": session_user_question}
        ):
            print(msg)
            yield msg + "\n"
    except Exception as e:
        print(e)
        yield "There is some Error to generate response. Please Contact AI-CoE with Full Screen Screenshot"


async def handle_nh_qa_model(callback, chat_history, session_user_question):
    try:
        contextualized_question, source_docs, formatted_docs = get_sourced_documents(
            session_user_question, chat_history
        )
        docs = [
            {"file_name": d.metadata["file_name"], "content": d.page_content}
            for d in source_docs
        ]
        yield json.dumps(docs, default=set_default)

        llm = ChatOpenAI(
            model=OpenAIModel.GPT_4.value,
            max_tokens=4000,
            streaming=True,
            verbose=True,
            callbacks=[callback],
            api_key=random.choice(OPENAI_API_KEY_LIST),
        )

        answer_input = {
            "chat_history": chat_history,
            "context": formatted_docs,
            "question": contextualized_question,
        }

        answer_chain = RunnablePassthrough() | qa_prompt | llm

        async for chunk in answer_chain.astream(answer_input):
            yield f"{chunk.content}\n"
    except Exception as e:
        print(e)
        yield "There is some Error to generate response. Please Contact AI-CoE with Full Screen Screenshot"


async def handle_title_model(callback, chat_history, user_prompt):
    llm = ChatOpenAI(
        model=OpenAIModel.GPT_4.value,
        max_tokens=4000,
        streaming=True,
        verbose=True,
        callbacks=[callback],
        api_key=random.choice(OPENAI_API_KEY_LIST),
    )
    try:
        chain = user_prompt | llm | StrOutputParser()
        title_result = ""
        async for msg in chain.astream({"chat_history": chat_history}):
            title_result += msg
        yield json.dumps({"title": title_result.strip()}, default=set_default)
    except Exception as e:
        print(e)
        yield "There is some Error to generate response. Please Contact AI-CoE with Full Screen Screenshot"


async def send_message(item: Item) -> AsyncIterable[str]:
    session_memory, chat_history, session_user_question, session_files, session_tags = (
        get_history_question(item)
    )
    callback = AsyncIteratorCallbackHandler()
    list_files = session_files if len(session_files) > 0 else []
    if item.model == OpenAIModel.GPT_4.value:
        user_prompt = get_prompt(item.model, session_tags, list_files)
        async for msg in handle_gpt_4model(
            callback, chat_history, session_user_question, user_prompt
        ):
            yield msg + "\n"

    elif (item.model == OpenAIModel.CLAUDE_3_OPUS.value or item.model == OpenAIModel.CLAUDE_3_OPOUS.value):
        user_prompt = get_prompt(item.model, session_tags, list_files)
        async for msg in handle_claude_model(
            callback, chat_history, session_user_question, user_prompt
        ):
            yield msg + "\n"

    elif item.model == OpenAIModel.NH_QA.value:
        async for msg in handle_nh_qa_model(
            callback, chat_history, session_user_question
        ):
            print(msg)
            yield msg + "\n"

    elif item.model == OpenAIModel.TITLE.value:
        user_prompt = get_prompt(item.model, session_tags, list_files)
        async for msg in handle_title_model(callback, chat_history, user_prompt):
            print(msg)            
            yield msg + "\n"

    else:
        pass


class StreamRequest(BaseModel):
    message: str


@app.post("/chat_stream/")
def stream(item: Item):
    return StreamingResponse(send_message(item), media_type="text/event-stream")


# To run the server, use the following command in the terminal:
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# where `main` is the name of your Python file (without .py)