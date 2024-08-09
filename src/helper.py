import re
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage

from utils.blob_utils import get_content_from_azure_blob
from utils.documents import *
from utils.get_tags import *
from utils.prompt import *


def get_prompt(model_name, session_tags, files):
    if len(session_tags) == 0 and len(files) == 0 and model_name != "title":
        prompt = ChatPromptTemplate.from_template(chat_template)

    elif len(session_tags) > 0 and len(files) == 0:
        prompt_text = """
            **Please first read the document, If you can give answer using only this document, please provide the answer using document content. If the answer is not belong from the document please use your own knowldge to answer the question. Take History as context.**

            When you respond about any questions, you need to think about the company from multiple perspectives, including what the company is doing and what is most important for the company.
            Each answer must contain: Introduction, Approach, and Conclusion.
            Approach should be comprehensive and detail every perspective of the given company.
            In Approach, describe each point in very detailed paragraphs.
            Always take a breath and think step by step.

            *CLEARLY FOLLOW THE QUESTION INSTRUCTION ABOUT LENGTH OF RESPONSE*
            {chat_history}
            """

        for tag in session_tags:
            tag_id = tag["id"]
            if tag_id in document_dictionary.keys():
                prompt_text += document_dictionary[tag_id]
            else:
                print(f"Tag: {tag}")
                prompt_text += get_content_from_tags(tag)

        prompt = ChatPromptTemplate.from_messages(
            [("system", prompt_text), ("human", "{user_question}")]
        )

    elif len(session_tags) > 0 and len(files) > 0:
        prompt_text = """
            **Please first read the document, If you can give answer using only this document, please provide the answer using document content. If the answer is not belong from the document please use your own knowldge to answer the question. Take History as context.**
            When you respond about any questions, you need to think about the company from multiple perspectives, including what the company is doing and what is most important for the company.
            Each answer must contain: Introduction, Approach, and Conclusion.
            Approach should be comprehensive and detail every perspective of the given company.
            In Approach, describe each point in very detailed paragraphs.
            Always take a breath and think step by step.

            *CLEARLY FOLLOW THE QUESTION INSTRUCTION ABOUT LENGTH OF RESPONSE*
            {chat_history}
            """
        print(f"Session Tags: {session_tags}")

        for tag in session_tags:
            if tag in document_dictionary.keys():
                prompt_text += document_dictionary[tag]
            else:
                prompt_text += get_content_from_tags(f"tags/{tag.displayName}")
        prompt_text += get_content_from_azure_blob(files)

        prompt = ChatPromptTemplate.from_messages(
            [("system", prompt_text), ("human", "{user_question}")]
        )

    elif len(session_tags) == 0 and len(files) > 0:
        prompt_text = """
            Please firest read the document, If you can give answer using only this document, please provide the answer using document content. If the answer is not belong from the document please use your own knowldge to answer the question. Take History as context.

            *CLEARLY FOLLOW THE QUESTION INSTRUCTION ABOUT LENGTH OF RESPONSE*
            {chat_history}
            """

        prompt_text += get_content_from_azure_blob(files)

        prompt = ChatPromptTemplate.from_messages(
            [("system", prompt_text), ("human", "{user_question}")]
        )

    elif model_name == "title":

        prompt_text = """
            Generate Chat title we see in ChatGPT of 2 to 3 words based on Chat History

            {chat_history}
            """
        prompt = ChatPromptTemplate.from_messages([("system", prompt_text)])

    return prompt

def clean_response(response):
    cleaned_response = re.sub(r"\b\w{26}_", "", response)  # Assumes the ID is 26 characters long followed by an underscore
    return cleaned_response

def get_history_question(item):
    user_question = ""
    memory = ConversationBufferMemory(memory_key="chat_history")
    chat_history = []

    last_files = []

    for elem in item.messages:
        if elem["role"] == "user":
            memory.chat_memory.add_user_message(elem["content"])
            user_question = elem["content"]
            chat_history.append(HumanMessage(content=elem["content"]))
            if "files" in elem.keys() and len(elem["files"]) > 0:
                last_files = elem["files"]
        if elem["role"] == "assistant":
            memory.chat_memory.add_ai_message(elem["content"])
            chat_history.append(AIMessage(content=elem["content"]))

    all_tags = []

    try:
        if item.tags:
            for i in item.tags:
                all_tags.append({"id": i["id"], "displayName": i["displayName"]})
    except:
        pass
    return (memory, chat_history, user_question, last_files, all_tags)
