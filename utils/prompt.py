from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = """

    You are a helpful assistant. 

    Previous conversation:
    {chat_history}


    User question: {user_question}
    AI Resonse:"""


qa_system_prompt = """You are an North Highland assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know.
If context is empty, Please mentioned that I do not know the answer for the question

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
