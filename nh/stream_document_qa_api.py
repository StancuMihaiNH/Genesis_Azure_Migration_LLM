import random
from urllib.parse import quote

from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import PGVector

from src.helper import *
from common.key_vault_manager import KeyVaultManager
from utils.constants import KeyVaultSecretKeys

# Initialize key vault manager
key_vault_manager = KeyVaultManager.getInstance()

# Retrieve secrets
openai_key = key_vault_manager.get_secret(KeyVaultSecretKeys.OPENAI_API_KEY)
OPENAI_API_KEY_LIST = key_vault_manager.get_secret(KeyVaultSecretKeys.OPENAI_API_KEY_LIST).split(",")
db_name = "postgres"
db_user = key_vault_manager.get_secret(KeyVaultSecretKeys.POSTGRES_USER)
db_password = key_vault_manager.get_secret(KeyVaultSecretKeys.POSTGRES_PASSWORD)
db_host = key_vault_manager.get_secret(KeyVaultSecretKeys.POSTGRES_HOST)
db_port = key_vault_manager.get_secret(KeyVaultSecretKeys.POSTGRES_PORT)

# Set environment variable for OpenAI API Key
os.environ["OPENAI_API_KEY"] = random.choice(OPENAI_API_KEY_LIST)

# Initialize embedding
embedding = OpenAIEmbeddings(api_key=openai_key)

# Connection string for PostgreSQL
CONNECTION_STRING = f"postgresql+psycopg2://{db_user}:{quote(db_password)}@{db_host}:{db_port}/{db_name}"
COLLECTION_NAME = "poc_v0"

# Initialize PGVector
nh_knowldge_store = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embedding,
)

# Initialize retriever from knowledge store
nh_knowldge_retriever = nh_knowldge_store.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.75}
)

# Initialize ChatOpenAI
llm = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"], temperature=0.1, model_name="gpt-4")

# Initialize MultiQueryRetriever
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=nh_knowldge_retriever, llm=llm
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
contextualize_q_chain = (contextualize_q_prompt | llm | StrOutputParser()).with_config(
    tags=["contextualize_q_chain"]
)

def get_sourced_documents(session_user_question, chat_history):

    # First, get the source documents
    contextualized_question = contextualize_q_chain.invoke(
        {"question": session_user_question, "chat_history": chat_history}
    )
    source_docs = retriever_from_llm.invoke(contextualized_question)
    formatted_docs = format_docs(source_docs)

    return contextualized_question, source_docs, formatted_docs