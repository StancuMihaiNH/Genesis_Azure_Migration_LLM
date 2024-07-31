from urllib.parse import quote

import chardet
import psycopg2
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from psycopg2 import sql

from common.azure_blob_client_manager import AzureBlobClientManager
from common.key_vault_manager import KeyVaultManager
from utils.constants import KeyVaultSecretKeys

# Constants
BLOB_PREFIXES = ["north-highland/text/raw/"]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
COLLECTION_NAME = "poc_v0"
ENCODINGS = ["utf-8", "latin1", "iso-8859-1", "windows-1252"]

# Initialize key vault manager
key_vault_manager = KeyVaultManager.getInstance()

# Retrieve secrets
db_name = "postgres"
db_user = key_vault_manager.get_secret(KeyVaultSecretKeys.POSTGRES_USER)
db_password = key_vault_manager.get_secret(KeyVaultSecretKeys.POSTGRES_PASSWORD)
db_host = key_vault_manager.get_secret(KeyVaultSecretKeys.POSTGRES_HOST)
db_port = key_vault_manager.get_secret(KeyVaultSecretKeys.POSTGRES_PORT)
openai_key = key_vault_manager.get_secret(KeyVaultSecretKeys.OPENAI_API_KEY)

container_instance = AzureBlobClientManager.getInstance()
container_client = container_instance.get_container_client()


def connect_to_db():
    conn = None
    """Establishes a connection to the PostgreSQL database and adds the vector extension if it doesn't exist."""
    try:
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
        )
        print("Connected to the database")

        conn.autocommit = True
        cursor = conn.cursor()

        print("Checking and adding Vector extension if it does not exist")
        create_cmd = sql.SQL("CREATE EXTENSION IF NOT EXISTS vector")
        cursor.execute(create_cmd)
        print("Vector extension added successfully or already exists")

    except psycopg2.Error as e:
        print(f"An error occurred: {e}")

    finally:
        if conn:
            cursor.close()
            conn.close()
            print("Database connection closed")


def get_blob_files(container_client, prefix):
    """Returns a list of blob files from the container with the specified prefix."""
    blobs = container_client.list_blobs(name_starts_with=prefix)
    return [blob.name for blob in blobs]


def try_decode(data, encodings):
    """Tries to decode the data using a list of encodings."""
    for encoding in encodings:
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("Unable to decode data with provided encodings.")


def load_and_split_documents_from_blob(container_client, blob_name):
    """Loads and splits documents from a blob."""
    blob_client = container_client.get_blob_client(blob_name)
    raw_data = blob_client.download_blob().readall()

    result = chardet.detect(raw_data)
    encoding = result["encoding"]

    document_text, used_encoding = try_decode(
        raw_data, [encoding] + ENCODINGS if encoding else ENCODINGS
    )

    doc = Document(page_content=document_text, metadata={"file_name": blob_name})
    if len(doc.page_content) > 100:
        doc_text = doc.page_content.replace("\x00", "")
        list_doc = [Document(page_content=doc_text, metadata=doc.metadata)]
    else:
        list_doc = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(list_doc)
    return [split for split in splits if len(split.page_content) > 10]


connect_to_db()

embeddings = OpenAIEmbeddings(api_key=openai_key)

blob_keys = []
for prefix in BLOB_PREFIXES:
    blob_keys.extend(get_blob_files(container_client, prefix))
print(f"Total blobs found: {len(blob_keys)}")

initial_splits = []
for key in blob_keys[:5]:
    print(f"Processing initial blob: {key}")
    splits = load_and_split_documents_from_blob(container_client, key)
    if splits:
        initial_splits.extend(splits)
print("Initial Splitting Done")
print(f"Total initial splits: {len(initial_splits)}")

CONNECTION_STRING = f"postgresql+psycopg2://{db_user}:{quote(db_password)}@{db_host}:{db_port}/{db_name}"
try:
    vectorstore = PGVector.from_documents(
        embedding=embeddings,
        documents=initial_splits,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
except Exception as e:
    print(f"An error occurred while initializing the vector store: {e}")
    vectorstore = PGVector.from_documents(
        embedding=embeddings,
        documents=initial_splits,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=False,
    )

for key in blob_keys[5:]:
    final_splits = []
    splits = load_and_split_documents_from_blob(container_client, key)
    if splits:
        final_splits.extend(splits)
        print(f"Processing {key} documents")
        vectorstore.add_documents(final_splits)
