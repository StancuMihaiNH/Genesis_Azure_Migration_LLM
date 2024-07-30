import os
import chardet
import psycopg2
from azure.storage.blob import BlobServiceClient
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import threading
from urllib.parse import quote

from psycopg2 import sql


class KeyVaultManager:
    _instance = None
    _lock = threading.Lock()
    _key_vault_name = "nh-aicoe-kv"

    def __new__(cls, *args, **kwargs):
        raise NotImplementedError("Use getInstance() method to get an instance.")

    @classmethod
    def getInstance(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(KeyVaultManager, cls).__new__(cls)
                    cls._instance._init(*args, **kwargs)
        return cls._instance

    def _init(self):
        self.key_vault_url = f"https://{self._key_vault_name}.vault.azure.net/"
        self.credential = DefaultAzureCredential()
        self.client = SecretClient(
            vault_url=self.key_vault_url, credential=self.credential
        )

    def get_secret(self, secret_name):
        return self.client.get_secret(secret_name).value


key_vault_manager = KeyVaultManager.getInstance()

db_name = "postgres"
db_user = key_vault_manager.get_secret("dbUser")
db_password = key_vault_manager.get_secret("dbPassword")
db_password_encoded = quote(db_password)
db_host = key_vault_manager.get_secret("dbHost")
db_port = key_vault_manager.get_secret("dbPort")
openAiKey = key_vault_manager.get_secret("openAIAPIKey")
connection_string = key_vault_manager.get_secret(
    "azureStorageContainerConnectionString"
)
        
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client("ai-coe-llm")
os.environ["OPENAI_API_KEY"] = openAiKey


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


try:
    conn = psycopg2.connect(
        dbname=db_name, user=db_user, password=db_password, host=db_host, port=db_port
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

CONNECTION_STRING = f"postgresql+psycopg2://{db_user}:{db_password_encoded}@{db_host}:{db_port}/{db_name}"
COLLECTION_NAME = "poc_v0"

embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])


def get_blob_files(container_client, prefix):
    blob_list = container_client.list_blobs(name_starts_with=prefix)
    return [blob.name for blob in blob_list]


def try_decode(data, encodings):
    for encoding in encodings:
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("Unable to decode data with provided encodings.")


def load_and_split_documents_from_blob(container_client, blob_name):
    blob_client = container_client.get_blob_client(blob_name)
    raw_data = blob_client.download_blob().readall()

    # Detect encoding
    result = chardet.detect(raw_data)
    encoding = result["encoding"]
    fallback_encodings = ["utf-8", "latin1", "iso-8859-1", "windows-1252"]

    document_text, used_encoding = try_decode(
        raw_data, [encoding] + fallback_encodings if encoding else fallback_encodings
    )

    doc = Document(page_content=document_text, metadata={"file_name": blob_name})
    if len(doc.page_content) > 100:
        doc_text = doc.page_content.replace("\x00", "")
        list_doc = [Document(page_content=doc_text, metadata=doc.metadata)]
    else:
        list_doc = []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(list_doc)
    return [split for split in splits if len(split.page_content) > 10]


blob_prefixes = ["north-highland/text/raw/"]
blob_keys = []
for prefix in blob_prefixes:
    blob_keys.extend(get_blob_files(container_client, prefix))

print(len(blob_keys))
intial_splits = []
for key in blob_keys[:5]:
    print(key)
    splits = load_and_split_documents_from_blob(container_client, key)
    if splits is not None:
        intial_splits.extend(splits)

print("Intial Splitting Done")
print(len(intial_splits))

try:
    vectorstore = PGVector.from_documents(
        embedding=embeddings,
        documents=intial_splits,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
except:
    vectorstore = PGVector.from_documents(
        embedding=embeddings,
        documents=intial_splits,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=False,
    )

for key in blob_keys[5:]:
    final_splits = []
    splits = load_and_split_documents_from_blob(container_client, key)
    if splits is not None:
        final_splits.extend(splits)
        print(f"Processing {key} documents")
        vectorstore.add_documents(final_splits)
