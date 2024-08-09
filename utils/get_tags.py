import os
import re
import uuid

from llama_index.core import SimpleDirectoryReader

from common.azure_blob_client_manager import AzureBlobClientManager
from common.key_vault_manager import KeyVaultManager
from utils.constants import KeyVaultSecretKeys

key_vault_manager = KeyVaultManager.getInstance()
connection_string = key_vault_manager.get_secret(
    KeyVaultSecretKeys.AZURE_STORAGE_CONTAINER_CONNECTION_STRING
)

container_client = AzureBlobClientManager.getInstance().get_container_client()


def get_blob_files(container_client, prefix):
    blob_list = container_client.list_blobs(name_starts_with=prefix)
    return [blob.name for blob in blob_list]


def get_content_from_tags(prefix):
    display_name = prefix["displayName"]
    prefix = f"tags/{display_name}"
    print(f"Getting content from Azure Blob Storage with prefix: {prefix}")
    list_files = get_blob_files(container_client, prefix)

    folder_name = str(uuid.uuid4())
    os.makedirs(f"/tmp/{folder_name}")

    for file_object in list_files:
        file_name = str(file_object.split("/")[-1])
        print(f"Downloading file {file_object} ...")
        blob_client = container_client.get_blob_client(file_object)
        raw_data = blob_client.download_blob().readall()
        with open(f"/tmp/{folder_name}/{file_name}", "wb") as file:
            file.write(raw_data)

    documents = SimpleDirectoryReader(f"/tmp/{folder_name}").load_data()

    text = ""
    for doc in documents:
        cleaned_file_name =  re.sub(r"\b\w{26}_", "", doc.metadata["file_name"])
        text += cleaned_file_name + "\n"
        text += doc.text.replace("\x00", "") + "\n\n"

    return text
