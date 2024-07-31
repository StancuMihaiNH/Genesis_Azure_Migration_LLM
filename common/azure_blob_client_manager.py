from azure.storage.blob import BlobServiceClient

from common.key_vault_manager import KeyVaultManager
from utils.constants import Constants, KeyVaultSecretKeys


class AzureBlobClientManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            if not cls._instance:
                cls._instance = super(AzureBlobClientManager, cls).__new__(cls)
                cls._instance._init(*args, **kwargs)
        return cls._instance

    def _init(self):
        key_vault_manager = KeyVaultManager.getInstance()
        connection_string = key_vault_manager.get_secret(
            KeyVaultSecretKeys.AZURE_STORAGE_CONTAINER_CONNECTION_STRING
        )
        self.blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )
        self.container_client = self.blob_service_client.get_container_client(
            Constants.LLM_CONTAINER_NAME
        )

    @classmethod
    def getInstance(cls, *args, **kwargs):
        if not cls._instance:
            if not cls._instance:
                cls._instance = cls(*args, **kwargs)
        return cls._instance

    def get_container_client(self):
        return self.container_client
