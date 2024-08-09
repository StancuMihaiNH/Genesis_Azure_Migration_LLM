import os
import threading
from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient

from utils.constants import Constants


class KeyVaultManager:
    _instance = None
    _lock = threading.Lock()
    _key_vault_name = Constants.KEY_VAULT_NAME

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
        key_vault_url = f"https://{self._key_vault_name}.vault.azure.net/"
        client_id = os.getenv('AZURE_CLIENT_ID')
        tenant_id = os.getenv('AZURE_TENANT_ID')
        client_secret = os.getenv('AZURE_CLIENT_SECRET')
       
        credential = ClientSecretCredential(tenant_id, client_id, client_secret)
        self.client = SecretClient(
            vault_url=key_vault_url, credential=credential
        )

    def get_secret(self, secret_name):
        return self.client.get_secret(secret_name).value