import threading

from azure.identity import DefaultAzureCredential
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
        self.key_vault_url = f"https://{self._key_vault_name}.vault.azure.net/"
        self.credential = DefaultAzureCredential()
        self.client = SecretClient(
            vault_url=self.key_vault_url, credential=self.credential
        )

    def get_secret(self, secret_name):
        return self.client.get_secret(secret_name).value
