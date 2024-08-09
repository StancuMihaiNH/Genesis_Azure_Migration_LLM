class Constants:
    # DEV/Test Environment
    # DATABASE_NAME = "NHChat"
    # PARTITION_KEY_PATH = "/PK"
    # OFFER_THROUGHPUT = 400
    # KEY_VAULT_NAME = "nh-aicoe-kv"
    # LLM_CONTAINER_NAME = "ai-coe-llm"

    # PROD Environment
    DATABASE_NAME = "ai-coe"
    PARTITION_KEY_PATH = "/PK"
    OFFER_THROUGHPUT = 400
    KEY_VAULT_NAME = "ai-coe-prod-kv"
    LLM_CONTAINER_NAME = "ai-coe-llm"


class KeyVaultSecretKeys:
    ANTHROPIC_KEY_LIST = "anthropicAPIKeyList"
    AZURE_STORAGE_CONTAINER_CONNECTION_STRING = "azureStorageContainerConnectionString"
    COSMOSDBENDPOINT = "cosmosDbEndpoint"
    OPENAI_API_KEY = "openAIAPIKey"
    OPENAI_API_KEY_LIST = "openAIAPIKeyList"
    POSTGRES_HOST = "dbHost"
    POSTGRES_PORT = "dbPort"
    POSTGRES_USER = "dbUser"
    POSTGRES_PASSWORD = "dbPassword"
    POSTGRES_DB = "dbName"
