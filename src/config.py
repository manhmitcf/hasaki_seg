import os
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

# MongoDB Configuration
MONGODB_USERNAME = os.getenv("MONGODB_USERNAME")
MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD")
MONGODB_HOST = os.getenv("MONGODB_HOST")
MONGODB_PORT = int(os.getenv("MONGODB_PORT", 27017))
MONGODB_AUTH_SOURCE = os.getenv("MONGODB_AUTH_SOURCE", "admin")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE")

# Tên collection
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")


# MongoDB URI
MONGODB_URI = (
    f"mongodb://{MONGODB_USERNAME}:{MONGODB_PASSWORD}"
    f"@{MONGODB_HOST}:{MONGODB_PORT}/?authSource={MONGODB_AUTH_SOURCE}"
)

# API PINECONE
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
