from pymongo import MongoClient
from config import MONGODB_URI, MONGODB_DATABASE, MONGO_COLLECTION

def load_data():
    """
    Load data from MongoDB.

    Returns:
        list: A list of documents from the MongoDB collection.
    """
    client = MongoClient(MONGODB_URI)
    db = client[f"{MONGODB_DATABASE}"]
    collection = db[f"{MONGO_COLLECTION}"]

    # Fetch all documents from the collection
    documents = list(collection.find())
    
    # Close the connection
    client.close()
    
    return documents
