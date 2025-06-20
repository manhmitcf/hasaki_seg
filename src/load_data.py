
from sentence_transformers import SentenceTransformer
from db_loader import load_data
import json
import os
from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY
from embedding import process_products_with_chunking
def load_and_process_data():
    """
    Load product data, create chunks, generate embeddings, and store in MongoDB
    """
    
    # Load the sentence transformer model
    print("Loading sentence-transformer model...")
    model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")
    print("Model loaded successfully.")
    
    # Load product data
    # print("Loading product data from MongoDB...")
    # products = load_data()

    # print(f"Loaded {len(products)} products.")
    print("Loading product data...")
    with open('data/products_info.json', 'r', encoding='utf-8') as f:
        products = json.load(f)
    print(f"Loaded {len(products)} products.")
    
    # Process products with chunking
    print("Processing products with chunking...")
    # chunked_documents = process_products_with_chunking(
    #     products=products,
    #     model=model,
    #     chunk_size=300,  
    #     overlap=50      
    # )
    chunked_documents = process_products_with_chunking(
        products=products,
        model=model,
        chunk_size=300,  
        overlap=50      
    )


    print(f"Processed {len(chunked_documents)} chunked documents.")
    # Save chunked documents to a JSON file
    with open('data/chunked_documents.json', 'w', encoding='utf-8') as f:
        json.dump(chunked_documents, f, ensure_ascii=False, indent=4)
    print("Chunked documents saved to 'data/chunked_documents.json'.")

    # # Initialize Pinecone
    # pc = Pinecone(
    # api_key=os.environ.get(f"{PINECONE_API_KEY}"),  
    # )
    # index_name = "product-embeddings"

    
    # if index_name not in pc.list_indexes().names():
    #     pc.create_index(
    #         name=index_name,
    #         dimension=768,
    #         metric='cosine',
    #         spec=ServerlessSpec(
    #             cloud='aws',      
    #             region='us-east-1'  
    #         )
    #     )
    # print(f"Index '{index_name}' is ready.")
    # # Connect to the index
    # print(f"Connecting to index '{index_name}'...")
    # index = pc.Index(index_name)
    # print("Connected to index.")
    # # Prepare data for upsert
    # to_upsert = [(item["id"], item["values"], item["metadata"]) for item in chunked_documents]
    # # Upsert data into Pinecone
    # print("Upserting data into Pinecone...")
    # index.upsert(vectors=to_upsert)
    # print("Data upserted into Pinecone.")
    # print("Data processing and storage completed successfully.")


if __name__ == "__main__":
    load_and_process_data()