# Before launching the application need to run this file (Only execute again when data is updated)

# File to save the embeddings (Need to run this again to update the embeddings with new files)

from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

# Setting the environment variable (Credentials)
load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Calling the three functions from src/helper.py
extracted_data=load_pdf_file(directory_path='Data/')
text_chunks=text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Pinecone Initialization

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "healthybot"

# Create the index

pc.create_index(
    name=index_name,
    dimension=384, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws", 
        region="us-east-1"
    ) 
) 

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)