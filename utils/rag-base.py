import firebase_admin
from firebase_admin import credentials, firestore
import openai
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
from langchain_google_firestore import FirestoreVectorStore
from dataclasses import dataclass
import os
import json
from dotenv import load_dotenv
from google.cloud import firestore
from google.oauth2 import service_account
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain_google_firestore import FirestoreVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Load environment variables from .env file
load_dotenv()

# Initialize Firestore
cred = credentials.Certificate('../certain-math-447716-d1-firebase-adminsdk-zm6q4-c5fefb392a.json')
firebase_admin.initialize_app(cred)


# Initialize OpenAI API with API key from environment variables
openai_key = os.getenv('OPENAI_API_KEY')
file_path = '../dataset/placement.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    file_content = file.read()

# Step 3: Initialize RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Maximum size of each chunk
    chunk_overlap=100,  # Overlap between chunks
)

# Step 4: Split the text into chunks
chunks = text_splitter.split_text(file_content)

# Step 5: Generate embeddings using OpenAI
embeddings = OpenAIEmbeddings()
import uuid  # For generating unique IDs

@dataclass
class Document:
    id: str  # Add an id attribute
    page_content: str
    metadata: dict  # Assuming metadata is a dictionary

# Create documents with unique IDs
documents = [
    Document(id=str(uuid.uuid4()), page_content=chunk, metadata={})
    for chunk in chunks
]

# Store chunks and embeddings in Firestore
vector_store = FirestoreVectorStore.from_documents(
    collection="skilzen",  # Firestore collection name
    documents=documents,
    embedding=embeddings
)

print(f"Successfully stored {len(documents)} text chunks in Firestore.")
