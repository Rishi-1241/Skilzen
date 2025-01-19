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
from pathlib import Path
from google.cloud import firestore
from groq import Groq
import config 
from google.oauth2 import service_account
from langchain_openai.embeddings import OpenAIEmbeddings
import time
from langchain.schema import Document
from langchain_google_firestore import FirestoreVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()
from langchain.document_loaders import TextLoader
# Initialize Firestore
from firebase_admin import credentials, firestore, initialize_app
cred = credentials.Certificate('utils/certain-math-447716-d1-firebase-adminsdk-zm6q4-c5fefb392a.json')
initialize_app(cred)
db = firestore.client()
import openai
# Initialize OpenAI (make sure to set your API key)
os.environ["OPENAI_API_KEY"] = "sk-c-cMt0Ej5AsKuL_rdID66WfxbOO8En9Mk-uhsTBLJCT3BlbkFJFyrLuvPvLUDbhRGZX5Lyrl4nx1LAXnbCECV9bQagEA"

import re

# Define the file path
file_path_pointwise = "../dataset/commerce.txt"

from typing import List

def load_file(file_path: str):
    """
    Load a text file using TextLoader.
    """
    try:
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        print(f"Successfully loaded the file: {file_path}")
        return documents
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return None
    

def split_recursive(documents: List[str], chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Split documents using RecursiveCharacterTextSplitter.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strip_whitespace=True,
            add_start_index=True,
        )
        split_docs = text_splitter.split_documents(documents)
        print(f"Split file into {len(split_docs)} chunks")
        return split_docs
    except Exception as e:
        print(f"Error splitting text: {str(e)}")
        return None
def split_chunks_pointwise(file_path):
    """
    Reads a file, splits its content into chunks based on numbering, 
    and returns the chunks as a list of dictionaries.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Split the content into chunks based on numbering
    chunks = re.split(r"\d+\.\s", content)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    # Convert chunks into a list of dictionaries with unique IDs
    return chunks


# List of specific files to process




def store_in_firestore(split_docs: List[Document], embeddings, collection_name: str, db):
    """
    Store split documents in Firestore with embeddings.
    """
    try:
        vector_store = FirestoreVectorStore.from_documents(
            documents=split_docs,
            embedding=embeddings,
            collection=collection_name,
            client=db
        )
        print(f"Successfully stored all chunks in Firestore collection: {collection_name}")
        return vector_store
    except Exception as e:
        print(f"Error storing in Firestore: {str(e)}")
        return None


def process_dataset_files(collection_name: str, file_path: str):
    """
    Main function to process a text file and store chunks in Firestore.
    """
    # Load the document
    documents = load_file(file_path)
    if documents is None:
        return None

    # Split the documents
    split_docs = split_recursive(documents)
    if split_docs is None:
        return None

    # Initialize embeddings
    embeddings = OpenAIEmbeddings()

    # Store in Firestore
    return store_in_firestore(split_docs, embeddings, collection_name, db)


if __name__ == "__main__":
    collection_name = "skilzen-recursive"
    file_path_pointwise = "../dataset/commerce.txt"

class Document:
    def __init__(self, page_content, metadata=None, doc_id=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}
        self.id = doc_id 

    

# files_to_process = ["../dataset/commerce.txt", "../dataset/management.txt"]
# all_chunks = []
# for file_path in files_to_process:
#     file_path = Path(file_path)
#     if file_path.exists():
#         print(f"Processing file: {file_path.name}")
#         file_chunks = split_chunks_pointwise(file_path)
#         # Add a unique ID (for example, using file name and chunk index)
#         all_chunks.extend([Document(chunk, doc_id=f"{file_path.stem}_{i}") for i, chunk in enumerate(file_chunks)])
#     else:
#         print(f"File not found: {file_path}")

# print(f"Total chunks processed: {len(all_chunks)}")

# # Output the chunks
# for i, chunk in enumerate(all_chunks):
#     print(f"Chunk {i}:\n{chunk.page_content}\n")

# print("Chunks created successfully")


embeddings = OpenAIEmbeddings()
#store_in_firestore(all_chunks, embeddings, "skilzen-recursive", db)

# vector_store = process_dataset_files(
#         collection_name=collection_name
#     )

    # Test the vector store
# if vector_store:
#     test_query = "what are placement stats"
#     results = vector_store.similarity_search(
#         query=test_query,
#         k=2
#     )
#     print("\nTest Search Results:")
#     for i, doc in enumerate(results, 1):
#         print(f"\nResult {i}:")
#         print(f"Content: {doc.page_content[:200]}...")

vectorstore = FirestoreVectorStore(
            collection='skilzen-recursive',
            embedding_service=embeddings,
        )


if vectorstore:
    starttime = time.time()
    test_query = "i want to know about B sc economics"
    results = vectorstore.similarity_search(
        query=test_query,
        k=2
    )
    endtime = time.time()
    print(f"Time taken: {endtime-starttime}")
    print("\nTest Search Results:")
    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Content: {doc.page_content}")


    context = " ".join([doc.page_content for doc in results])

prompt = f"""
Aap ek university ke admission counter par baith kar students ke questions ka jawab dene wale ek experienced employee hain. Aapke paas university ke sabhi courses, admission process aur dusri zaroori information ka accha gyaan hai. Aapko apne jawab simple, short, aur easily samajhne layak dene hain. Agar koi term zyada common hai jaise 'courses' ya 'admission', toh wo English mein rahe.

Aapko answer short dena hai, taaki vo easily bola ja sake. Agar student aur details chahe, toh aap unhe bata sakte hain ki wo aur information ke liye pooch sakte hain.

Nimnlikhit content ke adhar par, kripya is question ka jawab Hinglish mein dein:

Context (in English): {context}

Question (in English): {test_query}

Answer (in Hinglish):
"""


def _generate_groq_response(api_key, chat_history):
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=config.GROQ_LLM,
        messages=chat_history
    )
    return response.choices[0].message.content


# Call OpenAI's GPT model to generate an answer
messages = [{"role": "user", "content": prompt}]
model = "gpt-4o-mini"
response = openai.chat.completions.create(
    model=model,
    messages=messages,
    temperature=0.5,
)
print("\n\n\n Answer:")
print(response.choices[0].message.content.strip())
