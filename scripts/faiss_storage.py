import os
import shutil
import zipfile
from google.cloud import storage
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

openai_key = os.getenv("OPENAI_API_KEY")
BUCKET_NAME = "cora1faiss"

def save_faiss_to_gcs(faiss_vectorstore):
    # Save FAISS index locally to a temporary directory
    local_dir = "/tmp/faiss_index_dir"
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    # Save the vectorstore locally (this creates the necessary files in local_dir)
    faiss_vectorstore.save_local(local_dir)
    
    # Zip the directory into a single file
    zip_path = "/tmp/faiss_index.zip"
    shutil.make_archive("/tmp/faiss_index", 'zip', local_dir)
    
    # Upload the zip file to GCS with an increased timeout
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob("faiss_index.zip")
    blob.upload_from_filename(zip_path, timeout=300)  # 300-second timeout
    print("FAISS index saved to GCS as zip.")

def load_faiss_from_gcs():
    local_dir = "/tmp/faiss_index_dir"
    # If a local copy exists, load it directly
    if os.path.exists(local_dir):
        print("Loading FAISS index from local directory.")
        embedding_model = OpenAIEmbeddings(openai_api_key=openai_key)
        return FAISS.load_local(local_dir, embedding_model, allow_dangerous_deserialization=True)
    
    # Otherwise, download the zip file from GCS
    zip_path = "/tmp/faiss_index.zip"
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob("faiss_index.zip")
    blob.download_to_filename(zip_path)
    
    # Extract the zip to a temporary directory
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    os.makedirs(local_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(local_dir)
    
    # Load the vectorstore from the local directory.
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_key)
    vectorstore = FAISS.load_local(local_dir, embedding_model, allow_dangerous_deserialization=True)
    print("FAISS index loaded from GCS zip.")
    return vectorstore
