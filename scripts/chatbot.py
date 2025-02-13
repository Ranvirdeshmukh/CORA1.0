from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# New import (using FAISS)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import csv
import re
# In chatbot.py
from faiss_storage import save_faiss_to_gcs, load_faiss_from_gcs




load_dotenv()  # Load .env variables
openai_key = os.getenv("OPENAI_API_KEY")


VECTOR_DB_DIR = "./vector_db"

def parse_filters_from_query(query: str) -> dict:
    """
    Very simple parser that looks for patterns like:
      'professor = some name'
      'department = CHEM'
      'term = 06S'
    Returns a dict for filtering, e.g. {"professor": "John Winn"}.
    """
    filters = {}

    # Example: look for 'professor = John Winn'
    prof_match = re.search(r'professor\s*=\s*([A-Za-z\s\.]+)', query, re.IGNORECASE)
    if prof_match:
        filters["professor"] = prof_match.group(1).strip()

    # Example: look for 'department = CHEM'
    dept_match = re.search(r'department\s*=\s*([A-Za-z\s\.]+)', query, re.IGNORECASE)
    if dept_match:
        filters["department"] = dept_match.group(1).strip()

    # Example: look for 'term = 07F'
    term_match = re.search(r'term\s*=\s*([A-Za-z0-9]+)', query, re.IGNORECASE)
    if term_match:
        filters["term"] = term_match.group(1).strip()

    return filters


def load_vector_db():
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_key)
    try:
        faiss_index = load_faiss_from_gcs()  # See below for function implementation.
    except Exception as e:
        print("Could not load FAISS index from GCS; building a new index. Error:", e)
        documents = load_documents()  # <-- Implement this function.
        faiss_index = FAISS.from_documents(documents, embedding_model)
        save_faiss_to_gcs(faiss_index)  # See below.
    return faiss_index

def load_documents():
    """
    Loads documents from the CSV file and returns a list of Document objects.
    Adjust the columns used for the page content and metadata as needed.
    """
    documents = []
    csv_file_path = "data/quiver_export_embeddingObjects20241224235214 (1).csv"
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # For the main content, you could use the "Review" column (or any other column you prefer)
            page_content = row.get("Review", "")
            # Use selected fields from the CSV as metadata
            metadata = {
                "Title": row.get("Title", ""),
                "Class": row.get("Class", ""),
                "Class Num": row.get("Class Num", ""),
                "Course Name": row.get("Course Name", ""),
                "Department": row.get("Department", ""),
                "Professor": row.get("Professor", ""),
                "Term": row.get("Term", "")
            }
            documents.append(Document(page_content=page_content, metadata=metadata))
    return documents

def initialize_chatbot():
    vector_db = load_vector_db()
    
    chat_model = ChatOpenAI(
        openai_api_key=openai_key,
        model="gpt-3.5-turbo"
    )

    return vector_db, chat_model



if __name__ == "__main__":
    vector_db, chat_model = initialize_chatbot()
    print("Chatbot is ready! Type your query below:")

    while True:
        query = input("Enter your query: ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Goodbye!")
            break

        # 1) Parse filters
        filters = parse_filters_from_query(query)
        if filters:
            retriever = vector_db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3, "filter": filters}
            )
            print(f"[DEBUG] Applying filters: {filters}")
        else:
            retriever = vector_db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3}
            )

        # 2) Define the "stuff" prompt
        from langchain.prompts import PromptTemplate
        stuff_template = """You are CORA 1.0, a helpful college course advisor.

- If asked your name, reply: "I am CORA 1.0— your AI college advisor."
- If asked something unrelated to your data, say you don't have info.
- If asked about courses or professors, answer with the data.

Here is some context from your documents:
{context}

User question:
{question}

Answer:
"""
        stuff_prompt = PromptTemplate(
            template=stuff_template,
            input_variables=["context", "question"]
        )

        # 3) Build the QA chain with "stuff" type and pass the prompt
        qa_chain = RetrievalQA.from_chain_type(
            llm=chat_model,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": stuff_prompt
            }
        )

        # 4) Run the chain
        result = qa_chain.invoke({"query": query})
        answer = result["result"]
        source_docs = result["source_documents"]

        # 5) Print answer + sources
        print(f"Answer: {answer}")
        print("Sources:")
        for doc in source_docs:
            prof = doc.metadata.get("professor", "N/A")
            term = doc.metadata.get("term", "N/A")
            course_name = doc.metadata.get("course_name", "N/A")
            print(f"  - Professor: {prof}, Term: {term}, Course Name: {course_name}")


            
