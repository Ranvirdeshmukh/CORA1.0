from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import re


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


# Load the vector database
def load_vector_db():
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_key)
    
    return Chroma(
        persist_directory=VECTOR_DB_DIR, 
        embedding_function=embedding_model
    )


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


# if __name__ == "__main__":
#     # 1) Get vector_db and chat_model
#     vector_db, chat_model = initialize_chatbot()

#     print("Chatbot is ready! Type your query below:")
#     while True:
#         query = input("Enter your query: ")
#         if query.lower() in ['exit', 'quit']:
#             print("Exiting chatbot. Goodbye!")
#             break

#         # 2) Parse the user query for metadata filters
#         filters = parse_filters_from_query(query)
#         # Updated code snippet with MMR

#         if filters:
#             retriever = vector_db.as_retriever(
#                 search_type="mmr",
#                 search_kwargs={
#                     "k": 3,
#                     "filter": filters
#                 }
#             )
#             print(f"[DEBUG] Applying filters: {filters}")
#         else:
#             retriever = vector_db.as_retriever(
#                 search_type="mmr",
#                 search_kwargs={
#                     "k": 3
#                 }
#             )

#         # if filters:
#         #     # If there are filters, apply them
#         #     retriever = vector_db.as_retriever(
#         #         search_kwargs={"k": 3, "filter": filters}
#         #     )
#         #     print(f"[DEBUG] Applying filters: {filters}")
#         # else:
#         #     # Otherwise, do a normal retrieval
#         #     retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        
#         # qa_chain = RetrievalQA.from_chain_type(
#         #     llm=chat_model,
#         #     retriever=retriever
#         # )

#         # result = qa_chain({"query": query, "return_source_documents": True})
#         # answer = result["result"]
#         # source_docs = result["source_documents"]


#         #             # 3) Build a RetrievalQA chain on the fly using the chosen retriever
#         # qa_chain = RetrievalQA.from_chain_type(
#         #     llm=chat_model,
#         #     retriever=retriever,
#         #     return_source_documents=True  # <-- IMPORTANT: set here
#         # )
#         from langchain.prompts.chat import (
#             ChatPromptTemplate,
#             SystemMessagePromptTemplate,
#             HumanMessagePromptTemplate
#         )

#         system_template = """
#         You are CORA 1.0, a helpful college course advisor. 
#         - If asked your name, reply: "I am CORA 1.0—your college advisor."
#         - If asked something unrelated to your data, say you don't have info.
#         - If asked about courses or professors, answer with the data.
#         """
#         human_template = "{question}"

#         prompt = ChatPromptTemplate.from_messages([
#             SystemMessagePromptTemplate.from_template(system_template),
#             HumanMessagePromptTemplate.from_template(human_template)
#         ])


#         # 4) Call the chain WITHOUT return_source_documents parameter
#         result = qa_chain({"query": query})

#         answer = result["result"]             # The LLM answer
#         source_docs = result["source_documents"]  # The retrieved chunks

#         # Print the final answer
#         print(f"Answer: {answer}")

#         # Print source citations from metadata
#         print("Sources:")
#         for doc in source_docs:
#             prof = doc.metadata.get("professor", "N/A")
#             term = doc.metadata.get("term", "N/A")
#             course_name = doc.metadata.get("course_name", "N/A")
#             print(f"  - Professor: {prof}, Term: {term}, Course Name: {course_name}")
            
            
