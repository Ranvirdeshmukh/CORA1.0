import pandas as pd
from langchain.vectorstores import Chroma    # Correct
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings  # Correct


# File paths
CSV_FILE = "./data/quiver_export_embeddingObjects20241224235214 (1).csv"
VECTOR_DB_DIR = "./vector_db"

def process_data():
    # Load the CSV file
    try:
        data = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {CSV_FILE}")
        return

    # Ensure the necessary columns are present
    required_columns = ['Course Name', 'Professor', 'Review', 'Department', 'Term']
    for col in required_columns:
        if col not in data.columns:
            print(f"Error: CSV file must contain the column: {col}")
            return

    # Drop rows that have NaNs in these required columns
    data = data.dropna(subset=required_columns)

    # 1. Separate main text from metadata
    # -----------------------------------
    # We'll store the 'Review' in a "text" column,
    # and create a "metadata" column with key info.
    data['text'] = data['Review']

    # Use .get() in case some columns like "Quality" don't always exist
    # or if you want to add more columns later.
    data['metadata'] = data.apply(
        lambda row: {
            "course_name": row["Course Name"],
            "professor": row["Professor"],
            "department": row.get("Department", ""),
            "term": row.get("Term", ""),
            # Add more columns here if you want them in metadata
        },
        axis=1
    )

    # 2. Chunking long reviews
    # ------------------------
    # If reviews can be very long, we split them into smaller pieces.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Adjust as needed
        chunk_overlap=100 # Overlap between chunks, also adjustable
    )

    # We will store all the chunked Documents in this list
    all_documents = []

    # Iterate over each row and chunk the 'text' accordingly
    for idx, row in data.iterrows():
        review_text = row['text']

        # If 'review_text' isn’t a string for some reason, skip
        if not isinstance(review_text, str):
            continue

        # Split the text into chunks
        chunks = text_splitter.split_text(review_text)

        # For each chunk, create a Document with the original metadata
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata=row['metadata']  # Keep the same metadata for all chunks
            )
            all_documents.append(doc)

    # 3. Build the vector database with Documents
    # -------------------------------------------
    # Initialize OpenAI embeddings
    embedding_model = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Create a Chroma DB from the chunked Documents
    vector_db = Chroma.from_documents(
        documents=all_documents,
        embedding=embedding_model,
        persist_directory=VECTOR_DB_DIR
    )

    # Persist to disk
    # vector_db.persist()

    print(f"Vector database created and saved at: {VECTOR_DB_DIR}")

# Run the data processing if called directly
if __name__ == "__main__":
    process_data()
