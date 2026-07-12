from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

# Load and split the PDF
print("Loading PDF...")
loader = PyPDFLoader("ICH_E6(R3)_DraftGuideline_2023_0519.pdf")
pages = loader.load()
print(f"Loaded {len(pages)} pages")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(pages)
print(f"Created {len(chunks)} chunks")

# Store in Chroma
print("Building vector store...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db"
)
print("Done! Vector store saved to ./chroma_db")

# Test retrieval
query = "What are the responsibilities of the sponsor?"
results = vectorstore.similarity_search(query, k=3)
print(f"\nTop 3 results for: '{query}'\n")
for i, doc in enumerate(results):
    print(f"--- Chunk {i+1} ---")
    print(doc.page_content[:300])
    print()