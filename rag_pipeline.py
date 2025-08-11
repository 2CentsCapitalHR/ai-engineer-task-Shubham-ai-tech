import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Define paths for the knowledge base and the vector store
DOCS_PATH = 'adgm_docs'
DB_FAISS_PATH = 'vectorstore'

def create_vector_db():
    """
    Creates a FAISS vector store from both PDF and DOCX documents.
    """
    # 1. Load documents from the directory
    print("Loading documents...")
    
    # Create a loader for PDF files
    pdf_loader = DirectoryLoader(DOCS_PATH, glob='**/*.pdf', loader_cls=PyPDFLoader, show_progress=True)
    
    # Create a loader for DOCX files
    docx_loader = DirectoryLoader(DOCS_PATH, glob='**/*.docx', loader_cls=UnstructuredWordDocumentLoader, show_progress=True)
    
    # Load documents from both loaders
    pdf_documents = pdf_loader.load()
    docx_documents = docx_loader.load()
    
    # Combine the loaded documents into one list
    documents = pdf_documents + docx_documents
    print(f"Loaded {len(documents)} total documents (.pdf and .docx).")

    # 2. Split the documents into smaller chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} text chunks.")

    # 3. Create embeddings for the chunks
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )

    # 4. Create a FAISS vector store from the chunks and embeddings
    print("Creating FAISS vector store...")
    db = FAISS.from_documents(texts, embeddings)

    # 5. Save the vector store locally
    db.save_local(DB_FAISS_PATH)
    print(f"Vector store created and saved at: {DB_FAISS_PATH}")

if __name__ == "__main__":
    create_vector_db()