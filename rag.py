from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import Ollama
import os
import gc
import shutil
import warnings
from langchain.globals import set_debug, set_verbose

# Disable LangSmith logging
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = "none"

# Disable warnings and debug logging
warnings.filterwarnings('ignore')
set_debug(False)
set_verbose(False)

# Initialize Mistral through Ollama
llm = Ollama(model="mistral", base_url="http://localhost:11434")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'}
)

# Set your documents directory and vector DB path
directory = "docs"
vector_db_path = "vector_db"

def cleanup_vector_db():
    """Clean up vector database files if they exist"""
    if os.path.exists(vector_db_path):
        try:
            shutil.rmtree(vector_db_path)
            print(f"Cleaned up vector database at {vector_db_path}")
        except Exception as e:
            print(f"Error cleaning up vector database: {e}")

def load_docs(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            try:
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
                print(f"Successfully loaded {filename}")
            except Exception as e:
                print(f"Error loading file {file_path}")
                print(f"Error: {e}")
    return documents

def split_docs(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = text_splitter.split_documents(documents)
    return docs

def create_or_load_vector_db(docs=None):
    """Create a new vector database or load existing one"""
    if os.path.exists(vector_db_path):
        print("Loading existing vector database...")
        return FAISS.load_local(vector_db_path, embeddings)
    
    if docs is None:
        raise ValueError("Documents required to create new vector database")
    
    print("Creating new vector database...")
    db = FAISS.from_documents(docs, embeddings)
    
    # Save the vector database
    print("Saving vector database...")
    db.save_local(vector_db_path)
    
    return db

def get_answer(query, db, chain):
    similar_docs = db.similarity_search(query, k=2)
    # Use invoke instead of run
    answer = chain.invoke({"input_documents": similar_docs, "question": query})
    return answer["output_text"]

def main():
    # Ask user if they want to use existing vector database
    use_existing = False
    if os.path.exists(vector_db_path):
        response = input("Found existing vector database. Use it? (y/n): ").lower()
        use_existing = response == 'y'
        
        if not use_existing:
            cleanup_vector_db()
    
    if use_existing:
        db = create_or_load_vector_db()
    else:
        print("Loading documents...")
        documents = load_docs(directory)
        
        print("Splitting documents...")
        docs = split_docs(documents)
        
        print("Creating vector database...")
        db = create_or_load_vector_db(docs)
        
        # Clear memory
        del documents
        del docs
        gc.collect()
    
    chain = load_qa_chain(llm, chain_type="stuff")
    
    print("\nPrivate Q&A chatbot ready!")
    
    try:
        while True:
            prompt = input("\nEnter your query (or 'quit' to exit): ")
            
            if prompt.lower() == 'quit':
                break
                
            if prompt:
                try:
                    answer = get_answer(prompt, db, chain)
                    print(f"\nAnswer: {answer}")
                except Exception as e:
                    print(f"An error occurred: {e}")
    finally:
        # Cleanup
        del db
        gc.collect()
        
        # Ask if user wants to keep the vector database
        keep_db = input("\nKeep vector database for next time? (y/n): ").lower() == 'y'
        if not keep_db:
            cleanup_vector_db()

if __name__ == "__main__":
    main()
    