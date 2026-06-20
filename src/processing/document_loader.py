# src/processing/document_loader.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
import os

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_document(self, file_path: str):
        """Load document based on file type"""
        ext = os.path.splitext(file_path)[1].lower()
        
        loaders = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader,
            '.md': UnstructuredMarkdownLoader
        }
        
        if ext not in loaders:
            raise ValueError(f"Unsupported file type: {ext}")
        
        loader = loaders[ext](file_path)
        return loader.load()
    
    def process_directory(self, directory_path: str):
        """Process all documents in directory"""
        all_documents = []
        
        if not os.path.exists(directory_path):
            print(f"Warning: Directory {directory_path} does not exist")
            return all_documents
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if not os.path.isfile(file_path):
                continue
            try:
                documents = self.load_document(file_path)
                chunks = self.text_splitter.split_documents(documents)
                all_documents.extend(chunks)
                print(f"Processed: {filename} - {len(chunks)} chunks")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        if not all_documents:
            print(f"Warning: No documents found in {directory_path}")
        return all_documents