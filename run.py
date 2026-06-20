# run.py
import logging
from src.core.vector_store import VectorStoreManager
from src.core.llm_manager import LLMManager
from src.chains.qa_chain import UniversityQASystem
from src.processing.translator import ScriptTranslator
from src.processing.document_loader import DocumentProcessor
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_system():
    """Initialize all components"""
    logger.info("Initializing system components...")
    
    # Initialize components
    translator = ScriptTranslator()
    llm_manager = LLMManager(settings)
    vector_store = VectorStoreManager(settings)
    
    # Process documents if needed
    doc_processor = DocumentProcessor(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    
    # Load and index documents
    documents = doc_processor.process_directory("./data/raw")
    print(f"Processed {len(documents)} documents.")
    vector_store.add_documents(documents)
    
    # Create QA system
    qa_system = UniversityQASystem(llm_manager, vector_store, translator)
    
    logger.info("System initialized successfully!")
    return qa_system

if __name__ == "__main__":
    qa_system = initialize_system()
    
    # Example usage
    while True:
        question = input("\nAsk a question (or 'quit'): ")
        if question.lower() == 'quit':
            break
        
        response = qa_system.query(question)
        #print(f"\nAnswer: {response['answer']}")
        #print(f"Confidence: {response['confidence']:.2f}")