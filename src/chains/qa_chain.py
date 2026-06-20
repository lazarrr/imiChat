# src/chains/qa_chain.py
from collections.abc import AsyncGenerator
from unittest import result

from langchain_core.prompts import PromptTemplate

class UniversityQASystem:
    def __init__(self, llm_manager, vector_store, translator):
        self.llm_manager = llm_manager
        self.vector_store = vector_store
        self.translator = translator
        
        self.prompt_template = PromptTemplate(
            template="""
            Vi ste korisni univerzitetski asistent. Odgovarajte na pitanja na osnovu priloženog konteksta.

            Kontekst: {context}

            Pitanje: {question}

            Uputstva:
            1. Odgovarajte isključivo na srpskom latinskom pismu
            2. Pružite tačne informacije iz konteksta
            3. Ako informacija nije u kontekstu, jasno to navedite
            4. Navedite relevantne izvore kada je moguće

            Odgovor: """,
            input_variables=["context", "question"]
        )
        
    async def query_stream(self, question: str) -> AsyncGenerator[str, None]:
        """Stream response token by token"""
        
        # Step 1: Retrieve relevant documents (can be done first)
       # docs = self.vector_store.similarity_search(question)
        
        # Step 2: Stream tokens from LLM
        async for token in self.llm_manager.generate_stream(question):
            yield token
    
    def query(self, question: str, k: int = 5) -> dict:
        # Translate if needed
        normalized_question = self.translator.normalize_query(question)
        
        # Retrieve relevant documents
        results = self.vector_store.similarity_search(question, k=k)
   
        # Prepare context
        context = "\n\n".join(results['documents'][0])
        
        # Generate response
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )

        
        response = self.llm_manager.generate(prompt)
        
        return {
            "question": question,
            "answer": response,
            "sources": results['metadatas'][0] if results['metadatas'] else [],
            #"confidence": 1 - results['distances'][0][0] if results['distances'] and len(results['distances'][0]) > 0 else 0
        }