# src/core/llm_manager.py
from langchain_community.llms import Ollama
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from typing import Generator, AsyncGenerator
import asyncio

class LLMManager:
    def __init__(self, config):
        self.config = config
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        if self.config.LLM_PROVIDER == "ollama":
            return Ollama(
                model=self.config.LLM_MODEL,
                temperature=self.config.LLM_TEMPERATURE,
                num_predict=self.config.LLM_MAX_TOKENS,
                callbacks=[StreamingStdOutCallbackHandler()],
            )
        # Add other providers as needed
    
    def generate(self, prompt: str) -> str:
        """Non-streaming generation"""
        return self.llm.invoke(prompt)
    
    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """
        Synchronous streaming generator
        Yields tokens one by one as they're generated
        """
        # Using langchain's streaming capability
        for chunk in self.llm.stream(prompt):
            yield chunk
    
    async def generate_stream_async(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Asynchronous streaming generator
        Yields tokens one by one as they're generated
        """
        # Convert synchronous stream to async
        for chunk in self.llm.stream(prompt):
            yield chunk
            await asyncio.sleep(0)  # Allow event loop to switch