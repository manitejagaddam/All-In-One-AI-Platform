from abc import ABC, abstractmethod

class BaseLLMConnector(ABC):
    
    @abstractmethod
    async def chat(self, messages : list[dict]) -> str:
        "Takes a dictionaries of messages (previous messages chated with the different models) and returns a string as reply form llm"
        
        pass
    
