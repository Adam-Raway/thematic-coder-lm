import json
import re
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()

from langchain_ollama import OllamaLLM as LangchainOllama
from langchain_openai import ChatOpenAI

class AbstractLLM(ABC):
    def __init__(self, model_name: str, temperature: float = 0.7, max_tokens: int = 1024):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None  # To be initialized by subclasses

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    def clean_and_parse_json(self, response: str):
        """Cleans markdown and extracts JSON content robustly."""
        response = response.strip()
        response = re.sub(r"^```(?:json)?\s*", "", response, flags=re.IGNORECASE)
        response = re.sub(r"\s*```$", "", response)

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            match = re.search(r"\{.*\}", response, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError(f"LLM did not return valid JSON: {response}") from e

    def generate_json(self, prompt: str, schema: dict) -> dict:
        # Optionally add a JSON validation layer here
        response = self.generate(prompt).strip()

        parsed = self.clean_and_parse_json(response)

        # optional: validate against schema here
        return parsed


    @staticmethod
    def from_name(model_name: str, **kwargs) -> 'AbstractLLM':
        if "gpt" in model_name.lower():
            return OpenAILLM(model_name, **kwargs)
        elif "qwen" in model_name.lower() or "llama" in model_name.lower():
            return OllamaLLM(model_name, **kwargs)
        else:
            raise ValueError(f"Unknown model: {model_name}")

class OllamaLLM(AbstractLLM):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.llm = LangchainOllama(model=model_name, temperature=self.temperature)

    def generate(self, prompt: str) -> str:
        return self.llm.invoke(prompt)


class OpenAILLM(AbstractLLM):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

    def generate(self, prompt: str) -> str:
        return self.llm.invoke(prompt).content
    
if __name__ == "__main__":
    # Example usage
    llm = AbstractLLM.from_name("gpt-4o-mini", temperature=0.5)
    response = llm.generate("Hello, how are you? What are you?")
    print(response)

    llm = AbstractLLM.from_name("qwen3:4b", temperature=0.5)
    response = llm.generate("Hello, how are you?")
    print(response)