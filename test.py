
from config import CurrentConfig
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import Client
import toml

import os
import toml

openai_api_key = CurrentConfig.OPENAI_API_KEY

# Auto-trace LLM calls in-context
client = wrap_openai(Client(api_key=openai_api_key))

@traceable
def query_gpt(prompt: str) -> str:
    """
    A function that queries the GPT model.
    
    Parameters:
        prompt (str): The initial system prompt.
        
    Returns:
        str: The model's response or an error message.
    """
    model = CurrentConfig.LLM_MODEL
    temperature = CurrentConfig.LLM_TEMPERATURE
    
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        temperature=temperature,
    )
    return completion.choices[0].message.content

    
if __name__ == "__main__":
    x = query_gpt("Hello, how are you today?")
    print(x)