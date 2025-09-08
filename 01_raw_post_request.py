import os
import json
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import requests
from rich.console import Console

console = Console()
load_dotenv(find_dotenv(), override=True)  # read local .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # get API key from .env file
system_prompt = ""
user_prompt = "Tell me about Brighton in 100 words or so."
temperature = 0


model = "gpt-4o-mini"  # Model
model_endpoint = "https://api.openai.com/v1/chat/completions"  # just one endpoint
headers = {
    "Content-Type": "application/json",  # Authorisation
    "Authorization": f"Bearer {OPENAI_API_KEY}",
}
# payload structure may vary from LLM Organisation but it is a text string.
payload = {
    "model": model,
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    # additional parameters
    "stream": False,
    "temperature": temperature,
}
# Use HTTP POST method
response = requests.post(
    url=model_endpoint,  # The endpoint we are sending the request to. Low Temperature:
    headers=headers,  # Headers for authentication etc
    data=json.dumps(payload),  # Inputs, Context, Instructions, Additonal parameters
).json()

console.print(response)  # print the response
