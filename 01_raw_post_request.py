import os
import json
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import requests
from rich.console import Console

console = Console()
load_dotenv(find_dotenv(), override=True)  # read local .env file
# get API key from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
system_prompt = "You are a helpgul assistant."
user_prompt = "Tell me about Brighton in 100 words or so."


MODEL = "gpt-4o-mini"
# just one endpoint and does not change based on request
model_endpoint = "https://api.openai.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",  # Authorisation
    "Authorization": f"Bearer {OPENAI_API_KEY}",
}
# payload structure may vary from LLM Organisation but it is a text string.
payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
}
# Use HTTP POST method - a web form in essence
response = requests.post(
    url=model_endpoint,  # The endpoint we are sending the request to.
    headers=headers,  # Headers for authentication etc
    data=json.dumps(payload),  # Inputs, Context, Instructions, Additonal parameters
).json()

console.print(response)
