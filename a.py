import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import requests

api_key = ""
system_prompt = ""
user_prompt = ""
temperature = 0


model = "gpt-4o-mini"  # Model
model_endpoint = "https://api.openai.com/v1/chat/completions"  # just one endpoint
headers = {
    "Content-Type": "application/json",  # Authorisation
    "Authorization": f"Bearer {api_key}",
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
