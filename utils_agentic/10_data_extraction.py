import pandas as pd
import json
from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv
from rich.console import Console

console = Console()

load_dotenv(find_dotenv(), override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
console.print(f"[dark_orange]Using OpenAI API key: {OPENAI_API_KEY[:12]}[/]")
# Set up OpenAI API key
client = OpenAI(api_key=OPENAI_API_KEY)


def get_first_value(dict_variable):
    """The function clean takes a dictionary (dict_variable) as input and returns the value of the first key-value pair in the dictionary.

    Here's a breakdown:

    iter(dict_variable.values()) creates an iterator over the dictionary's values.
    next(...) retrieves the first value from the iterator.
    So, if you have a dictionary like {'a': 1, 'b': 2, 'c': 3}, calling clean on it would return 1, which is the value associated with the first key 'a'.

    Note that this function assumes the dictionary is not empty. If the dictionary is empty, next will raise a StopIteration exception.
    """
    return next(iter(dict_variable.values()))


text_list = []
for filename in os.listdir("./data/contracts"):
    file_path = os.path.join("./data/contracts", filename)
    with open(file_path, "r", encoding="utf-8") as file:
        console.print(f"[cyan bold]Processing file: {filename}[/]")
        text = file.read()
        text_list.append(text)


def get_features(text):

    prompt = f"""Given this contract text, extract the following fields: 'Employee Name', 
    'Yearly Salary', 'Non-Compete Clause (Y/N)', 'Start Date'. Output in the following JSON format
    "Agreement": "Employee Name": "..." 
    
    Contract text:
    {text}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    return get_first_value(json.loads(response.choices[0].message.content))


output_list = []

for t in text_list:
    console.print("[dark_orange bold]Extracting features from text[/]")
    output_list.append(get_features(t))

output_df = pd.DataFrame(output_list)
output_filename = "./output/extracted_features.csv"
console.print(f"[green bold]Outputing to {output_filename}[/]")
output_df.to_csv(output_filename, index=False)
