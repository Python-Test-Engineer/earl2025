# Given a prompt and a some sample CSV data plus headers from 25_medical_appointments.csv, this generates an analysis of the data (without the the data card) and produces an analysis of the data and the python code to clean it.

# The analysis is saved in a folder `code_responses` as a file called code_response_NNN.md, where NNN is a 3-digit number with leading zeros and the next number in sequence.

# The code is then extracted from the markdown file and saved in a python file called code_response_001_python_code.py. The code is then executed and the cleaned data is saved in a file called cleaned_25_medical_appointments.csv

# The initial CSV file can be made a variable but is hard coded for now.

import os
import re
import base64

from dotenv import load_dotenv, find_dotenv
from rich.console import Console

from utils_agentic.base_import import LLMClient, check_api_keys, read_prompt_file


console = Console()

OUTPUT_FOLDER = "./code_responses/"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


def find_next_response_file():
    """
    Searches the root folder for files matching the pattern 'claude_response_NNN.md',
    finds the greatest numerical value of N (after removing leading zeros),
    and returns the next number in sequence formatted as a 3-digit string with leading zeros.

    Returns:
        tuple: (next_number_formatted, next_filename) where:
              - next_number_formatted is a 3-digit string with leading zeros
              - next_filename is the complete next filename in the sequence

    Examples:
        - If highest file is 'code_response_001.md' (N=1), returns ('002', 'code_response_002.md')
        - If highest file is 'code_response_011.md' (N=11), returns ('012', 'code_response_012.md')
        - If highest file is 'code_response_102.md' (N=102), returns ('103', 'code_response_103.md')
    """
    # Pattern to match files like code_response_001.md, code_response_042.md, etc.
    pattern = re.compile(r"code_response_(\d{3})\.md")

    # Get all files in the current directory
    files = os.listdir(f"./{OUTPUT_FOLDER}")
    console.print(f"Files in directory {OUTPUT_FOLDER}: {files}")

    # Find all matching files and extract the numerical values
    n_values = []
    for filename in files:
        match = pattern.match(filename)
        if match:
            # Convert the 3-digit string to an integer (removing leading zeros)
            n = int(match.group(1))
            n_values.append(n)

    # If no matching files found, start with 0
    if not n_values:
        highest_n = 0
    else:
        highest_n = max(n_values)

    # Calculate the next number in sequence
    next_n = highest_n + 1

    # Format the next number as a 3-digit string with leading zeros
    next_n_formatted = f"{next_n:03d}"
    next_filename = f"code_response_{next_n_formatted}.md"

    return next_n_formatted, next_filename


def extract_python_code(markdown_file, output_file=None):
    """
    Extract Python code blocks from a markdown file and save to a Python file.

    Parameters:
    markdown_file (str): Path to the markdown file
    output_file (str, optional): Path to save the extracted Python code. If None,
                                a name will be generated based on the markdown filename.

    Returns:
    str: Path to the saved Python file
    """
    console.print(
        f"\n[green]Extracting Python code from: {markdown_file} to {output_file}[/]"
    )
    # If output_file is not specified, generate one based on the markdown filename
    if output_file is None:
        # Get the base name without extension
        base_name = os.path.splitext(os.path.basename(markdown_file))[0]
        # Extract the last three digits if they exist
        digits_match = re.search(r"(\d{3})$", base_name)
        if digits_match:
            digits = digits_match.group(1)
            output_file = f"code_response_python_{digits}.py"
        else:
            output_file = f"{base_name}_extracted.py"

    # Read the markdown file
    with open(markdown_file, "r", encoding="utf-8") as f:
        console.print(f"\n[green]Reading file: {markdown_file}[/]")
        content = f.read()

    # Extract Python code blocks (text between ```python and ```)
    # This pattern looks for ```python followed by any text (non-greedy) until ```
    pattern = r"```python\s*(.*?)\s*```"

    # re.DOTALL allows . to match newlines
    matches = re.findall(pattern, content, re.DOTALL)

    if not matches:
        # Try a more generic pattern if no Python blocks found
        pattern = r"```\s*(.*?)\s*```"
        matches = re.findall(pattern, content, re.DOTALL)

    if matches:
        # Combine all code blocks with newlines between them
        extracted_code = "\n\n".join(matches)

        # replace 'DATA_FILE' with the actual file name - this is result from the prompt - a refined prompt might be needed.
        extracted_code = extracted_code.replace(
            "DATA_FILE", "25_medical_appointments.csv"
        )

        # Write the extracted code to the output file
        with open(output_file, "w", encoding="utf-8") as f:
            console.print(f"\n[green]Writing extracted code to: {output_file}[/]")
            f.write(extracted_code)

        return output_file
    else:
        print("No Python code blocks found in the markdown file.")
        return None


def main():
    """Main function to run the LLM client."""
    load_dotenv(find_dotenv(), override=True)
    console = Console()

    # Check API keys
    console.print("[bold]Checking API Keys:[/bold]")
    check_api_keys()

    provider = "ANTHROPIC"  # Change as needed
    # https://www.kaggle.com/datasets/joniarroba/noshowappointments/data
    DATA_FILE = "./25_medical_appointments_sample.csv"

    PROMPT_FILE = "./prompts/02_base_prompt.md"
    # You are an experienced Python data analyst. Analyse the CSV content and list all the actions needed to clean the data and the Python Code to do it. Keep the code small and simple. The content of the file is called {DATA_FILE} and its content is {file_content}

    OUTPUT_FILE = find_next_response_file()[1]
    console.print(f"\n[green]Next response file: {OUTPUT_FILE}[/]\n")
    # Read the file content and encode as base64
    with open(PROMPT_FILE, "r") as f:
        prompt = f.read()

    try:
        # Initialize LLM client
        llm = LLMClient(provider=provider)

        # Read the file content and encode/decode as base64 to prevent ASCII errors.
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            # file_content = base64.b64encode(f.read()).decode("utf-8") when rb
            file_content = f.read()

        # Read prompt
        system_message = prompt + file_content

        user_message = f"\nThank you"

        # Generate response
        response = llm.generate(system_message, user_message)

        # Display response
        console.print("\n[bold]Response:[/bold]")
        console.print(response)
        with open(f"{OUTPUT_FOLDER}{OUTPUT_FILE}", "w") as f:
            f.write(response)
            console.print(f"[green]Analysis saved to: {OUTPUT_FOLDER}{OUTPUT_FILE}[/]")

        extract_python_code(
            f"{OUTPUT_FOLDER}{OUTPUT_FILE}",
            # "./code_responses/code_response_002.md",
            output_file=f"{OUTPUT_FOLDER}{OUTPUT_FILE}_python_code.py",
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


if __name__ == "__main__":
    main()
# Days between may give -1 if they are on the same day and the time is before the appointment time.
