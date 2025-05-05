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
    import re
    import os

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

        # Write the extracted code to the output file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(extracted_code)

        print(f"Python code extracted and saved to {output_file}")
        return output_file
    else:
        print("No Python code blocks found in the markdown file.")
        return None


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        markdown_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        extract_python_code(markdown_file, output_file)
    else:
        print("Usage: python script.py markdown_file [output_file]")
        print("If output_file is not provided, it will be generated based on the markdown filename.")