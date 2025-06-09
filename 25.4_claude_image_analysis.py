"""
Tutorial: Analyzing Charts and Plots with Claude API

This script demonstrates how to use Claude's vision capabilities to analyze charts and plots.
We'll download a sample chart from the web and use Claude to interpret it.

The workflow includes:
1. Setting up the Anthropic API client
2. Downloading an image from the web
3. Encoding the image for Claude
4. Crafting a prompt for chart analysis
5. Submitting the request to Claude
6. Processing and displaying the response

Prerequisites:
- Python 3.7+
- anthropic package (pip install anthropic)
- requests package (pip install requests)
- PIL package (pip install pillow)
- base64 (standard library)
- io (standard library)
"""

import os
import base64
import requests
import io
from PIL import Image
from anthropic import Anthropic

# Set your API key
# In production, use environment variables or a secure method
# For this tutorial, we'll use a placeholder
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


def download_image(url):
    """
    Download an image from a URL

    Args:
        url (str): URL of the image to download

    Returns:
        PIL.Image: Downloaded image
    """
    print(f"Downloading image from {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for 4XX/5XX responses

    # Convert the response content to an image
    image = Image.open(io.BytesIO(response.content))
    print(f"Successfully downloaded image: {image.format}, {image.size}")
    return image


def encode_image_base64(image):
    """
    Encode a PIL Image to base64 string

    Args:
        image (PIL.Image): Image to encode

    Returns:
        str: Base64 encoded image string
    """
    # Convert to RGB if the image is in a different mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Save the image to an in-memory bytes buffer
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    # Encode to base64
    img_bytes = buffer.getvalue()
    base64_encoded = base64.b64encode(img_bytes).decode("utf-8")

    return base64_encoded


def analyze_chart_with_claude(image_base64, specific_questions=None):
    """
    Use Claude to analyze a chart or plot

    Args:
        image_base64 (str): Base64 encoded image
        specific_questions (list, optional): Specific questions about the chart

    Returns:
        str: Claude's analysis of the chart
    """
    # Initialize the client
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    # Prepare the prompt for Claude
    system_prompt = """You are an expert data analyst who specializes in interpreting charts and plots.
    When given an image of a chart, analyze it thoroughly and provide:
    1. The type of chart/plot
    2. The variables shown and their relationships
    3. Key trends or patterns
    4. Notable outliers or anomalies
    5. A concise summary of what the chart is conveying
    
    Be precise and quantitative where possible.
    """

    # Create the user message with the image
    user_message = "Analyze this chart in detail."
    if specific_questions:
        user_message += " Please answer these specific questions as well:\n"
        for i, question in enumerate(specific_questions, 1):
            user_message += f"{i}. {question}\n"

    # Create the message with the image
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_base64,
                    },
                },
                {"type": "text", "text": user_message},
            ],
        }
    ]

    # Send the request to Claude
    print("Sending request to Claude...")
    response = client.messages.create(
        model="claude-3-opus-20240229",
        system=system_prompt,
        messages=messages,
        max_tokens=1000,
    )

    return response.content[0].text


def main():
    """
    Main function to demonstrate chart analysis with Claude
    """
    # For this tutorial, we'll use a sample chart from a public source
    # This URL points to a sample bar chart from a public data visualization site
    chart_url = "https://craig-west.netlify.app/images/age_distribution.png"
    chart_url = "https://craig-west.netlify.app/images/feature_importance.png"

    # Specific questions we want Claude to answer about the chart
    questions = [
        "What does this chart show?",
        "What's the overall trend shown in this visualization?",
        "Are there any surprising insights from this data?",
    ]

    try:
        # Download the image
        image = download_image(chart_url)

        # Display a message about image loading
        # In a notebook, you could display the image with:
        # display(image)
        print(f"Image loaded successfully. Size: {image.width}x{image.height}")

        # Encode the image
        base64_image = encode_image_base64(image)
        print(f"Image encoded successfully. Base64 length: {len(base64_image)}")

        # Analyze the chart
        analysis = analyze_chart_with_claude(base64_image, questions)
        with open("25.5_chart_analysis.md", "w") as f:
            f.write(analysis)
        print("Analysis saved to 25.5_chart_analysis.md")

        # Print the analysis
        print("\n=== CLAUDE'S CHART ANALYSIS ===\n")
        print(analysis)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
