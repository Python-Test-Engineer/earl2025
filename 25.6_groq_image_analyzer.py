#!/usr/bin/env python3
"""
Simple script to analyze an image using Groq's vision model
"""

import os
import base64
from groq import Groq


def encode_image(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_image_with_groq(
    image_path, question="What can you tell me about this feature importance chart?"
):
    """
    Analyze an image using Groq's vision model

    Args:
        image_path (str): Path to the image file
        question (str): Question to ask about the image

    Returns:
        str: Response from the vision model
    """

    # Initialize Groq client
    # Make sure to set your GROQ_API_KEY environment variable
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Encode the image
    base64_image = encode_image(image_path)

    try:
        # Create chat completion with vision model
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",  # Groq's vision model
            temperature=0.1,
            max_tokens=1000,
        )

        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"Error analyzing image: {str(e)}"


def main():
    """Main function"""
    image_path = "./images/feature_importance.png"
    image_path = "./images/shap_importance.png"

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return

    # Check if API key is set
    if not os.getenv("GROQ_API_KEY"):
        print("Error: Please set your GROQ_API_KEY environment variable!")
        print("You can get an API key from: https://console.groq.com/")
        return

    print(f"Analyzing {image_path}...")
    print("-" * 50)

    # Ask a specific question about the feature importance chart
    question = """
    Analyze this feature importance chart and tell me:
    1. What are the top 3 most important features?
    2. What type of model or analysis does this appear to be from?
    3. Are there any notable patterns or insights you can observe?
    """

    response = analyze_image_with_groq(image_path, question)
    print(response)
    with open("./output/groq_image_analysis.md", "w") as f:
        f.write(response)


if __name__ == "__main__":
    main()
