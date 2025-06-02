import base64
from groq import Groq

def analyze_image_with_groq(image_path, question, api_key):
    """
    Analyze a local image using Groq's vision model.
    
    Args:
        image_path (str): Path to the local image file
        question (str): Question to ask about the image
        api_key (str): Your Groq API key
    
    Returns:
        str: Groq's response about the image
    """
    # Initialize Groq client
    client = Groq(api_key=api_key)
    
    # Read and encode the image
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
    
    # Create the chat completion request
    response = client.chat.completions.create(
        model="llava-v1.5-7b-4096-preview",  # Groq's vision model
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000,
        temperature=0.7
    )
    
    return response.choices[0].message.content

# Example usage
if __name__ == "__main__":
    # Replace with your actual API key
    API_KEY = "your_groq_api_key_here"
    
    # Example usage
    try:
        result = analyze_image_with_groq(
            image_path="path/to/your/image.jpg",
            question="What do you see in this image?",
            api_key=API_KEY
        )
        print(result)
    except Exception as e:
        print(f"Error: {e}")
