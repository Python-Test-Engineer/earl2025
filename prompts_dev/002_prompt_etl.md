# Using this UniversallLLM class to connect to LLMS

## Basic clients for different LLM providers

class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None):
        openai.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = openai.OpenAI()

    def generate(self, prompt: str, model: str = "gpt-4o-mini", **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], **kwargs
        )
        return response.choices[0].message.content

class AnthropicClient:
    def __init__(self, api_key: Optional[str] = None):
        import anthropic

        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    def generate(
        self, prompt: str, model: str = "claude-3-7-sonnet-20250219", **kwargs
    ) -> str:
        response = self.client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.pop("max_tokens", 1024),
            **kwargs,
        )
        return response.content[0].text

class GroqClient:
    def __init__(self, api_key: Optional[str] = None):

        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY"),
        )

    def generate(self, prompt: str, model: str = "llama3-8b-8192", **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], **kwargs
        )
        return response.choices[0].message.content

class UniversalLLM:
    """Simple universal interface for generating text with different LLM providers."""

    def __init__(self, provider: str = "openai", api_key: Optional[str] = None):
        """
        Initialize the UniversalLLM interface.

        Args:
            provider: The LLM provider to use ("openai", "anthropic", or "groq")
            api_key: API key for the provider (if None, will look for environment variable)
        """
        self.provider = provider.lower()

        # Initialize the appropriate client
        if self.provider == "openai":
            self.client = OpenAIClient(api_key)
        elif self.provider == "anthropic":
            self.client = AnthropicClient(api_key)
        elif self.provider == "groq":
            self.client = GroqClient(api_key)
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. Choose 'openai', 'anthropic', or 'groq'"
            )

    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the configured LLM provider.

        Args:
            prompt: The text prompt to send to the LLM
            **kwargs: Additional provider-specific parameters

        Returns:
            The generated text response
        """
        return self.client.generate(prompt, **kwargs)

## Example usage

if __name__ == "__main__":
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"

    question = "What countries are in the UK? Answer in a list format each item on a new line and keep answer concise and short."
    # Example with OpenAI
    # openai_llm = UniversalLLM(provider="openai")

    # openai_response = openai_llm.generate_text(question, model="gpt-4o-mini")
    # console.print("\n[dark_orange bold]OpenAI Response:[/]\n", openai_response)

    # Example with Anthropic
    client = UniversalLLM(provider=ANTHROPIC)
    anthropic_response = client.generate_text(
        question, model="claude-3-7-sonnet-latest", max_tokens=4024
    )
    console.print("\n[green bold]Anthropic Response:[/]\n", anthropic_response)

    # Example with Groq
    client = UniversalLLM(provider=GROQ)
    groq_response = client.generate_text(question, model="llama-3.3-70b-versatile")
    console.print("\n[cyan bold]Groq Response:[/]\n", groq_response)

Import the necessary classes and then

Create the following:

A small sample csv dataset with a number of issues:

- missing values
- a mix of different date formats
- NaN
- other issues you feel appropriate to create a dirty dataset.

Then create an AI Agent that uses an LLM to load the dataset into a CSV, perform an analysis that is saved to fix.md file.

Use Groq as LLM.

Then carry out the fixes to clean the dataset.

Finally, create an AI Agent that uses an LLM to review the dataset and the analysis as well as the final version. It then creates a report of the effectiveness of the agent and saves to evaluation.md

Keep all code clean and simple. Short better than long.

Do not use hyphens in file names but underscores
