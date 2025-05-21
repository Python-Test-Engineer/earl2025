import os
import anthropic
from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console


class LLMClient:
    """Client for different LLM providers (OpenAI, Groq, Anthropic)."""

    SUPPORTED_PROVIDERS = {
        "GROQ": {
            "client_class": OpenAI,
            "default_model": "llama-3.3-70b-versatile",
            "api_key_env": "GROQ_API_KEY",
            "base_url": "https://api.groq.com/openai/v1",
        },
        "OPENAI": {
            "client_class": OpenAI,
            "default_model": "gpt-4o-mini",
            "api_key_env": "OPENAI_API_KEY",
            "base_url": None,
        },
        "ANTHROPIC": {
            "client_class": anthropic.Anthropic,
            "default_model": "claude-3-7-sonnet-20250219",
            "api_key_env": "ANTHROPIC_API_KEY",
            "base_url": None,
        },
    }

    def __init__(self, provider, model=None, temperature=0):
        """Initialize LLM client.

        Args:
            provider: LLM provider (OPENAI, GROQ, or ANTHROPIC)
            model: Model name (if None, uses provider's default)
        """
        load_dotenv()
        self.console = Console()

        # Validate provider
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Invalid provider. Choose from: {', '.join(self.SUPPORTED_PROVIDERS.keys())}"
            )

        self.provider = provider
        self.provider_config = self.SUPPORTED_PROVIDERS[provider]
        self.temperature = temperature

        # Set model (use default if none specified)
        self.model = model or self.provider_config["default_model"]

        # Get API key
        api_key_env = self.provider_config["api_key_env"]
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise ValueError(f"{api_key_env} environment variable not set")

        # Create client
        self._initialize_client()

        self.console.print(
            f"Using [bold]{self.provider}[/bold] with model: [bold]{self.model}[/bold]"
        )

    def _initialize_client(self):
        """Initialize the appropriate client based on provider."""
        client_class = self.provider_config["client_class"]
        base_url = self.provider_config["base_url"]

        if base_url:
            self.client = client_class(base_url=base_url, api_key=self.api_key)
        else:
            self.client = client_class(api_key=self.api_key)

    def generate(
        self,
        user_message,
        temperature=1.0,
        max_tokens=1000,
        system_message="You are a helpful assistant",
    ):
        """Generate a response from the LLM.

        Args:
            system_message: System message/instructions
            user_message: User's message/query
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response

        Returns:
            Generated text response
        """
        self.console.print("\nGenerating response...\n")

        if self.provider == "ANTHROPIC":
            # Anthropic uses a different API format
            combined_prompt = f"{system_message}\n\n{user_message}"
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": combined_prompt}],
                temperature=1.0,
            )
            return message.content[0].text
        else:
            # OpenAI-compatible API format (OpenAI and Groq)
            prompts = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
            response = self.client.chat.completions.create(
                model=self.model, messages=prompts, temperature=self.temperature
            )
            return response.choices[0].message.content


def read_prompt_file(filepath):
    """Read prompt from a file.

    Args:
        filepath: Path to the prompt file

    Returns:
        String content of the file
    """
    try:
        with open(filepath, "r") as f:
            content = f.read()
            Console().print(f"Read prompt from [bold]{filepath}[/bold]")
            return content
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {filepath}")


def check_api_keys():
    """Check and display status of API keys."""
    console = Console()
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    }

    for key_name, key_value in api_keys.items():
        if key_value:
            masked_key = (
                f"{key_value[:5]}...{key_value[-3:]}" if len(key_value) > 8 else "***"
            )
            console.print(f"[green]✓[/green] {key_name}: {masked_key}")
        else:
            console.print(f"[red]✗[/red] {key_name}: Not set")
