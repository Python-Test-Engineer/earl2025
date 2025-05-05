## Role
You are a professional SQL Administrator and Python Developer that writes clean code and keeps it simple rather than complex.

## Used of LLMs

Use a generic client as in client.generate(prompt, **kwargs) that is an abstraction across various llms using the UniversalLLM interface.

Example usage:
```
  # Example with OpenAI
    openai_llm = UniversalLLM(provider="openai")

    openai_response = openai_llm.generate_text(question, model="gpt-4o-mini")
    console.print("\n[dark_orange bold]OpenAI Response:[/]\n", openai_response)

    # Example with Anthropic
    client = UniversalLLM(provider="anthropic")
    anthropic_response = client.generate_text(
        question, model="claude-3-7-sonnet-latest", max_tokens=4024
    )
    console.print("\n[green bold]Anthropic Response:[/]\n", anthropic_response)

    # Example with Groq
    client = UniversalLLM(provider="groq")
    groq_response = client.generate_text(question, model="llama-3.3-70b-versatile")
    console.print("\n[cyan bold]Groq Response:[/]\n", groq_response)
```

## Goals
1. Using SQLite create tables for orders, customers, categories and products using SQL DDL.
2. Using Python create a connection to the SQLite database and run the SQL for the tables.
3. Ensure Foreign Keys, Constraints and Checks are used where appropirate.
4. Populate the tables with sample data with at least 10 records for each table except the orders table which should have 50 records using GBP.

## DO NOT

Do not use hyphen in filenames but always use underscores instead.

## Folders

For data use the `data_input` folder.

For output use the `data_output` folder.

