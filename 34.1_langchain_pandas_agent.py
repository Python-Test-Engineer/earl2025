# An example of using LangChain with a pandas DataFrame agent to analyze a CSV dataset.

import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import os

# Set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"


def main():
    # Load and prepare the DataFrame
    df = pd.read_csv("./data/all-states-history.csv").fillna(value=0)

    # Initialize the OpenAI model
    llm = ChatOpenAI(
        model="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create the pandas DataFrame agent
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        allow_dangerous_code=True,  # Required for pandas agent
        handle_parsing_errors=True,
        max_iterations=3,
        early_stopping_method="generate",
    )

    # Example queries
    queries = [
        "What are the column names in this dataset?",
        "How many rows are in the dataset?",
        "Show me the first 5 rows",
        "What are the data types of each column?",
        "Give me basic statistics about the numerical columns",
    ]

    print("DataFrame loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\n" + "=" * 50 + "\n")

    # Run example queries
    for i, query in enumerate(queries, 1):
        print("==================================================")
        print(f"Query {i}: {query}")
        try:
            result = agent.invoke({"input": query})
            print(f"Result: {result['output']}")
        except Exception as e:
            print(f"Error: {str(e)}")
        print("\n" + "-" * 30 + "\n")

    # Interactive mode
    print("Interactive mode - Ask questions about your data (type 'quit' to exit):")
    while True:
        user_query = input("\nYour question: ").strip()
        if user_query.lower() in ["quit", "exit", "q"]:
            break

        try:
            result = agent.invoke({"input": user_query})
            print(f"Answer: {result['output']}")
        except Exception as e:
            print(f"Error: {str(e)}")


# Alternative configuration for better error handling
def create_robust_agent():
    """Create a more robust pandas agent with better error handling"""

    df = pd.read_csv("./data/all-states-history.csv").fillna(value=0)

    llm = ChatOpenAI(
        model="gpt-3.5-turbo", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create agent with specific configuration to avoid common errors
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        allow_dangerous_code=True,
        handle_parsing_errors=True,
        max_iterations=5,
        early_stopping_method="generate",
        prefix="""
You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
You should use the tools below to answer the question posed of you:

python_repl_ast: A Python shell. Use this to execute python commands. 
Input should be a valid python command. When using this tool, sometimes output is abbreviated - 
make sure it does not look abbreviated before using the observation in your answer.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be python_repl_ast
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
        """,
        format_instructions="""
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be python_repl_ast
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
        """,
    )

    return agent, df


# Usage example with error handling
def run_with_error_handling():
    try:
        agent, df = create_robust_agent()

        # Test query
        query = "What are all the column names in this dataframe?"
        print(f"Testing query: {query}")

        result = agent.invoke({"input": query})
        print(f"Result: {result['output']}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Make sure you have:")
        print("1. Set your OPENAI_API_KEY environment variable")
        print(
            "2. Installed required packages: pip install langchain langchain-openai langchain-experimental pandas"
        )
        print("3. Your CSV file exists at the specified path")


if __name__ == "__main__":
    # Choose which version to run
    print("Choose an option:")
    print("1. Run main example")
    print("2. Run robust agent example")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        main()
    elif choice == "2":
        run_with_error_handling()
    else:
        print("Invalid choice. Running robust agent example...")
        run_with_error_handling()
