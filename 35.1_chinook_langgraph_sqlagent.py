# Note for Craig - Run as python 35.1_chinook_langgraph.py not using arrow play for VSCode


import requests
from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from rich.console import Console

console = Console()
llm = init_chat_model("openai:gpt-4o-mini")


# url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"

# response = requests.get(url)

# if response.status_code == 200:
#     # Open a local file in binary write mode
#     with open("35.2_Chinook.db", "wb") as file:
#         # Write the content of the response (the file) to the local file
#         file.write(response.content)
#     print("File downloaded and saved as Chinook.db")
# else:
#     print(f"Failed to download the file. Status code: {response.status_code}")

db = SQLDatabase.from_uri("sqlite:///35.2_Chinook.db")

print(f"Dialect: {db.dialect}")
print(f"Available tables: {db.get_usable_table_names()}")
print(f'Sample output: {db.run("SELECT * FROM Artist LIMIT 5;")}')


toolkit = SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()

for tool in tools:
    console.print(f"\n[green][bold]{tool.name}:[/bold] {tool.description}[/]\n")

from langgraph.prebuilt import create_react_agent

# another way to create the prompt at EOF

system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
    dialect=db.dialect,
    top_k=5,
)

agent = create_react_agent(
    llm,
    tools,
    prompt=system_prompt,
)

question = "Which sales agent made the most in sales in 2009?"
question = "What is the best selling album?"
question = "How many genres are there?"

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()


#############
 # Prompt to get recommended steps from the LLM
recommend_steps_prompt = PromptTemplate(
            template="""
            You are a SQL Database Instructions Expert. Given the following information about the SQL database, 
            recommend a series of numbered steps to take to collect the data and process it according to user instructions. 
            The steps should be tailored to the SQL database characteristics and should be helpful 
            for a sql database coding agent that will write the SQL code.
            
            IMPORTANT INSTRUCTIONS:
            - Take into account the user instructions and the previously recommended steps.
            - If no user instructions are provided, just return the steps needed to understand the database.
            - Take into account the database dialect and the tables and columns in the database.
            - Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
            - IMPORTANT: Pay attention to the table names and column names in the database. Make sure to use the correct table and column names in the SQL code. If a space is present in the table name or column name, make sure to account for it.
            
            
            User instructions / Question:
            {user_instructions}

            Previously Recommended Steps (if any):
            {recommended_steps}

            Below are summaries of the database metadata and the SQL tables:
            {all_sql_database_summary}

            Return steps as a numbered list. You can return short code snippets to demonstrate actions. But do not return a fully coded solution. The code will be generated separately by a Coding Agent.
            
            Consider these:
            
            1. Consider the database dialect and the tables and columns in the database.
            
            
            Avoid these:
            1. Do not include steps to save files.
            2. Do not include steps to modify existing tables, create new tables or modify the database schema.
            3. Do not include steps that alter the existing data in the database.
            4. Make sure not to include unsafe code that could cause data loss or corruption or SQL injections.
            5. Make sure to not include irrelevant steps that do not help in the SQL agent's data collection and processing. Examples include steps to create new tables, modify the schema, save files, create charts, etc.
  
            
            """

