import csv
import re
import chromadb
from chromadb import Client
from chromadb.utils import embedding_functions
import os
from typing import List, Dict, Any, Set, Tuple

PATH = "./chroma_db"
COLLECTION_NAME = "chinook_sql_01"


def extract_tables_from_sql(sql: str) -> Set[str]:
    """
    Extract table names from SQL query using regex pattern matching.
    Enhanced to handle table aliases and more SQL patterns.
    """
    # Common tables in the Chinook database
    chinook_tables = {
        "Album",
        "Artist",
        "Customer",
        "Employee",
        "Genre",
        "Invoice",
        "InvoiceLine",
        "MediaType",
        "Playlist",
        "PlaylistTrack",
        "Track",
    }

    # This more complex regex handles:
    # 1. Tables after FROM keyword
    # 2. Tables after JOIN keywords (including LEFT/RIGHT/INNER/OUTER JOIN)
    # 3. Handles potential AS keyword for aliases
    # 4. Handles potential schema prefix like "dbo."
    pattern = r"(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)(?:\s+(?:AS\s+)?[A-Za-z0-9_]+)?"

    matches = re.findall(pattern, sql, re.IGNORECASE)

    # Process matches to extract actual table names
    tables = set()
    for match in matches:
        # Remove schema prefix if present
        if "." in match:
            table_name = match.split(".")[-1]
        else:
            table_name = match

        # Only include if it's a known Chinook table
        # Case-insensitive comparison
        for chinook_table in chinook_tables:
            if table_name.lower() == chinook_table.lower():
                tables.add(chinook_table)  # Add with proper case
                break

    return tables


def extract_question_type(question: str) -> str:
    """
    Determine the type of SQL operation based on the question.
    """
    question = question.lower()

    if any(
        word in question
        for word in ["average", "sum", "count", "total", "most", "least"]
    ):
        return "aggregation"
    elif "join" in question or "relationship" in question or "related" in question:
        return "join"
    elif any(word in question for word in ["where", "find", "filter", "which"]):
        return "filter"
    elif "group" in question or "each" in question:
        return "group"
    else:
        return "general"


def query_by_tables(
    client,
    collection_name: str,
    tables: List[str],
    query_text: str = "",
    n_results: int = 5,
):
    """
    Helper function to query the collection by specific tables.

    Args:
        client: ChromaDB client instance
        collection_name: Name of the collection to query
        tables: List of table names to filter by
        query_text: Optional natural language query text
        n_results: Number of results to return

    Returns:
        Query results from ChromaDB
    """
    collection = client.get_collection(name=COLLECTION_NAME)

    # Build where clauses for each table
    where_clauses = []
    for table in tables:
        where_clauses.append({"tables_involved": {"$contains": table}})

    # Execute query
    if query_text:
        results = collection.query(query_texts=[query_text], n_results=n_results)
    else:
        # If no query text, just filter by tables
        results = collection.get(
            where=(
                {"$and": where_clauses} if len(where_clauses) > 1 else where_clauses[0]
            ),
            limit=n_results,
        )

    return results


def load_sql_to_chroma(
    csv_files: List[str], collection_name: str = COLLECTION_NAME
) -> None:
    """
    Load SQL questions and queries from CSV files into a ChromaDB collection.
    Extracts metadata from questions and SQL queries.

    Args:
        csv_files: List of paths to CSV files containing question-SQL pairs
        collection_name: Name of the ChromaDB collection to create
    """
    # Initialize ChromaDB client
    # client = chromadb.Client()

    client = chromadb.PersistentClient(
        path=PATH
    )  # Creates file chroma.sqlite3 if it does not exist

    # Create or get collection
    collection_name = COLLECTION_NAME
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"Using existing collection: {COLLECTION_NAME}")
    except:
        print(f"Creating new collection: {COLLECTION_NAME}")
        # Using the default embedding function
        embedding_function = embedding_functions.DefaultEmbeddingFunction()
        collection = client.create_collection(
            name=COLLECTION_NAME, embedding_function=embedding_function
        )

    # Process each CSV file
    all_data = []

    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"Warning: File not found: {csv_file}")
            continue

        print(f"Processing file: {csv_file}")

        with open(csv_file, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip header row

            for row in reader:
                if len(row) >= 2:
                    question, sql = row[0], row[1]

                    # Extract tables mentioned in the SQL
                    tables = extract_tables_from_sql(sql)

                    # Extract question type
                    question_type = extract_question_type(question)

                    # Create metadata - convert lists to comma-separated strings
                    # since ChromaDB only accepts simple types for metadata values
                    metadata = {
                        "tables_involved": ",".join(tables) if tables else "none",
                        "tables_count": len(tables),
                        "question_type": question_type,
                        "sql_complexity": (
                            "complex"
                            if len(tables) > 2 or "WITH" in sql or "OVER" in sql
                            else "simple"
                        ),
                    }

                    all_data.append((question, sql, metadata))

    # Add documents to collection in batch
    if all_data:
        ids = [f"query_{i}" for i in range(len(all_data))]
        documents = [f"Question: {item[0]}\nSQL: {item[1]}" for item in all_data]
        metadatas = [item[2] for item in all_data]

        collection.add(ids=ids, documents=documents, metadatas=metadatas)

        print(f"Added {len(all_data)} documents to collection '{collection_name}'")
    else:
        print("No data was loaded into ChromaDB")

    # Print some collection stats
    print(f"Collection '{collection_name}' now has {collection.count()} documents")

    # Example query to verify
    results = collection.query(query_texts=["What are the total sales?"], n_results=2)

    print("\nExample query results:")
    for i, (doc, metadata) in enumerate(
        zip(results["documents"][0], results["metadatas"][0])
    ):
        print(f"\nMatch {i+1}:")
        print(f"Document: {doc}")
        print(f"Metadata: {metadata}")


if __name__ == "__main__":
    load_sql_to_chroma(
        ["./chinook_questions_sql.csv", "./chinook_questions_sql_set2.csv"]
    )
    # Initialize the client
    client = chromadb.Client()

    try:
        collection = client.get_collection(name=COLLECTION_NAME)

        # Example queries
        test_queries = [
            "Show me sales information by country",
            "Which artist has the most tracks?",
            "Find all customers from Germany",
            "What's the revenue from rock music?",
        ]

        print("\n=== ChromaDB Query Demo ===")
        for query in test_queries:
            print(f"\nQuerying: '{query}'")
            results = collection.query(query_texts=[query], n_results=1)

            print_query_results(results)

    except Exception as e:
        print(f"Error accessing collection: {e}")
