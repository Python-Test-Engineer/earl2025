from typing import List, Dict, Any, Set, Tuple
import re
import chromadb

PATH = "./chroma_db"
COLLECTION_NAME = "chinook_sql_01"


client = chromadb.PersistentClient(path=PATH)
collections = client.list_collections()
print("=" * 70)
for collection in collections:
    print(collection)

print("=" * 70)
# Initialize ChromaDB client with the latest configuration
client = chromadb.PersistentClient(path=PATH)

# Create or get a collection
collection = client.get_or_create_collection(name=COLLECTION_NAME)


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


def demonstrate_table_extraction():
    """Show how the table extraction function works."""
    test_queries = [
        "SELECT * FROM Customer WHERE Country = 'USA'",
        "SELECT Customer.FirstName, Customer.LastName, SUM(Invoice.Total) AS TotalSpent FROM Customer JOIN Invoice ON Customer.CustomerId = Invoice.CustomerId GROUP BY Customer.CustomerId ORDER BY TotalSpent DESC LIMIT 5",
        "SELECT MediaType.Name, SUM(InvoiceLine.UnitPrice * InvoiceLine.Quantity) AS Revenue FROM MediaType JOIN Track ON MediaType.MediaTypeId = Track.MediaTypeId JOIN InvoiceLine ON Track.TrackId = InvoiceLine.TrackId GROUP BY MediaType.MediaTypeId ORDER BY Revenue DESC LIMIT 1",
    ]

    print("\n=== Table Extraction Demo ===")
    for query in test_queries:
        tables = extract_tables_from_sql(query)
        print(f"\nSQL: {query}")
        print(f"Tables detected: {tables}")


def print_query_results(results):
    """Helper function to print query results consistently."""
    if "documents" in results and results["documents"]:
        # Handle results from query() function
        if isinstance(results["documents"], list) and isinstance(
            results["documents"][0], list
        ):
            for i, doc in enumerate(results["documents"][0]):
                print(f"\nMatch {i+1}: {doc}")
                print(f"Metadata: {results['metadatas'][0][i]}")
        # Handle results from get() function
        else:
            for i, doc in enumerate(results["documents"]):
                print(f"\nMatch {i+1}: {doc}")
                print(f"Metadata: {results['metadatas'][i]}")
    else:
        print("No results found")


def query_collection():
    """Demonstrate querying the ChromaDB collection."""

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


def demonstrate_advanced_queries():
    """Show more advanced filtering capabilities."""
    client = chromadb.Client()
    print("...in demonstrate_advanced_queries")
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        print("Got collection in demonstrate_advanced_queries")
        print("\n=== Advanced Query Demo ===")

        # Query by tables involved with string contains
        print("\nQueries involving Customer table:")
        results = collection.query(
            query_texts=["customer purchase information"],
            where={"tables_involved": {"$contains": "Customer"}},
            n_results=2,
        )

        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                print(f"\nMatch {i+1}: {doc}")
                print(f"Metadata: {results['metadatas'][0][i]}")

        # Query by complexity
        print("\nComplex queries:")
        results = collection.query(
            query_texts=["advanced database analysis"],
            where={"sql_complexity": "complex"},
            n_results=2,
        )

        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                print(f"\nMatch {i+1}: {doc}")
                print(f"Metadata: {results['metadatas'][0][i]}")

        # Use our helper function to query by multiple tables
        print(
            "\nUsing helper function to query by multiple tables (Customer and Invoice):"
        )
        results = query_by_tables(
            client=client,
            collection_name="test_sql",
            tables=["Customer", "Invoice"],
            query_text="sales analysis",
            n_results=2,
        )

        print_query_results(results)

    except Exception as e:
        print(f"Error in advanced queries: {e}")


def analyze_collection():
    """Analyze the collection to get statistics on metadata."""
    client = chromadb.Client()

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        results = collection.get()
        print("Got collection in analyze_collection")
        if not results or not results["metadatas"]:
            print("Collection is empty or no metadata available")
            return

        # Analyze tables
        table_counts = {}
        question_types = {}
        complexity_count = {"simple": 0, "complex": 0}

        for metadata in results["metadatas"]:
            # Count tables
            if "tables_involved" in metadata:
                tables = metadata["tables_involved"].split(",")
                for table in tables:
                    if table and table != "none":
                        table_counts[table] = table_counts.get(table, 0) + 1

            # Count question types
            if "question_type" in metadata:
                qtype = metadata["question_type"]
                question_types[qtype] = question_types.get(qtype, 0) + 1

            # Count complexity
            if "sql_complexity" in metadata:
                complexity = metadata["sql_complexity"]
                complexity_count[complexity] = complexity_count.get(complexity, 0) + 1

        # Print results
        print("\n=== Collection Analysis ===")
        print(f"Total documents: {len(results['metadatas'])}")

        print("\nTable Distribution:")
        sorted_tables = sorted(table_counts.items(), key=lambda x: x[1], reverse=True)
        for table, count in sorted_tables:
            print(f"  {table}: {count}")

        print("\nQuestion Type Distribution:")
        sorted_types = sorted(question_types.items(), key=lambda x: x[1], reverse=True)
        for qtype, count in sorted_types:
            print(f"  {qtype}: {count}")

        print("\nComplexity Distribution:")
        for complexity, count in complexity_count.items():
            print(f"  {complexity}: {count}")

    except Exception as e:
        print(f"Error analyzing collection: {e}")


def main():

    # Demonstrate table extraction
    demonstrate_table_extraction()
    print("=" * 70)
    # Analyze the collection
    analyze_collection()
    print("=" * 70)
    # Demonstrate querying
    query_collection()
    print("=" * 70)
    client = chromadb.PersistentClient(path=PATH)
    collections = client.list_collections()
    print("=" * 70)
    print("before demonstrating advanced queries in main()")
    for collection in collections:
        print(collection)
    print("=" * 70)
    # Demonstrate advanced queries
    demonstrate_advanced_queries()


if __name__ == "__main__":

    main()
