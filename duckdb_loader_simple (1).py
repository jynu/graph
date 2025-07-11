#!/usr/bin/env python3
"""
Simple DuckDB Knowledge Graph Loader
This version just loads an existing DuckDB file without creating one.
Perfect for corporate environments with restricted permissions.
"""

import sys
import logging

# Simple logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all required imports."""
    logger.info("Testing imports...")
    
    try:
        import duckdb
        logger.info(f"DuckDB imported successfully, version: {duckdb.__version__}")
    except ImportError as e:
        logger.error(f"DuckDB import failed: {e}")
        return False
    
    try:
        import pandas as pd
        logger.info("Pandas imported successfully")
    except ImportError as e:
        logger.error(f"Pandas import failed: {e}")
        return False
    
    try:
        import json
        logger.info("JSON imported successfully")
    except ImportError as e:
        logger.error(f"JSON import failed: {e}")
        return False
    
    return True

def load_existing_database(db_path="knowledge_graph.duckdb"):
    """Load an existing DuckDB knowledge graph."""
    logger.info(f"Loading existing database: {db_path}")
    
    try:
        import duckdb
        
        # Connect to existing database
        conn = duckdb.connect(db_path)
        logger.info("Database connection successful")
        
        # Verify database structure
        logger.info("Verifying database structure...")
        
        # Check tables
        tables = conn.execute("SHOW TABLES").fetchall()
        logger.info(f"Found {len(tables)} tables:")
        for table in tables:
            logger.info(f"  - {table[0]}")
        
        # Get basic statistics
        if tables:
            for table_name, in tables:
                try:
                    count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                    logger.info(f"Table {table_name}: {count} rows")
                except Exception as e:
                    logger.warning(f"Could not count rows in {table_name}: {e}")
        
        logger.info("Database loaded successfully!")
        return conn
        
    except FileNotFoundError:
        logger.error(f"Database file not found: {db_path}")
        logger.error("Please copy the knowledge_graph.duckdb file to the current directory")
        return None
    except Exception as e:
        logger.error(f"Failed to load database: {e}")
        return None

def simple_table_search(conn, query_text):
    """Simple table search using SQL queries."""
    logger.info(f"Searching for tables related to: {query_text}")
    
    try:
        # Search in table names and descriptions
        search_sql = """
        SELECT name, description, table_type
        FROM tables 
        WHERE LOWER(name) LIKE ? 
           OR LOWER(description) LIKE ?
        ORDER BY name
        LIMIT 10
        """
        
        search_term = f"%{query_text.lower()}%"
        results = conn.execute(search_sql, [search_term, search_term]).fetchall()
        
        if results:
            logger.info(f"Found {len(results)} matching tables:")
            for name, desc, table_type in results:
                logger.info(f"  - {name} ({table_type}): {desc[:100]}...")
        else:
            logger.info("No matching tables found")
            
        return results
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []

def get_table_columns(conn, table_name):
    """Get columns for a specific table."""
    try:
        sql = """
        SELECT name, data_type, description 
        FROM columns 
        WHERE table_name = ?
        ORDER BY name
        """
        
        results = conn.execute(sql, [table_name]).fetchall()
        
        if results:
            logger.info(f"Columns in {table_name}:")
            for name, data_type, desc in results:
                logger.info(f"  - {name} ({data_type}): {desc[:50]}...")
        else:
            logger.info(f"No columns found for {table_name}")
            
        return results
        
    except Exception as e:
        logger.error(f"Failed to get columns for {table_name}: {e}")
        return []

def interactive_mode(conn):
    """Simple interactive mode for exploring the knowledge graph."""
    logger.info("=== Interactive Knowledge Graph Explorer ===")
    logger.info("Commands:")
    logger.info("  search <text>     - Search for tables")
    logger.info("  columns <table>   - Show columns for a table")
    logger.info("  tables           - List all tables")
    logger.info("  quit             - Exit")
    
    while True:
        try:
            command = input("\nKG> ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                break
            elif command.lower() == 'tables':
                tables = conn.execute("SELECT name, table_type FROM tables ORDER BY name").fetchall()
                print(f"\nAll tables ({len(tables)}):")
                for name, table_type in tables:
                    print(f"  - {name} ({table_type})")
            elif command.lower().startswith('search '):
                query_text = command[7:].strip()
                simple_table_search(conn, query_text)
            elif command.lower().startswith('columns '):
                table_name = command[8:].strip()
                get_table_columns(conn, table_name)
            elif command.lower() == 'help':
                logger.info("Available commands: search, columns, tables, quit")
            else:
                logger.info("Unknown command. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {e}")
    
    logger.info("Goodbye!")

def main():
    """Main function."""
    logger.info("=== Simple DuckDB Knowledge Graph Loader ===")
    
    # Test imports first
    if not test_imports():
        logger.error("Import tests failed. Please install required packages.")
        return
    
    # Load existing database
    conn = load_existing_database()
    if not conn:
        logger.error("Failed to load database. Exiting.")
        return
    
    # Start interactive mode
    try:
        interactive_mode(conn)
    except Exception as e:
        logger.error(f"Interactive mode failed: {e}")
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)
