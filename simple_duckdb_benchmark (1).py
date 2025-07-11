#!/usr/bin/env python3
"""
Simple DuckDB Table Retrieval Benchmark - For Existing Database
This version connects to an existing DuckDB file without creating anything new.
Perfect for corporate environments with restricted permissions.
"""

import sys
import logging
import json
from datetime import datetime
from typing import List, Dict, Any
import re

# Simple logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Enhanced Base Retriever Class with Connection Diagnostics ---
class SimpleDuckDBRetriever:
    """Simplified base class for DuckDB-based retrieval methods with enhanced connection handling."""
    def __init__(self, db_path: str = "knowledge_graph.duckdb"):
        self.conn = None
        self.db_path = db_path
        self._connect_with_diagnostics(db_path)
            
    def _connect_with_diagnostics(self, db_path: str):
        """Connect to DuckDB with comprehensive diagnostics."""
        logger.info(f"Attempting DuckDB connection for {self.__class__.__name__}...")
        logger.info(f"Database path: {db_path}")
        
        # Check if file exists
        import os
        if not os.path.exists(db_path):
            logger.error(f"Database file does not exist: {db_path}")
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        file_size = os.path.getsize(db_path)
        logger.info(f"Database file size: {file_size} bytes")
        
        # Try multiple connection strategies
        connection_strategies = [
            ("direct", lambda: self._direct_connection(db_path)),
            ("read_only", lambda: self._readonly_connection(db_path)),
            ("memory_copy", lambda: self._memory_copy_connection(db_path)),
        ]
        
        for strategy_name, connection_func in connection_strategies:
            try:
                logger.info(f"Trying {strategy_name} connection...")
                self.conn = connection_func()
                if self.conn:
                    logger.info(f"Success with {strategy_name} connection")
                    # Test the connection
                    self._test_connection()
                    return
            except Exception as e:
                logger.warning(f"{strategy_name} connection failed: {e}")
                continue
        
        # If all strategies failed
        logger.error("All connection strategies failed")
        raise Exception("Could not establish DuckDB connection")
    
    def _direct_connection(self, db_path: str):
        """Try direct file connection."""
        import duckdb
        conn = duckdb.connect(db_path)
        return conn
    
    def _readonly_connection(self, db_path: str):
        """Try read-only connection."""
        import duckdb
        conn = duckdb.connect(f"{db_path}?access_mode=read_only")
        return conn
    
    def _memory_copy_connection(self, db_path: str):
        """Copy database to memory and connect."""
        import duckdb
        import shutil
        
        # Create memory connection
        memory_conn = duckdb.connect(":memory:")
        
        # Try to attach the file database
        try:
            memory_conn.execute(f"ATTACH '{db_path}' AS file_db (READ_ONLY)")
            logger.info("Database attached to memory connection")
            return memory_conn
        except Exception as e:
            logger.warning(f"Could not attach database: {e}")
            memory_conn.close()
            raise
    
    def _test_connection(self):
        """Test the database connection with basic queries."""
        try:
            # Test basic query
            result = self.conn.execute("SELECT 1 as test").fetchone()
            logger.info(f"Basic query test: {result}")
            
            # Test tables query
            if hasattr(self, '_use_attached') and self._use_attached:
                tables = self.conn.execute("SELECT name FROM file_db.information_schema.tables WHERE table_schema = 'main'").fetchall()
            else:
                tables = self.conn.execute("SHOW TABLES").fetchall()
            
            logger.info(f"Found {len(tables)} tables in database")
            
            # Test specific table query
            if tables:
                first_table = tables[0][0]
                count = self.conn.execute(f"SELECT COUNT(*) FROM {first_table}").fetchone()[0]
                logger.info(f"Table {first_table} has {count} rows")
            
        except Exception as e:
            logger.warning(f"Connection test failed: {e}")
            # Don't raise - connection might still work for our purposes
    
    def get_tables(self, query: str) -> List[str]:
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_method_name(self) -> str:
        return self.__class__.__name__.replace('Retriever', '')
    
    def _safe_execute(self, sql: str, params: List = None):
        """Safely execute SQL with error handling."""
        try:
            if params:
                return self.conn.execute(sql, params).fetchall()
            else:
                return self.conn.execute(sql).fetchall()
        except Exception as e:
            logger.warning(f"SQL execution failed: {e}")
            return []

# --- Method 1: Simple Keyword Search ---
class KeywordRetriever(SimpleDuckDBRetriever):
    """Simple keyword matching using SQL queries."""
    
    def get_tables(self, query: str) -> List[str]:
        logger.debug(f"Keyword search for: '{query[:50]}...'")
        
        try:
            # Extract key terms
            key_terms = self._extract_key_terms(query)
            tables = set()
            
            # Search in table names and descriptions
            for term in key_terms[:5]:  # Limit to top 5 terms
                sql = """
                SELECT DISTINCT name FROM tables
                WHERE LOWER(name) LIKE ? OR LOWER(description) LIKE ?
                LIMIT 3
                """
                results = self._safe_execute(sql, [f"%{term}%", f"%{term}%"])
                tables.update([row[0] for row in results])
            
            return list(tables)[:10]
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key business terms from query."""
        # Common financial terms
        financial_terms = [
            'trade', 'trader', 'execution', 'venue', 'product', 'currency',
            'notional', 'price', 'cusip', 'ticker', 'counterparty', 'etd'
        ]
        
        query_lower = query.lower()
        found_terms = [term for term in financial_terms if term in query_lower]
        
        # Extract uppercase terms
        uppercase_terms = re.findall(r'\b[A-Z][A-Z_]+[A-Z]\b', query)
        found_terms.extend([term.lower() for term in uppercase_terms])
        
        # Add words from query
        words = [word.strip('.,!?()[]') for word in query.split() if len(word) > 3]
        found_terms.extend([word.lower() for word in words])
        
        return list(set(found_terms))  # Remove duplicates

# --- Method 2: Simple Column-Based Search ---
class ColumnRetriever(SimpleDuckDBRetriever):
    """Find tables by searching column names."""
    
    def get_tables(self, query: str) -> List[str]:
        logger.debug(f"Column search for: '{query[:50]}...'")
        
        try:
            # Extract potential column names from query
            column_terms = self._extract_column_terms(query)
            tables = set()
            
            for term in column_terms[:5]:  # Limit terms
                sql = """
                SELECT DISTINCT table_name FROM columns
                WHERE LOWER(name) LIKE ? OR LOWER(full_name) LIKE ?
                LIMIT 3
                """
                results = self._safe_execute(sql, [f"%{term}%", f"%{term}%"])
                tables.update([row[0] for row in results])
            
            return list(tables)[:10]
            
        except Exception as e:
            logger.error(f"Column search failed: {e}")
            return []
    
    def _extract_column_terms(self, query: str) -> List[str]:
        """Extract potential column names from query."""
        # Look for column-like patterns
        uppercase_terms = re.findall(r'\b[A-Z][A-Z_]+[A-Z]\b', query)
        terms = [term.lower() for term in uppercase_terms]
        
        # Common column patterns
        column_patterns = ['_id', '_key', '_sk', '_code', '_date', '_price', '_amount']
        for pattern in column_patterns:
            if pattern in query.lower():
                terms.append(pattern.strip('_'))
        
        return terms

# --- Method 3: Simple Relationship Traversal ---
class RelationshipRetriever(SimpleDuckDBRetriever):
    """Find tables via relationship connections."""
    
    def get_tables(self, query: str) -> List[str]:
        logger.debug(f"Relationship search for: '{query[:50]}...'")
        
        try:
            # First find seed tables
            seed_tables = self._find_seed_tables(query)
            all_tables = set(seed_tables)
            
            # Expand via relationships
            for seed_table in seed_tables[:3]:
                related = self._find_related_tables(seed_table)
                all_tables.update(related)
            
            return list(all_tables)[:10]
            
        except Exception as e:
            logger.error(f"Relationship search failed: {e}")
            return []
    
    def _find_seed_tables(self, query: str) -> List[str]:
        """Find initial candidate tables."""
        try:
            # Simple keyword search for seed tables
            terms = [word for word in query.lower().split() if len(word) > 3]
            tables = []
            
            for term in terms[:3]:
                sql = """
                SELECT name FROM tables
                WHERE LOWER(name) LIKE ? OR LOWER(description) LIKE ?
                LIMIT 2
                """
                results = self._safe_execute(sql, [f"%{term}%", f"%{term}%"])
                tables.extend([row[0] for row in results])
            
            return tables
        except:
            return []
    
    def _find_related_tables(self, table_name: str) -> List[str]:
        """Find tables related via relationships."""
        try:
            sql = """
            SELECT DISTINCT 
                CASE 
                    WHEN from_table = ? THEN to_table
                    ELSE from_table
                END as related_table
            FROM relationships
            WHERE from_table = ? OR to_table = ?
            LIMIT 5
            """
            results = self._safe_execute(sql, [table_name, table_name, table_name])
            return [row[0] for row in results if row[0] != table_name]
        except:
            return []

# --- Method 4: Pattern-Based Search ---
class PatternRetriever(SimpleDuckDBRetriever):
    """Search based on table type patterns."""
    
    def get_tables(self, query: str) -> List[str]:
        logger.debug(f"Pattern search for: '{query[:50]}...'")
        
        try:
            # Determine query type and find appropriate tables
            table_types = self._determine_table_types(query)
            tables = set()
            
            for table_type in table_types:
                sql = """
                SELECT name FROM tables
                WHERE table_type = ? OR LOWER(name) LIKE ?
                LIMIT 5
                """
                pattern = f"%{table_type}%"
                results = self._safe_execute(sql, [table_type, pattern])
                tables.update([row[0] for row in results])
            
            return list(tables)[:10]
            
        except Exception as e:
            logger.error(f"Pattern search failed: {e}")
            return []
    
    def _determine_table_types(self, query: str) -> List[str]:
        """Determine what table types might be relevant."""
        query_lower = query.lower()
        types = []
        
        # Transaction-related queries
        if any(word in query_lower for word in ['trade', 'transaction', 'execution', 'settlement']):
            types.append('fact')
        
        # Reference data queries
        if any(word in query_lower for word in ['product', 'trader', 'venue', 'reference', 'lookup']):
            types.append('dimension')
            types.append('reference')
        
        # Market data queries
        if any(word in query_lower for word in ['market', 'price', 'quote', 'intraday']):
            types.append('market_data')
        
        # Default to all types if no specific pattern
        if not types:
            types = ['fact', 'dimension', 'reference']
        
        return types

# --- Load Queries Function ---
def load_test_queries() -> List[Dict[str, str]]:
    """Load test queries - try Excel first, fallback to hardcoded."""
    queries = []
    
    # Try to load from Excel
    try:
        import pandas as pd
        df = pd.read_excel("DC_feedback_report_2025Apr.xlsx", sheet_name='feedback_report')
        
        # Find question column
        question_column = None
        for col in ['QUESTION', 'Question', 'question']:
            if col in df.columns:
                question_column = col
                break
        
        if question_column:
            for idx, row in df.iterrows():
                question = row[question_column]
                if pd.notna(question) and str(question).strip():
                    queries.append({
                        'id': f"Q{idx+1}",
                        'question': str(question).strip(),
                        'source': 'Excel'
                    })
            
            logger.info(f"Loaded {len(queries)} queries from Excel")
            return queries[:20]  # Limit for testing
            
    except Exception as e:
        logger.warning(f"Could not load Excel queries: {e}")
    
    # Fallback to test queries
    fallback_queries = [
        {"id": "TEST1", "question": "give me distinct source systems for cash ETD trades for yesterday", "source": "Test"},
        {"id": "TEST2", "question": "show me EXECUTING_TRADER_SOEID EXECUTION_VENUE PRODUCT_SK where TRADE_PRICE_CURRENCY and ISSUE_CURRENCY are not the same", "source": "Test"},
        {"id": "TEST3", "question": "Show me the counterparty for trade ID 18871106", "source": "Test"},
        {"id": "TEST4", "question": "get me the CUSIP that was traded highest last week", "source": "Test"},
        {"id": "TEST5", "question": "show me all trades by government entities", "source": "Test"},
        {"id": "TEST6", "question": "show me the count of credit swap trades done on 28 may 2025", "source": "Test"},
        {"id": "TEST7", "question": "find me all equity trades for RIC VOD.L", "source": "Test"},
        {"id": "TEST8", "question": "what are the top 10 most traded products", "source": "Test"},
    ]
    
    logger.info(f"Using {len(fallback_queries)} test queries")
    return fallback_queries

# --- Export Results Function ---
def export_results(results: List[Dict]):
    """Export benchmark results to CSV."""
    if not results:
        logger.warning("No results to export")
        return
    
    try:
        import pandas as pd
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Export to CSV
        csv_filename = f"simple_benchmark_results_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        
        logger.info(f"Results exported to {csv_filename}")
        
        # Show simple summary
        method_columns = [col for col in df.columns if col.endswith('_Count')]
        
        logger.info("\nSUMMARY:")
        for col in method_columns:
            method_name = col.replace('_Count', '')
            avg_count = df[col].mean()
            success_rate = (sum(df[col] > 0) / len(df)) * 100
            logger.info(f"{method_name:15}: {success_rate:5.1f}% success, {avg_count:4.1f} avg tables")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")

# --- Database Testing Function ---
def test_database_connection(db_path: str = "knowledge_graph.duckdb"):
    """Test database connection with detailed diagnostics."""
    logger.info("=" * 60)
    logger.info("Testing Database Connection")
    logger.info("=" * 60)
    
    # Check file existence
    import os
    if not os.path.exists(db_path):
        logger.error(f"Database file not found: {db_path}")
        logger.error("Please copy knowledge_graph.duckdb to this directory")
        return False
    
    file_size = os.path.getsize(db_path) / (1024 * 1024)  # MB
    logger.info(f"Database file: {db_path} ({file_size:.2f} MB)")
    
    # Test DuckDB import
    try:
        import duckdb
        logger.info(f"DuckDB version: {duckdb.__version__}")
    except ImportError as e:
        logger.error(f"DuckDB import failed: {e}")
        return False
    
    # Test connections
    connection_methods = [
        ("Direct connection", lambda: duckdb.connect(db_path)),
        ("Read-only connection", lambda: duckdb.connect(f"{db_path}?access_mode=read_only")),
        ("Memory connection", lambda: duckdb.connect(":memory:")),
    ]
    
    working_connection = None
    
    for method_name, connect_func in connection_methods:
        try:
            logger.info(f"Testing {method_name}...")
            conn = connect_func()
            
            # Test basic query
            result = conn.execute("SELECT 'test' as result").fetchone()
            logger.info(f"  Basic query: {result}")
            
            # If this is file connection, test table access
            if "memory" not in method_name.lower():
                try:
                    tables = conn.execute("SHOW TABLES").fetchall()
                    logger.info(f"  Found {len(tables)} tables")
                    
                    if tables:
                        # Test accessing first table
                        first_table = tables[0][0]
                        count = conn.execute(f"SELECT COUNT(*) FROM {first_table}").fetchone()[0]
                        logger.info(f"  Table {first_table}: {count} rows")
                        
                        working_connection = conn
                        logger.info(f"  {method_name}: SUCCESS")
                        break
                except Exception as e:
                    logger.warning(f"  Table access failed: {e}")
            else:
                logger.info(f"  {method_name}: SUCCESS (basic)")
            
            conn.close()
            
        except Exception as e:
            logger.warning(f"  {method_name}: FAILED - {e}")
    
    if working_connection:
        working_connection.close()
        logger.info("Database connection test: PASSED")
        return True
    else:
        logger.error("Database connection test: FAILED")
        return False

# --- Main Benchmark Function ---
def run_simple_benchmark():
    """Run simple benchmark on existing DuckDB database."""
    
    print("=" * 60)
    print("Simple DuckDB Table Retrieval Benchmark")
    print("=" * 60)
    
    # Test database first
    DB_PATH = "knowledge_graph.duckdb"
    if not test_database_connection(DB_PATH):
        logger.error("Database connection test failed. Cannot proceed.")
        return
    
    # Load queries
    queries = load_test_queries()
    if not queries:
        logger.error("No queries loaded")
        return
    
    # Initialize retrievers with enhanced error handling
    retrievers = {}
    
    print(f"\nInitializing retrieval methods...")
    
    retriever_classes = [
        ("Keyword", KeywordRetriever),
        ("Column", ColumnRetriever), 
        ("Relationship", RelationshipRetriever),
        ("Pattern", PatternRetriever)
    ]
    
    for name, retriever_class in retriever_classes:
        try:
            logger.info(f"Initializing {name} retriever...")
            retriever = retriever_class(DB_PATH)
            retrievers[name] = retriever
            print(f"  - {name} Search: OK")
        except Exception as e:
            logger.error(f"Failed to initialize {name} retriever: {e}")
            print(f"  - {name} Search: FAILED ({e})")
    
    if not retrievers:
        logger.error("No retrieval methods initialized successfully")
        return
    
    print(f"\nRunning benchmark with {len(retrievers)} methods on {len(queries)} queries...")
    
    # Run benchmark
    results = []
    
    for i, query_info in enumerate(queries, 1):
        query_id = query_info['id']
        question = query_info['question']
        
        print(f"\n[{i}/{len(queries)}] {query_id}: {question[:60]}...")
        
        query_results = {
            'Query_ID': query_id,
            'Question': question,
            'Source': query_info['source']
        }
        
        for method_name, retriever in retrievers.items():
            try:
                start_time = datetime.now()
                tables = retriever.get_tables(question)
                end_time = datetime.now()
                
                duration = (end_time - start_time).total_seconds()
                
                # Clean results
                clean_tables = []
                if tables:
                    seen = set()
                    for table in tables:
                        if table and table.strip() and table not in seen:
                            clean_tables.append(table.strip())
                            seen.add(table)
                
                tables_str = "; ".join(clean_tables) if clean_tables else "No tables found"
                query_results[f'{method_name}_Tables'] = tables_str
                query_results[f'{method_name}_Count'] = len(clean_tables)
                query_results[f'{method_name}_Duration_sec'] = round(duration, 3)
                
                print(f"  {method_name:12}: {len(clean_tables):2d} tables ({duration:5.3f}s)")
                
            except Exception as e:
                query_results[f'{method_name}_Tables'] = f"ERROR: {str(e)}"
                query_results[f'{method_name}_Count'] = 0
                query_results[f'{method_name}_Duration_sec'] = 0
                print(f"  {method_name:12}: FAILED - {e}")
        
        results.append(query_results)
    
    # Export results
    export_results(results)
    
    print(f"\n{'=' * 60}")
    print("Benchmark completed successfully!")
    print(f"Processed {len(queries)} queries with {len(retrievers)} methods")
    print(f"{'=' * 60}")

def main():
    """Main function."""
    try:
        run_simple_benchmark()
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
