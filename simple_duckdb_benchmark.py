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

# --- Simple Base Retriever Class ---
class SimpleDuckDBRetriever:
    """Simplified base class for DuckDB-based retrieval methods."""
    def __init__(self, db_path: str = "knowledge_graph.duckdb"):
        try:
            import duckdb
            self.conn = duckdb.connect(db_path)
            # Test connection
            self.conn.execute("SELECT COUNT(*) FROM tables").fetchone()
            logger.info(f"Connected to DuckDB for {self.__class__.__name__}")
        except Exception as e:
            logger.error(f"Failed to connect to DuckDB: {e}")
            raise
            
    def get_tables(self, query: str) -> List[str]:
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_method_name(self) -> str:
        return self.__class__.__name__.replace('Retriever', '')

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
                results = self.conn.execute(sql, [f"%{term}%", f"%{term}%"]).fetchall()
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
                results = self.conn.execute(sql, [f"%{term}%", f"%{term}%"]).fetchall()
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
                results = self.conn.execute(sql, [f"%{term}%", f"%{term}%"]).fetchall()
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
            results = self.conn.execute(sql, [table_name, table_name, table_name]).fetchall()
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
                results = self.conn.execute(sql, [table_type, pattern]).fetchall()
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

# --- Main Benchmark Function ---
def run_simple_benchmark():
    """Run simple benchmark on existing DuckDB database."""
    
    print("=" * 60)
    print("Simple DuckDB Table Retrieval Benchmark")
    print("=" * 60)
    
    # Check if database exists
    import os
    DB_PATH = "knowledge_graph.duckdb"
    if not os.path.exists(DB_PATH):
        print(f"Database file not found: {DB_PATH}")
        print("Please copy the knowledge_graph.duckdb file to this directory")
        return
    
    # Load queries
    queries = load_test_queries()
    if not queries:
        logger.error("No queries loaded")
        return
    
    # Initialize retrievers
    retrievers = {}
    
    print(f"\nInitializing retrieval methods...")
    try:
        retrievers["Keyword"] = KeywordRetriever(DB_PATH)
        print(f"  - Keyword Search: OK")
    except Exception as e:
        print(f"  - Keyword Search: FAILED ({e})")
    
    try:
        retrievers["Column"] = ColumnRetriever(DB_PATH)
        print(f"  - Column Search: OK")
    except Exception as e:
        print(f"  - Column Search: FAILED ({e})")
    
    try:
        retrievers["Relationship"] = RelationshipRetriever(DB_PATH)
        print(f"  - Relationship Search: OK")
    except Exception as e:
        print(f"  - Relationship Search: FAILED ({e})")
    
    try:
        retrievers["Pattern"] = PatternRetriever(DB_PATH)
        print(f"  - Pattern Search: OK")
    except Exception as e:
        print(f"  - Pattern Search: FAILED ({e})")
    
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
