import gradio as gr
import json
import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
import re
from collections import Counter
import duckdb
import asyncio
import time

# AI/ML Libraries
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ sklearn/sentence_transformers not available - Vector and TF-IDF methods disabled")

# Import your client manager if available
try:
    from app.utils.client_manager import client_manager
    CLIENT_MANAGER_AVAILABLE = True
except ImportError:
    CLIENT_MANAGER_AVAILABLE = False
    print("⚠️ client_manager not available - LLM methods may be limited")

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Configuration
UAT_H2O_API_KEY = os.getenv("UAT_H2O_API_KEY", "sk-rerYub6aZ0ypytPg7FMQwbfe129h3oh1UeIA0UNX5Z7yYUyS")
UAT_AZURE_API_URL = os.getenv("UAT_AZURE_API_URL", "https://r2d2-c3p0-icg-msst-genaihub-178909.apps.namicg39023u.ecs.dyn.nsroot.net/azure")
UAT_VERTEX_API_URL = os.getenv("UAT_VERTEX_API_URL", "https://r2d2-c3p0-icg-msst-genaihub-178909.apps.namicg39023u.ecs.dyn.nsroot.net/vertex")

# Global variables for caching
embedding_model = None
retrievers = {}
db_stats = {}

# --- Base Retriever Class ---
class BaseDuckDBRetriever:
    """Base class for all DuckDB-based retrieval methods."""
    def __init__(self, db_path: str = "knowledge_graph.duckdb"):
        try:
            self.conn = duckdb.connect(db_path)
            # Test connection
            self.conn.execute("SELECT COUNT(*) FROM tables").fetchone()
            logger.info(f"✅ Connected to DuckDB for {self.__class__.__name__}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to DuckDB: {e}")
            raise
            
    def get_tables(self, query: str) -> List[str]:
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_method_name(self) -> str:
        return self.__class__.__name__.replace('Retriever', '')

# --- Method 1: Enhanced Keyword Search ---
class KeywordRetriever(BaseDuckDBRetriever):
    """Enhanced keyword matching using SQL queries."""
    
    def get_tables(self, query: str) -> List[str]:
        logger.info(f"🔍 Running Enhanced Keyword search for: '{query[:50]}...'")
        
        # Extract key terms from query
        key_terms = self._extract_key_terms(query)
        
        tables = set()
        
        # Strategy 1: Direct table search
        tables.update(self._direct_table_search(query, key_terms))
        
        # Strategy 2: Column-based search
        tables.update(self._column_based_search(key_terms))
        
        # Strategy 3: Pattern-based search
        tables.update(self._pattern_based_search(query))
        
        return list(tables)[:10]  # Limit to top 10
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key business terms from query."""
        # Common financial/trading terms
        financial_terms = [
            'trade', 'trader', 'execution', 'venue', 'product', 'currency', 
            'notional', 'price', 'cusip', 'ticker', 'counterparty', 'etd',
            'source', 'system', 'business_date', 'government', 'entity'
        ]
        
        query_lower = query.lower()
        found_terms = [term for term in financial_terms if term in query_lower]
        
        # Also extract column-like terms (UPPERCASE_WORDS)
        uppercase_terms = re.findall(r'\b[A-Z][A-Z_]+[A-Z]\b', query)
        
        return found_terms + [term.lower() for term in uppercase_terms]
    
    def _direct_table_search(self, query: str, key_terms: List[str]) -> List[str]:
        """Direct search on table metadata."""
        try:
            # Search in table names and descriptions
            sql = """
            SELECT DISTINCT name FROM tables
            WHERE LOWER(name) LIKE ? 
               OR LOWER(description) LIKE ?
               OR LOWER(rules) LIKE ?
            LIMIT 5
            """
            
            search_term = f"%{query.lower()}%"
            results = self.conn.execute(sql, [search_term, search_term, search_term]).fetchall()
            return [row[0] for row in results]
        except Exception as e:
            logger.warning(f"Direct table search failed: {e}")
            return []
    
    def _column_based_search(self, key_terms: List[str]) -> List[str]:
        """Find tables via column name matching."""
        tables = []
        for term in key_terms:
            try:
                sql = """
                SELECT DISTINCT table_name FROM columns
                WHERE LOWER(name) LIKE ? OR LOWER(full_name) LIKE ?
                LIMIT 3
                """
                results = self.conn.execute(sql, [f"%{term}%", f"%{term}%"]).fetchall()
                tables.extend([row[0] for row in results])
            except:
                continue
        return tables
    
    def _pattern_based_search(self, query: str) -> List[str]:
        """Pattern-based table identification."""
        patterns = {
            'fact': r'\b(trade|transaction|execution|market|intra.?day)\b',
            'dimension': r'\b(product|trader|venue|calendar|date|reference)\b',
            'reference': r'\b(ref|lookup|master|code|type)\b'
        }
        
        tables = []
        query_lower = query.lower()
        
        for table_type, pattern in patterns.items():
            if re.search(pattern, query_lower):
                try:
                    sql = "SELECT name FROM tables WHERE table_type = ? LIMIT 3"
                    results = self.conn.execute(sql, [table_type]).fetchall()
                    tables.extend([row[0] for row in results])
                except:
                    continue
        
        return tables

# --- Method 2: TF-IDF Similarity Search ---
class TFIDFRetriever(BaseDuckDBRetriever):
    """TF-IDF based similarity search on table/column descriptions."""
    
    def __init__(self, db_path: str = "knowledge_graph.duckdb"):
        super().__init__(db_path)
        if SKLEARN_AVAILABLE:
            self._build_corpus()
        else:
            self.table_corpus = {}
    
    def _build_corpus(self):
        """Build TF-IDF corpus from table and column descriptions."""
        try:
            # Get all table and column descriptions
            sql = """
            SELECT t.name as table_name,
                   t.description as table_desc,
                   STRING_AGG(c.name || ': ' || COALESCE(c.description, ''), ' ') as column_descs
            FROM tables t
            LEFT JOIN columns c ON t.name = c.table_name
            GROUP BY t.name, t.description
            """
            
            results = self.conn.execute(sql).fetchall()
            
            self.table_corpus = {}
            corpus_texts = []
            
            for table_name, table_desc, column_descs in results:
                # Combine table and column descriptions
                text = f"{table_desc or ''} {column_descs or ''}"
                self.table_corpus[table_name] = text
                corpus_texts.append(text)
            
            # Build TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)
            self.table_names = list(self.table_corpus.keys())
            
            logger.info(f"✅ Built TF-IDF corpus with {len(self.table_names)} tables")
            
        except Exception as e:
            logger.error(f"Failed to build TF-IDF corpus: {e}")
            self.table_corpus = {}
    
    def get_tables(self, query: str) -> List[str]:
        logger.info(f"📊 Running TF-IDF search for: '{query[:50]}...'")
        
        if not SKLEARN_AVAILABLE or not self.table_corpus:
            return []
        
        try:
            # Transform query to TF-IDF vector
            query_vector = self.vectorizer.transform([query])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top tables
            top_indices = similarities.argsort()[-10:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.01:  # Minimum threshold
                    results.append(self.table_names[idx])
            
            return results
            
        except Exception as e:
            logger.error(f"TF-IDF search failed: {e}")
            return []

# --- Method 3: Graph Traversal Search ---
class GraphTraversalRetriever(BaseDuckDBRetriever):
    """Uses relationship structure for table discovery."""
    
    def get_tables(self, query: str) -> List[str]:
        logger.info(f"🕸️ Running Graph Traversal search for: '{query[:50]}...'")
        
        tables = set()
        
        # Strategy 1: Find seed tables via keywords
        seed_tables = self._find_seed_tables(query)
        tables.update(seed_tables)
        
        # Strategy 2: Expand via relationships
        for seed_table in seed_tables[:3]:  # Limit expansion
            related_tables = self._find_related_tables(seed_table)
            tables.update(related_tables)
        
        return list(tables)[:10]
    
    def _find_seed_tables(self, query: str) -> List[str]:
        """Find initial candidate tables."""
        query_terms = query.lower().split()
        seed_tables = []
        
        try:
            for term in query_terms:
                if len(term) > 3:  # Skip short words
                    sql = """
                    SELECT name FROM tables
                    WHERE LOWER(name) LIKE ? OR LOWER(description) LIKE ?
                    LIMIT 2
                    """
                    results = self.conn.execute(sql, [f"%{term}%", f"%{term}%"]).fetchall()
                    seed_tables.extend([row[0] for row in results])
        except:
            pass
        
        return seed_tables
    
    def _find_related_tables(self, table_name: str) -> List[str]:
        """Find tables related via joins."""
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

# --- Method 4: OpenAI GPT-4 Retriever ---
class OpenAIRetriever(BaseDuckDBRetriever):
    """Uses OpenAI GPT-4 for intelligent table selection."""
    
    def __init__(self, db_path: str = "knowledge_graph.duckdb"):
        super().__init__(db_path)
        self.schema_summary = self._get_enhanced_schema_summary()
    
    def _get_enhanced_schema_summary(self) -> str:
        """Get detailed schema with table and key column information."""
        try:
            sql = """
            SELECT t.name as table_name,
                   t.description as table_desc,
                   t.table_type,
                   STRING_AGG(DISTINCT c.name, ', ') as key_columns,
                   COUNT(c.id) as column_count
            FROM tables t
            LEFT JOIN columns c ON t.name = c.table_name
            WHERE c.column_category IN ('id', 'key', 'code', 'measure') 
               OR c.name ILIKE '%date%' 
               OR c.name ILIKE '%trader%'
               OR c.name ILIKE '%product%'
               OR c.name ILIKE '%price%'
               OR c.name ILIKE '%currency%'
            GROUP BY t.name, t.description, t.table_type
            ORDER BY t.table_type, t.name
            """
            
            results = self.conn.execute(sql).fetchall()
            
            schema_text = []
            for table_name, table_desc, table_type, key_columns, col_count in results:
                table_line = f"• {table_name} ({table_type})"
                if table_desc:
                    table_line += f": {table_desc}"
                if key_columns:
                    table_line += f" [Key columns: {key_columns}]"
                table_line += f" [{col_count} total columns]"
                schema_text.append(table_line)
            
            return "\n".join(schema_text)
        except Exception as e:
            logger.error(f"Failed to get schema summary: {e}")
            return "Schema unavailable"
    
    def get_tables(self, query: str) -> List[str]:
        logger.info(f"🤖 Running OpenAI GPT-4 search for: '{query[:50]}...'")
        
        if not CLIENT_MANAGER_AVAILABLE:
            return []
        
        system_prompt = """You are an expert database analyst specializing in financial trading data systems. Your task is to identify the most relevant database tables for answering user queries.

Key Guidelines:
1. FACT tables contain transactional data (trades, executions, market data)
2. DIMENSION tables contain reference data (products, traders, venues, dates)
3. REFERENCE tables contain lookup data (codes, types, mappings)
4. Consider data relationships and join patterns
5. Focus on tables that directly contain the requested information
6. Return ONLY table names, comma-separated
7. Maximum 5 most relevant tables
8. Prioritize precision over coverage"""

        user_prompt = f"""
**Financial Data Query:**
{query}

**Available Database Tables:**
{self.schema_summary}

**Analysis Instructions:**
1. Identify the main business entities mentioned (trades, products, traders, venues, etc.)
2. Determine if this is a fact-based query (transaction data) or dimension-based query (reference data)
3. Consider temporal aspects (dates, time periods)
4. Think about necessary joins between fact and dimension tables

**Most Relevant Table Names (comma-separated):**"""
        
        try:
            # Use the internal client manager
            full_message = f"{system_prompt}\n\n{user_prompt}"
            
            # Call the async function synchronously
            response = asyncio.run(client_manager.ask_gpt(full_message))
            
            logger.info(f"OpenAI Response received: {len(response)} characters")
            
            # Parse the response
            tables = []
            for name in response.split(','):
                clean_name = name.strip().strip('"\'')
                if clean_name and not clean_name.lower().startswith(('based', 'the', 'to', 'for', 'analysis', 'instructions')):
                    tables.append(clean_name)
            
            return tables[:10]
            
        except Exception as e:
            logger.error(f"OpenAI retrieval failed: {e}")
            return []

# --- Method 5: Google Gemini Retriever ---
class GeminiRetriever(BaseDuckDBRetriever):
    """Uses Google Gemini for intelligent table selection."""
    
    def __init__(self, db_path: str = "knowledge_graph.duckdb"):
        super().__init__(db_path)
        self.schema_summary = self._get_enhanced_schema_summary()
    
    def _get_enhanced_schema_summary(self) -> str:
        """Get detailed schema with business context - optimized for Gemini."""
        try:
            sql = """
            SELECT t.name as table_name,
                   t.description as table_desc,
                   t.table_type
            FROM tables t
            ORDER BY 
                CASE t.table_type 
                    WHEN 'fact' THEN 1 
                    WHEN 'dimension' THEN 2 
                    WHEN 'reference' THEN 3 
                    ELSE 4 
                END, t.name
            LIMIT 30
            """
            
            results = self.conn.execute(sql).fetchall()
            
            schema_sections = {"fact": [], "dimension": [], "reference": []}
            
            for table_name, table_desc, table_type in results:
                # Keep only essential info for Gemini
                table_line = f"{table_name}"
                if table_desc and len(table_desc) > 10:
                    table_line += f" - {table_desc[:60]}"
                
                section = table_type if table_type in schema_sections else "fact"
                schema_sections[section].append(table_line)
            
            # Create more focused schema text
            schema_text = []
            if schema_sections["fact"]:
                schema_text.append("FACT TABLES:")
                schema_text.extend(schema_sections["fact"][:8])  # Limit to 8 tables
            
            if schema_sections["dimension"]:
                schema_text.append("\nDIMENSION TABLES:")
                schema_text.extend(schema_sections["dimension"][:8])
            
            if schema_sections["reference"]:
                schema_text.append("\nREFERENCE TABLES:")
                schema_text.extend(schema_sections["reference"][:6])
            
            return "\n".join(schema_text)
        except Exception as e:
            logger.error(f"Failed to get schema summary: {e}")
            return "Schema unavailable"
    
    def _parse_and_validate_tables(self, response_text: str) -> List[str]:
        """Parse response and validate table names against actual schema."""
        # Get all actual table names from database
        try:
            sql = "SELECT DISTINCT name FROM tables"
            actual_tables = set([row[0] for row in self.conn.execute(sql).fetchall()])
        except:
            actual_tables = set()
        
        # Parse the response
        tables = []
        
        # Split by common separators
        for separator in [',', '\n', ';']:
            if separator in response_text:
                candidate_names = response_text.split(separator)
                break
        else:
            candidate_names = [response_text]
        
        for name in candidate_names:
            clean_name = name.strip().strip('"\'').strip()
            
            # Skip empty or explanatory text
            if not clean_name or len(clean_name) < 3:
                continue
                
            # Skip common explanatory words
            skip_words = ['table', 'based', 'the', 'to', 'for', 'with', 'contains', 
                         'provides', 'analysis', 'framework', 'selected', 'names',
                         'response', 'format', 'task', 'tables:']
            
            if any(word in clean_name.lower() for word in skip_words):
                continue
            
            # Check if it's an actual table name (exact match)
            if clean_name in actual_tables:
                tables.append(clean_name)
            else:
                # Try to find close matches (in case of minor variations)
                for actual_table in actual_tables:
                    if actual_table.upper() == clean_name.upper():
                        tables.append(actual_table)
                        break
                    elif clean_name.upper() in actual_table.upper():
                        tables.append(actual_table)
                        break
        
        logger.info(f"Validated Gemini tables: {tables}")
        return tables
    
    def get_tables(self, query: str) -> List[str]:
        logger.info(f"💎 Running Google Gemini search for: '{query[:50]}...'")
        
        if not CLIENT_MANAGER_AVAILABLE:
            return []
        
        prompt = f"""You are a database expert. Your task is to select the most relevant table names from the provided schema.

**Task:** Select the most relevant EXACT table names from the schema above that would help answer the user query.

**User Query:**
{query}

**Available Database Schema:**
{self.schema_summary}

**Analysis Framework:**
1. **Query Type Analysis**: Determine if this is asking for:
   - Transactional data (trades, executions) → Use FACT tables
   - Reference data (products, traders, venues) → Use DIMENSION tables
   - Lookup information (codes, mappings) → Use REFERENCE tables

2. **Business Context**: Consider:
   - Trading entities: What instruments, traders, venues are mentioned?
   - Temporal aspects: Any date ranges or time periods?
   - Metrics: What calculations or aggregations are needed?
   - Relationships: What joins between tables are implied?

3. **Table Selection Strategy**:
   - Start with the core table containing the main data
   - Add dimension tables for context and filtering
   - Include reference tables for lookups and descriptions

**Instructions:**
- Return ONLY the table names, separated by commas
- Maximum 5 tables
- Prioritize tables that directly answer the query
- Consider necessary join relationships

**Selected Table Names:**"""
        
        try:
            # Use the internal client manager
            response = asyncio.run(client_manager.ask_vertexai(prompt))
            
            logger.info(f"Gemini Response received: {len(response)} characters")
            
            tables = self._parse_and_validate_tables(response)
            return tables[:10]
            
        except Exception as e:
            logger.error(f"Gemini retrieval failed: {e}")
            return []

# --- Utility Functions ---
def initialize_retrievers():
    """Initialize all available retrieval methods."""
    global retrievers, db_stats
    
    DB_PATH = "knowledge_graph.duckdb"
    
    if not os.path.exists(DB_PATH):
        return False, "❌ DuckDB file 'knowledge_graph.duckdb' not found!"
    
    retrievers = {}
    
    try:
        # Always available methods
        retrievers["Keyword"] = KeywordRetriever(DB_PATH)
        retrievers["Graph Traversal"] = GraphTraversalRetriever(DB_PATH)
        
        # TF-IDF (requires sklearn)
        if SKLEARN_AVAILABLE:
            retrievers["TF-IDF"] = TFIDFRetriever(DB_PATH)
        
        # LLM methods (require client manager)
        if CLIENT_MANAGER_AVAILABLE:
            retrievers["GPT-4"] = OpenAIRetriever(DB_PATH)
            retrievers["Gemini"] = GeminiRetriever(DB_PATH)
        
        # Get database statistics
        conn = duckdb.connect(DB_PATH)
        table_count = conn.execute("SELECT COUNT(*) FROM tables").fetchone()[0]
        column_count = conn.execute("SELECT COUNT(*) FROM columns").fetchone()[0]
        
        db_stats = {
            "tables": table_count,
            "columns": column_count,
            "file_size_mb": round(os.path.getsize(DB_PATH) / (1024*1024), 2)
        }
        
        conn.close()
        
        return True, f"✅ Initialized {len(retrievers)} retrieval methods"
        
    except Exception as e:
        return False, f"❌ Failed to initialize retrievers: {e}"

def query_tables(user_query: str, selected_methods: List[str]):
    """Execute table retrieval using selected methods."""
    if not user_query.strip():
        return "Please enter a query.", {}, ""
    
    if not selected_methods:
        return "Please select at least one retrieval method.", {}, ""
    
    results = {}
    performance_data = []
    status_messages = []
    
    for method_name in selected_methods:
        if method_name not in retrievers:
            status_messages.append(f"⚠️ {method_name} not available")
            continue
        
        try:
            start_time = time.time()
            tables = retrievers[method_name].get_tables(user_query)
            end_time = time.time()
            
            duration = round(end_time - start_time, 3)
            
            # Clean and format results
            if tables:
                unique_tables = []
                seen = set()
                for table in tables:
                    if table and table.strip() and table.strip() not in seen:
                        clean_table = table.strip()
                        unique_tables.append(clean_table)
                        seen.add(clean_table)
                tables = unique_tables
            
            results[method_name] = {
                "tables": tables,
                "count": len(tables),
                "duration": duration
            }
            
            performance_data.append({
                "Method": method_name,
                "Tables Found": len(tables),
                "Duration (s)": duration,
                "Status": "✅ Success"
            })
            
            status_messages.append(f"✅ {method_name}: {len(tables)} tables ({duration}s)")
            
        except Exception as e:
            results[method_name] = {
                "tables": [],
                "count": 0,
                "duration": 0,
                "error": str(e)
            }
            
            performance_data.append({
                "Method": method_name,
                "Tables Found": 0,
                "Duration (s)": 0,
                "Status": f"❌ Error: {str(e)[:50]}..."
            })
            
            status_messages.append(f"❌ {method_name}: Failed - {e}")
    
    # Create summary
    total_methods = len(selected_methods)
    successful_methods = len([r for r in results.values() if "error" not in r])
    total_unique_tables = len(set(table for r in results.values() 
                                 if "error" not in r for table in r["tables"]))
    
    summary = f"""
## Query Results Summary

**Query:** {user_query}

**Performance:**
- Methods executed: {successful_methods}/{total_methods}
- Total unique tables found: {total_unique_tables}
- Database stats: {db_stats.get('tables', 'N/A')} tables, {db_stats.get('columns', 'N/A')} columns

**Status Messages:**
{chr(10).join(status_messages)}
"""
    
    # Create performance DataFrame
    perf_df = pd.DataFrame(performance_data)
    
    return summary, results, perf_df

def format_results_display(results: Dict):
    """Format results for display in the UI."""
    if not results:
        return "No results to display."
    
    display_text = ""
    
    for method_name, result in results.items():
        display_text += f"\n## 🔍 {method_name}\n"
        
        if "error" in result:
            display_text += f"❌ **Error:** {result['error']}\n"
            continue
        
        tables = result["tables"]
        count = result["count"]
        duration = result["duration"]
        
        display_text += f"**Found {count} tables in {duration}s**\n\n"
        
        if tables:
            for i, table in enumerate(tables, 1):
                display_text += f"{i}. `{table}`\n"
        else:
            display_text += "*No tables found*\n"
        
        display_text += "\n---\n"
    
    return display_text

def get_database_info():
    """Get database information for display."""
    if not db_stats:
        return "Database not initialized."
    
    info = f"""
## 📊 Database Information

- **Tables:** {db_stats.get('tables', 'N/A')}
- **Columns:** {db_stats.get('columns', 'N/A')}
- **File Size:** {db_stats.get('file_size_mb', 'N/A')} MB
- **Available Methods:** {len(retrievers)}

### 🔧 Available Retrieval Methods:
"""
    
    for method_name in retrievers.keys():
        if method_name in ["GPT-4", "Gemini"]:
            info += f"- 🤖 **{method_name}** (LLM-powered)\n"
        else:
            info += f"- ⚡ **{method_name}** (Local processing)\n"
    
    return info

# --- Gradio Interface ---
def create_interface():
    """Create the Gradio web interface."""
    
    # Initialize retrievers
    init_success, init_message = initialize_retrievers()
    
    with gr.Blocks(title="DuckDB Table Query System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🗄️ DuckDB Table Query System
        
        **Intelligent table retrieval for financial trading data using multiple AI methods**
        
        Enter your query below and select which retrieval methods you'd like to use. 
        The system will analyze your query and return the most relevant database tables.
        """)
        
        # Status display
        with gr.Row():
            status_display = gr.Markdown(f"**System Status:** {init_message}")
        
        # Database info
        with gr.Row():
            db_info = gr.Markdown(get_database_info())
        
        # Main interface
        with gr.Row():
            with gr.Column(scale=2):
                # Query input
                query_input = gr.Textbox(
                    label="📝 Enter your query",
                    placeholder="Example: give me distinct source systems for cash ETD trades for yesterday",
                    lines=3,
                    value=""
                )
                
                # Method selection
                available_methods = list(retrievers.keys()) if init_success else []
                method_selection = gr.CheckboxGroup(
                    choices=available_methods,
                    label="🔧 Select Retrieval Methods",
                    value=available_methods[:3] if available_methods else [],  # Select first 3 by default
                    info="Choose which methods to use for table retrieval"
                )
                
                # Submit button
                submit_btn = gr.Button("🚀 Search Tables", variant="primary", size="lg")
                
                # Example queries
                gr.Markdown("""
                ### 💡 Example Queries:
                - `give me distinct source systems for cash ETD trades for yesterday`
                - `show me EXECUTING_TRADER_SOEID EXECUTION_VENUE PRODUCT_SK where TRADE_PRICE_CURRENCY and ISSUE_CURRENCY are not the same`
                - `Show me the counterparty for trade ID 18871106`
                - `get me the CUSIP that was traded highest last week`
                - `show me all trades by government entities`
                """)
            
            with gr.Column(scale=3):
                # Results display
                results_summary = gr.Markdown("Enter a query and click 'Search Tables' to see results.")
                
                # Performance metrics
                performance_table = gr.DataFrame(
                    headers=["Method", "Tables Found", "Duration (s)", "Status"],
                    label="📊 Performance Metrics"
                )
        
        # Detailed results section
        with gr.Row():
            with gr.Column():
                results_display = gr.Markdown("", label="📋 Detailed Results")
        
        # Example buttons for quick testing
        with gr.Row():
            gr.Markdown("### 🎯 Quick Test Examples:")
        
        with gr.Row():
            example_btn1 = gr.Button("ETD Trades Query", size="sm")
            example_btn2 = gr.Button("Currency Mismatch Query", size="sm") 
            example_btn3 = gr.Button("Government Entities Query", size="sm")
            example_btn4 = gr.Button("CUSIP Highest Volume Query", size="sm")
        
        # Event handlers
        def run_query(query, methods):
            if not init_success:
                return "❌ System not initialized properly. Please check DuckDB file.", pd.DataFrame(), ""
            
            summary, results, perf_df = query_tables(query, methods)
            detailed_results = format_results_display(results)
            
            return summary, perf_df, detailed_results
        
        # Main submit handler
        submit_btn.click(
            fn=run_query,
            inputs=[query_input, method_selection],
            outputs=[results_summary, performance_table, results_display]
        )
        
        # Example button handlers
        def set_example_1():
            return "give me distinct source systems for cash ETD trades for yesterday"
        
        def set_example_2():
            return "show me EXECUTING_TRADER_SOEID EXECUTION_VENUE PRODUCT_SK where TRADE_PRICE_CURRENCY and ISSUE_CURRENCY are not the same"
        
        def set_example_3():
            return "show me all trades by government entities"
        
        def set_example_4():
            return "get me the CUSIP that was traded highest last week"
        
        example_btn1.click(fn=set_example_1, outputs=query_input)
        example_btn2.click(fn=set_example_2, outputs=query_input)
        example_btn3.click(fn=set_example_3, outputs=query_input)
        example_btn4.click(fn=set_example_4, outputs=query_input)
        
        # Footer
        gr.Markdown("""
        ---
        ### 🔧 Technical Details:
        
        **Retrieval Methods:**
        - **Keyword**: Fast pattern matching using SQL queries
        - **TF-IDF**: Text similarity using term frequency analysis
        - **Graph Traversal**: Relationship-based table discovery
        - **GPT-4**: AI-powered contextual understanding
        - **Gemini**: Google's AI for alternative perspective
        
        **Performance Notes:**
        - Local methods (Keyword, TF-IDF, Graph) are fastest
        - LLM methods (GPT-4, Gemini) provide better context understanding
        - Results are cached for faster subsequent queries
        
        Built with ❤️ using DuckDB, Gradio, and multiple AI models.
        """)
    
    return demo

# --- Advanced Features ---
def export_results_to_csv(results: Dict, query: str):
    """Export results to CSV format."""
    if not results:
        return None
    
    # Flatten results for CSV export
    csv_data = []
    
    for method_name, result in results.items():
        if "error" not in result:
            for i, table in enumerate(result["tables"], 1):
                csv_data.append({
                    "Query": query,
                    "Method": method_name,
                    "Rank": i,
                    "Table_Name": table,
                    "Duration_sec": result["duration"],
                    "Total_Tables_Found": result["count"],
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        filename = f"table_query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        return filename
    
    return None

def get_table_details(table_name: str):
    """Get detailed information about a specific table."""
    if "Keyword" not in retrievers:
        return "No database connection available."
    
    try:
        conn = retrievers["Keyword"].conn
        
        # Get table info
        table_sql = "SELECT * FROM tables WHERE name = ?"
        table_info = conn.execute(table_sql, [table_name]).fetchone()
        
        if not table_info:
            return f"Table '{table_name}' not found in database."
        
        # Get column info
        columns_sql = "SELECT name, data_type, description FROM columns WHERE table_name = ? ORDER BY id"
        columns = conn.execute(columns_sql, [table_name]).fetchall()
        
        # Format output
        details = f"""
## 📋 Table Details: {table_name}

**Description:** {table_info[2] if len(table_info) > 2 else 'No description available'}
**Type:** {table_info[3] if len(table_info) > 3 else 'Unknown'}

### 📊 Columns ({len(columns)}):
"""
        
        for col_name, data_type, description in columns:
            details += f"- **{col_name}** ({data_type})"
            if description:
                details += f": {description}"
            details += "\n"
        
        return details
        
    except Exception as e:
        return f"Error getting table details: {e}"

# --- Main Application ---
def main():
    """Main function to launch the web application."""
    print("🚀 Starting DuckDB Table Query Web Interface...")
    
    # Check for required files
    if not os.path.exists("knowledge_graph.duckdb"):
        print("❌ DuckDB file 'knowledge_graph.duckdb' not found!")
        print("💡 Please run the DuckDB knowledge graph builder first.")
        return
    
    # Create and launch interface
    demo = create_interface()
    
    print("✅ Web interface created successfully!")
    print("🌐 Launching on http://localhost:7860")
    print("📊 Available features:")
    print("   - Multiple retrieval methods (Keyword, TF-IDF, Graph, GPT-4, Gemini)")
    print("   - Real-time performance metrics")
    print("   - Interactive query examples")
    print("   - Detailed results display")
    
    # Launch with custom settings
    demo.launch(
    server_name="0.0.0.0",  # Allow external access
    server_port=7860,
    share=False,  # Set to True for public sharing
    show_error=True
    )

if __name__ == "__main__":
    main()