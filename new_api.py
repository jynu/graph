Step 1: Move and Update Import

Move your duckdb_benchmark_llm.py to the same folder as others.py and client_manager.py
Update the import section at the top of your benchmark script:

pythonimport asyncio
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

# Import the internal client manager
from client_manager import client_manager

# Remove these external API imports since we're using internal APIs
# import openai
# import google.generativeai as genai

# Keep other imports for local methods
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
Step 2: Update OpenAI Retriever
Replace the entire OpenAIRetriever class:
pythonclass OpenAIRetriever(BaseDuckDBRetriever):
    """Uses internal OpenAI client for intelligent table selection."""
    
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
            LIMIT 50
            """
            
            results = self.conn.execute(sql).fetchall()
            
            schema_text = []
            for table_name, table_desc, table_type, key_columns, col_count in results:
                table_line = f"• {table_name} ({table_type})"
                if table_desc:
                    table_line += f": {table_desc[:100]}"
                if key_columns:
                    table_line += f" [Key columns: {key_columns[:50]}]"
                schema_text.append(table_line)
            
            return "\n".join(schema_text)
        except Exception as e:
            logger.error(f"Failed to get schema summary: {e}")
            return "Schema unavailable"
    
    def get_tables(self, query: str) -> List[str]:
        logger.info(f"🤖 Running Internal OpenAI GPT-4 search for: '{query[:50]}...'")
        
        system_prompt = """You are an expert database analyst specializing in financial trading data systems. Your task is to identify the most relevant database tables for answering user queries.

Key Guidelines:
1. FACT tables contain transactional data (trades, executions, market data)
2. DIMENSION tables contain reference data (products, traders, venues, dates)
3. REFERENCE tables contain lookup data (codes, types, mappings)
4. Focus on tables that directly contain the requested information
5. Return ONLY table names, comma-separated
6. Maximum 5 most relevant tables
7. Prioritize precision over coverage"""

        user_prompt = f"""
**Financial Data Query:**
{query}

**Available Database Tables:**
{self.schema_summary}

**Analysis Instructions:**
1. Identify main business entities (trades, products, traders, venues, etc.)
2. Determine if this needs transaction data or reference data
3. Consider temporal aspects and necessary joins

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
Step 3: Update Gemini Retriever
Replace the entire GeminiRetriever class:
pythonclass GeminiRetriever(BaseDuckDBRetriever):
    """Uses internal Vertex AI client for intelligent table selection."""
    
    def __init__(self, db_path: str = "knowledge_graph.duckdb"):
        super().__init__(db_path)
        self.schema_summary = self._get_enhanced_schema_summary()
    
    def _get_enhanced_schema_summary(self) -> str:
        """Get detailed schema with business context."""
        try:
            sql = """
            SELECT t.name as table_name,
                   t.description as table_desc,
                   t.table_type,
                   STRING_AGG(DISTINCT c.name, ', ') as important_columns,
                   COUNT(c.id) as total_columns
            FROM tables t
            LEFT JOIN columns c ON t.name = c.table_name
            WHERE c.column_category IN ('id', 'key', 'code', 'measure', 'date') 
               OR c.name ILIKE ANY(ARRAY['%trader%', '%product%', '%price%', '%currency%', '%venue%', '%notional%', '%cusip%', '%ticker%'])
            GROUP BY t.name, t.description, t.table_type
            ORDER BY 
                CASE t.table_type 
                    WHEN 'fact' THEN 1 
                    WHEN 'dimension' THEN 2 
                    WHEN 'reference' THEN 3 
                    ELSE 4 
                END, t.name
            LIMIT 50
            """
            
            results = self.conn.execute(sql).fetchall()
            
            schema_sections = {"fact": [], "dimension": [], "reference": [], "other": []}
            
            for table_name, table_desc, table_type, important_columns, total_columns in results:
                table_line = f"  • {table_name}"
                if table_desc:
                    table_line += f" - {table_desc[:80]}"
                if important_columns:
                    table_line += f" (Key fields: {important_columns[:40]})"
                
                section = table_type if table_type in schema_sections else "other"
                schema_sections[section].append(table_line)
            
            schema_text = []
            if schema_sections["fact"]:
                schema_text.append("FACT TABLES (Transaction Data):")
                schema_text.extend(schema_sections["fact"][:10])
            
            if schema_sections["dimension"]:
                schema_text.append("\nDIMENSION TABLES (Reference Data):")
                schema_text.extend(schema_sections["dimension"][:10])
            
            if schema_sections["reference"]:
                schema_text.append("\nREFERENCE TABLES (Lookup Data):")
                schema_text.extend(schema_sections["reference"][:10])
            
            return "\n".join(schema_text)
        except Exception as e:
            logger.error(f"Failed to get schema summary: {e}")
            return "Schema unavailable"
    
    def get_tables(self, query: str) -> List[str]:
        logger.info(f"💎 Running Internal Gemini Vertex search for: '{query[:50]}...'")
        
        prompt = f"""You are a financial data expert analyzing database queries for trading systems.

**Task:** Identify the most relevant database tables to answer this query.

**User Query:**
{query}

**Available Database Schema:**
{self.schema_summary}

**Analysis Framework:**
1. **Query Type**: Determine if asking for:
   - Transactional data (trades, executions) → Use FACT tables
   - Reference data (products, traders, venues) → Use DIMENSION tables
   - Lookup information (codes, mappings) → Use REFERENCE tables

2. **Business Context**: Consider trading entities, temporal aspects, metrics, relationships

**Instructions:**
- Return ONLY the table names, separated by commas
- Maximum 5 tables
- Prioritize tables that directly answer the query

**Selected Table Names:**"""
        
        try:
            # Use the internal client manager
            response = asyncio.run(client_manager.ask_vertexai(prompt))
            
            logger.info(f"Gemini Response received: {len(response)} characters")
            
            # Parse the response
            tables = []
            for name in response.split(','):
                clean_name = name.strip().strip('"\'').strip()
                if clean_name and not any(word in clean_name.lower() for word in 
                    ['table', 'based', 'the', 'to', 'for', 'with', 'contains', 'provides', 'analysis', 'framework']):
                    tables.append(clean_name)
            
            return tables[:10]
            
        except Exception as e:
            logger.error(f"Gemini retrieval failed: {e}")
            return []
Step 4: Remove Azure OpenAI Retriever
Simply remove or comment out the entire AzureOpenAIRetriever class and its initialization in the main function.
Step 5: Update Initialization in Main Function
Replace the LLM initialization section:
python# Enhanced LLM Methods using internal client manager
try:
    retrievers["OpenAI GPT-4"] = OpenAIRetriever(DB_PATH)
    print(f"  ✅ Internal OpenAI GPT-4 initialized")
except Exception as e:
    print(f"  ❌ OpenAIRetriever failed: {e}")

try:
    retrievers["Gemini Vertex"] = GeminiRetriever(DB_PATH)
    print(f"  ✅ Internal Gemini Vertex initialized")
except Exception as e:
    print(f"  ❌ GeminiRetriever failed: {e}")

# Remove Azure OpenAI initialization completely