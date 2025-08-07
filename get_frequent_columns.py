# 1_get_frequent_columns.py
"""
Core Module 1: Get Frequent Columns
===================================
Input: Table name(s)
Output: Configuration file (tbc_XXX.conf) with discovered frequent columns

This module analyzes query logs and table metadata to discover frequently used columns
and generates a configuration file for subsequent processing. It uses a hybrid approach
combining rule-based analysis with optional LLM/GPT-4 analysis for complex queries.
"""

import os
import sys
import json
import logging
import asyncio
import configparser
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
import subprocess
import tempfile
from collections import Counter

# Try to import the internal client manager for GPT-4 calls
try:
    from app.utils.client_manager import client_manager
    LLM_CLIENT_AVAILABLE = True
    print("✅ Internal LLM client manager imported successfully")
except ImportError as e:
    LLM_CLIENT_AVAILABLE = False
    print(f"⚠️  LLM client manager not available: {e}")
    print("   Column extraction will use rule-based analysis only")

# =============================================================================
# CONFIGURATION SETTINGS - Modify these as needed
# =============================================================================

# Database Connection Settings
DB_HOST = "--application.server.host=ws://olympus-high-volume-api-icg-isg-olympus-high-volume-api-167969.apps.namicgrut37p.ecs.dyn.nsroot.net"
DB_ENVIRONMENT = "prod"

# System Paths (modify based on your environment)
WINDOWS_JAVA_PATH = "C:/work/training_data/jdk-17.0.7/jdk-17.0.7/bin/java.exe"
WINDOWS_JAR_PATH = "C:/work/training_data/mktdata-report-hvapi-commandline-client-1.0.30.jar"
LINUX_JAVA_PATH = "/opt/jdk/17.0_9l64/bin/java"
LINUX_JAR_PATH = "/home/bj33244/mktdata-report-hvapi-commandline-client-1.0.27-SNAPSHOT.jar"

# Analysis Settings
LOG_ANALYSIS_MONTHS = 1  # Number of months to analyze
LOG_ANALYSIS_MIN_USAGE = 5  # Minimum usage count to consider a column frequent
SQL_CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for SQL validation
MAX_COLUMNS_PER_TABLE = 20  # Maximum columns to include in output

# LLM/GPT-4 Integration Settings
ENABLE_LLM_ANALYSIS = True  # Enable LLM-based column extraction for complex queries
MAX_LLM_CALLS_PER_TABLE = 1000  # Maximum LLM calls per table analysis
LLM_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for LLM extraction results
USE_HYBRID_APPROACH = True  # Use both rule-based and LLM analysis

# Output Settings
BASE_OUTPUT_DIR = "./output"
CONFIG_OUTPUT_DIR = "./configs"

# Date Settings
DEFAULT_DAYS_BACK = 90  # Default number of days for analysis
DEFAULT_START_DATE = "20250311"  # Format: YYYYMMDD

# =============================================================================
# CORE CLASSES
# =============================================================================

@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    host: str
    user_id: str
    password: str
    environment: str = "prod"

@dataclass
class SystemConfig:
    """System configuration"""
    java_path: str
    jar_path: str
    output_dir: str
    config_dir: str

class DatabaseClient:
    """Handles database operations"""
    
    def __init__(self, db_config: DatabaseConfig, system_config: SystemConfig):
        self.db_config = db_config
        self.system_config = system_config
        self.logger = logging.getLogger(__name__)
    
    def execute_query(self, query: str, output_file: str, format_type: str = "TXT") -> bool:
        """Execute database query and save results"""
        try:
            command = [
                self.system_config.java_path,
                "--add-opens=java.base/java.nio=ALL-UNNAMED",
                "-jar", self.system_config.jar_path,
                self.db_config.host,
                f"--query={query}",
                f"--user={self.db_config.user_id}",
                f"--pass={self.db_config.password}",
                f"--env={self.db_config.environment}",
                f"--format={format_type}",
                f"--destination={output_file}"
            ]
            
            self.logger.info(f"Executing query: {query}")
            
            # Validate paths
            if not os.path.exists(self.system_config.java_path):
                self.logger.error(f"Java executable not found: {self.system_config.java_path}")
                return False
            
            if not os.path.exists(self.system_config.jar_path):
                self.logger.error(f"JAR file not found: {self.system_config.jar_path}")
                return False
            
            result = subprocess.run(command, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                self.logger.error(f"Query failed: {result.stderr}")
                return False
            
            return os.path.exists(output_file) and os.path.getsize(output_file) > 0
            
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            return False
    
    def get_table_schema(self, table_name: str) -> Dict[str, str]:
        """Get table schema information"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.schema', delete=False) as temp_file:
            schema_file = temp_file.name
        
        try:
            query = f"describe {table_name}"
            if self.execute_query(query, schema_file):
                return self._parse_schema_file(schema_file)
            return {}
        finally:
            if os.path.exists(schema_file):
                os.unlink(schema_file)
    
    def _parse_schema_file(self, schema_file: str) -> Dict[str, str]:
        """Parse schema file and return column information"""
        schema_dict = {}
        try:
            with open(schema_file, 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                if i == 0:  # Skip header
                    continue
                
                parts = line.strip().split(',', 1)
                if len(parts) >= 2:
                    column_name = parts[0].strip().lower()
                    column_type = parts[1].strip()
                    mapped_type = self._map_column_type(column_type)
                    schema_dict[column_name] = f"{column_type}|{mapped_type}"
            
            return schema_dict
        except Exception as e:
            self.logger.error(f"Error parsing schema file: {e}")
            return {}
    
    def _map_column_type(self, column_type: str) -> str:
        """Map database column types to simplified types"""
        column_type_lower = column_type.lower()
        
        if any(t in column_type_lower for t in ['varchar', 'string', 'char']):
            return "string"
        elif any(t in column_type_lower for t in ['bigint', 'int']):
            return "integer"
        elif any(t in column_type_lower for t in ['decimal', 'double']):
            return "float"
        elif 'timestamp' in column_type_lower:
            return "timestamp"
        elif column_type_lower.startswith('map<'):
            return "nestedtype"
        else:
            return "string"

class QueryLogAnalyzer:
    """Analyzes query logs to discover frequent columns using hybrid rule-based + LLM approach"""
    
    def __init__(self, db_client: DatabaseClient):
        self.db_client = db_client
        self.logger = logging.getLogger(__name__)
        
        # SQL validation patterns
        self.sql_keywords = {
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER',
            'GROUP', 'ORDER', 'BY', 'HAVING', 'LIMIT', 'INSERT', 'UPDATE', 'DELETE'
        }
        
        # Noise detection patterns
        self.noise_patterns = [
            r'The batch size', r'cumulative row', r'Finalizing Resources',
            r'Connection Details', r'Count of total', r'^\s*\d+\s*$', 
            r'^\s*null\s*$', r'^\s*$'
        ]
        
        # LLM call tracking
        self.llm_calls_made = 0
        self.max_llm_calls = MAX_LLM_CALLS_PER_TABLE
    
    def analyze_table_usage(self, table_name: str, months_back: int = LOG_ANALYSIS_MONTHS) -> Dict[str, Any]:
        """Analyze query logs for table usage patterns using hybrid approach"""
        try:
            self.logger.info(f"Analyzing query logs for table: {table_name}")
            self.llm_calls_made = 0  # Reset LLM call counter
            
            # Extract query logs
            log_data = self._extract_query_logs(table_name, months_back)
            
            if not log_data:
                self.logger.warning(f"No query logs found for table: {table_name}")
                return self._empty_result(table_name)
            
            # Filter and clean queries
            clean_queries = self._filter_and_clean_queries(log_data)
            
            # Enhanced column usage parsing with hybrid approach
            if ENABLE_LLM_ANALYSIS and LLM_CLIENT_AVAILABLE and USE_HYBRID_APPROACH:
                self.logger.info(f"Using hybrid rule-based + LLM analysis for {table_name}")
                column_usage = asyncio.run(self._parse_column_usage_hybrid(clean_queries, table_name))
            else:
                self.logger.info(f"Using rule-based analysis only for {table_name}")
                column_usage = self._parse_column_usage_rules_only(clean_queries, table_name)
            
            # Generate statistics
            stats = self._generate_usage_stats(column_usage, clean_queries)
            stats['llm_calls_made'] = self.llm_calls_made
            stats['analysis_method'] = 'hybrid' if (ENABLE_LLM_ANALYSIS and LLM_CLIENT_AVAILABLE) else 'rule_based'
            
            return {
                'table_name': table_name,
                'total_queries': len(log_data),
                'valid_queries': len(clean_queries),
                'column_usage_frequency': column_usage,
                'top_columns': self._get_top_columns(column_usage, MAX_COLUMNS_PER_TABLE),
                'usage_statistics': stats
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed for {table_name}: {e}")
            return self._empty_result(table_name)
    
    def _extract_query_logs(self, table_name: str, months_back: int) -> List[Dict[str, Any]]:
        """Extract query logs from audit table"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Use 7 days for reliability
        
        query = f"""
        SELECT 
            dwh_business_date,
            client_id,
            SUBSTR(query_text, 1, 300) as query_text
        FROM gfolydal_managed.jdbc_centralized_audit_log
        WHERE 
            dwh_business_date >= {start_date.strftime('%Y%m%d')}
            AND dwh_business_date <= {end_date.strftime('%Y%m%d')}
            AND query_text LIKE '%SELECT%'
        LIMIT 200
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.logs', delete=False) as temp_file:
            temp_output = temp_file.name
        
        try:
            if self.db_client.execute_query(query, temp_output):
                return self._parse_log_file(temp_output)
            return []
        finally:
            if os.path.exists(temp_output):
                os.unlink(temp_output)
    
    def _parse_log_file(self, log_file: str) -> List[Dict[str, Any]]:
        """Parse log file and extract structured data"""
        log_data = []
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                if i == 0:  # Skip header
                    continue
                
                line = line.strip()
                if not line:
                    continue
                
                # Split CSV line
                parts = self._split_csv_line(line)
                
                if len(parts) >= 3:
                    log_entry = {
                        'dwh_business_date': parts[0].strip(),
                        'client_id': parts[1].strip(),
                        'query_text': parts[2].strip()
                    }
                    
                    if log_entry['query_text']:
                        log_data.append(log_entry)
            
            return log_data
            
        except Exception as e:
            self.logger.error(f"Error parsing log file: {e}")
            return []
    
    def _split_csv_line(self, line: str) -> List[str]:
        """Split CSV line handling quoted fields"""
        parts = []
        current_part = ""
        in_quotes = False
        
        for char in line:
            if char == '"':
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                parts.append(current_part.strip())
                current_part = ""
            else:
                current_part += char
        
        if current_part:
            parts.append(current_part.strip())
        
        return parts
    
    def _filter_and_clean_queries(self, log_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and clean query log data"""
        clean_queries = []
        
        for entry in log_data:
            query_text = entry.get('query_text', '').strip()
            
            if self._is_valid_sql(query_text):
                entry_copy = entry.copy()
                entry_copy['query_text'] = self._clean_query_text(query_text)
                clean_queries.append(entry_copy)
        
        return clean_queries
    
    def _is_valid_sql(self, query_text: str) -> bool:
        """Check if query text is valid SQL"""
        if not query_text or len(query_text.strip()) < 10:
            return False
        
        query_upper = query_text.upper().strip()
        
        # Check for noise patterns
        import re
        for pattern in self.noise_patterns:
            if re.search(pattern, query_text, re.IGNORECASE):
                return False
        
        # Must have SELECT and FROM
        has_select = 'SELECT' in query_upper
        has_from = 'FROM' in query_upper
        
        return has_select and has_from
    
    def _clean_query_text(self, query_text: str) -> str:
        """Clean and normalize query text"""
        import re
        query_text = re.sub(r'\s+', ' ', query_text)
        query_text = re.sub(r'\|+', ' ', query_text)
        return query_text.strip()
    
    def _parse_column_usage_rules_only(self, queries: List[Dict[str, Any]], table_name: str) -> Dict[str, int]:
        """Parse column usage using rule-based approach only"""
        column_usage = Counter()
        
        for entry in queries:
            query_text = entry['query_text']
            columns = self._extract_columns_regex(query_text, table_name)
            
            for column in columns:
                if column and len(column) > 1:
                    column_usage[column] += 1
        
        return dict(column_usage)
    
    async def _parse_column_usage_hybrid(self, queries: List[Dict[str, Any]], table_name: str) -> Dict[str, int]:
        """Parse column usage using hybrid rule-based + LLM approach"""
        column_usage = Counter()
        
        # Track queries by complexity for intelligent LLM usage
        simple_queries = []
        complex_queries = []
        
        for entry in queries:
            query_text = entry['query_text']
            if self._is_complex_query(query_text):
                complex_queries.append(entry)
            else:
                simple_queries.append(entry)
        
        self.logger.info(f"Query complexity analysis: {len(simple_queries)} simple, {len(complex_queries)} complex")
        
        # Process simple queries with rule-based approach
        for entry in simple_queries:
            query_text = entry['query_text']
            columns = self._extract_columns_regex(query_text, table_name)
            
            for column in columns:
                if column and len(column) > 1:
                    column_usage[column] += 1
        
        # Process complex queries with LLM (with limits)
        if complex_queries and self.llm_calls_made < self.max_llm_calls:
            self.logger.info(f"Processing {len(complex_queries)} complex queries with LLM analysis...")
            
            for entry in complex_queries[:self.max_llm_calls]:  # Limit LLM calls
                if self.llm_calls_made >= self.max_llm_calls:
                    self.logger.warning(f"Reached LLM call limit ({self.max_llm_calls}), using rule-based for remaining queries")
                    break
                
                query_text = entry['query_text']
                
                # Try LLM extraction first
                llm_columns = await self._extract_columns_llm(query_text, table_name)
                
                if llm_columns:
                    # LLM extraction successful
                    for column in llm_columns:
                        if column and len(column) > 1:
                            column_usage[column] += 1
                else:
                    # Fallback to rule-based for this query
                    rule_columns = self._extract_columns_regex(query_text, table_name)
                    for column in rule_columns:
                        if column and len(column) > 1:
                            column_usage[column] += 1
        
        # Process remaining complex queries with rule-based approach
        remaining_complex = complex_queries[self.llm_calls_made:]
        if remaining_complex:
            self.logger.info(f"Processing {len(remaining_complex)} remaining complex queries with rule-based approach")
            for entry in remaining_complex:
                query_text = entry['query_text']
                columns = self._extract_columns_regex(query_text, table_name)
                
                for column in columns:
                    if column and len(column) > 1:
                        column_usage[column] += 1
        
        return dict(column_usage)
    
    def _is_complex_query(self, query_text: str) -> bool:
        """Determine if a query is complex enough to warrant LLM analysis"""
        query_upper = query_text.upper()
        
        # Consider complex if it has:
        complexity_indicators = [
            'SUBQUERY' in query_upper or '(' in query_text,  # Subqueries
            'CASE WHEN' in query_upper,  # Case statements
            'UNION' in query_upper,  # Unions
            query_text.count('JOIN') > 1,  # Multiple joins
            query_text.count(',') > 10,  # Many columns
            len(query_text) > 200,  # Long queries
            'WINDOW' in query_upper or 'OVER(' in query_upper,  # Window functions
        ]
        
        return any(complexity_indicators)
    
    async def _extract_columns_llm(self, query_text: str, table_name: str) -> Set[str]:
        """Extract column names using LLM for complex cases"""
        if not LLM_CLIENT_AVAILABLE:
            self.logger.warning("LLM client not available, falling back to rule-based extraction")
            return self._extract_columns_regex(query_text, table_name)
        
        try:
            self.llm_calls_made += 1
            
            # Create prompt for GPT-4
            extraction_prompt = f"""You are a SQL expert. Extract all column names referenced in this SQL query for table '{table_name}'.

**SQL Query:**
{query_text}

**Target Table:** {table_name}

**Instructions:**
1. Extract ONLY column names that belong to the target table
2. Ignore functions (COUNT, SUM, etc.), literals, and aliases
3. Return column names in lowercase
4. If the query is invalid/incomplete, return empty list

**Required Output Format (JSON):**
{{
    "is_valid_sql": true/false,
    "extracted_columns": ["column1", "column2", "column3"],
    "confidence": 0.0-1.0
}}

**Examples:**
- "SELECT t.trade_id, t.amount FROM trades t" → ["trade_id", "amount"]
- "WHERE settlement_date > '2024-01-01'" → ["settlement_date"]
- "GROUP BY currency, status" → ["currency", "status"]

**Respond with only the JSON:**"""

            # Call LLM
            response = await client_manager.ask_gpt(extraction_prompt)
            result = json.loads(response)
            
            # Validate response
            if result.get('is_valid_sql', False) and result.get('confidence', 0) >= LLM_CONFIDENCE_THRESHOLD:
                extracted_columns = set(result.get('extracted_columns', []))
                self.logger.debug(f"LLM extracted {len(extracted_columns)} columns with confidence {result.get('confidence', 0):.2f}")
                return extracted_columns
            else:
                self.logger.debug(f"LLM extraction below confidence threshold or invalid SQL")
                return set()
                
        except Exception as e:
            self.logger.error(f"LLM extraction failed: {e}")
            return set()
    
    def _extract_columns_regex(self, query_text: str, table_name: str) -> Set[str]:
        """Extract column names using rule-based regex patterns (enhanced version)"""
        import re
        columns = set()
        
        try:
            # Extract table.column references
            table_simple_name = table_name.split('.')[-1].upper()
            table_column_pattern = rf'\b(?:{re.escape(table_simple_name)}|[A-Z_]+)\s*\.\s*([a-zA-Z_][a-zA-Z0-9_]*)\b'
            matches = re.findall(table_column_pattern, query_text, re.IGNORECASE)
            columns.update([col.lower() for col in matches])
            
            # Extract from SELECT clause
            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', query_text.upper(), re.DOTALL)
            if select_match:
                select_clause = select_match.group(1)
                if '*' not in select_clause:
                    for item in select_clause.split(','):
                        item = item.strip()
                        item = re.sub(r'\s+AS\s+\w+', '', item, flags=re.IGNORECASE)
                        
                        if '.' in item:
                            col_name = item.split('.')[-1].strip()
                        else:
                            col_name = item.strip()
                        
                        if (col_name and len(col_name) > 1 and 
                            not any(func in col_name.upper() for func in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'CASE']) and
                            not col_name.isdigit() and not col_name.startswith("'")):
                            columns.add(col_name.lower())
            
            # Extract from WHERE clause
            where_matches = re.findall(r'WHERE\s+.*?([a-zA-Z_][a-zA-Z0-9_]*)\s*[=<>!]', query_text, re.IGNORECASE)
            columns.update([col.lower() for col in where_matches])
            
            # Extract from GROUP BY and ORDER BY
            for clause in ['GROUP BY', 'ORDER BY']:
                pattern = rf'{clause}\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*)'
                matches = re.findall(pattern, query_text, re.IGNORECASE)
                for match in matches:
                    for col in match.split(','):
                        col = col.strip().split()[0]  # Remove ASC/DESC
                        if '.' in col:
                            col = col.split('.')[-1]
                        columns.add(col.lower())
            
            # Extract common trading column patterns
            trading_patterns = [
                r'\b(TRADE_\w+)\b', r'\b(SETTLEMENT_\w+)\b', r'\b(\w+_DATE)\b',
                r'\b(\w+_ID)\b', r'\b(\w+_SK)\b', r'\b(QUANTITY\w*)\b',
                r'\b(PRICE_\w+)\b', r'\b(AMOUNT_\w+)\b', r'\b(CURRENCY\w*)\b'
            ]
            
            for pattern in trading_patterns:
                matches = re.findall(pattern, query_text.upper())
                for match in matches:
                    columns.add(match.lower())
            
        except Exception as e:
            self.logger.debug(f"Regex extraction error: {e}")
        
        return columns
    
    def _generate_usage_stats(self, column_usage: Dict[str, int], queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate usage statistics"""
        total_columns = len(column_usage)
        total_usage = sum(column_usage.values())
        
        return {
            'total_unique_columns': total_columns,
            'total_column_references': total_usage,
            'average_usage_per_column': total_usage / total_columns if total_columns > 0 else 0,
            'most_used_column': max(column_usage.items(), key=lambda x: x[1]) if column_usage else None,
            'unique_clients': len(set(entry.get('client_id', '') for entry in queries)),
            'llm_calls_made': getattr(self, 'llm_calls_made', 0),
            'analysis_method': 'hybrid' if (ENABLE_LLM_ANALYSIS and LLM_CLIENT_AVAILABLE) else 'rule_based'
        }
    
    def _get_top_columns(self, column_usage: Dict[str, int], limit: int) -> List[tuple]:
        """Get top N most frequently used columns"""
        return sorted(column_usage.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def _empty_result(self, table_name: str) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            'table_name': table_name,
            'total_queries': 0,
            'valid_queries': 0,
            'column_usage_frequency': {},
            'top_columns': [],
            'usage_statistics': {
                'total_unique_columns': 0,
                'total_column_references': 0,
                'average_usage_per_column': 0,
                'most_used_column': None,
                'unique_clients': 0,
                'llm_calls_made': 0,
                'analysis_method': 'no_data'
            }
        }

class ConfigGenerator:
    """Generates configuration files"""
    
    def __init__(self, system_config: SystemConfig):
        self.system_config = system_config
        self.logger = logging.getLogger(__name__)
    
    def generate_config_file(self, table_name: str, analysis_result: Dict[str, Any], 
                           user_id: str, password: str) -> str:
        """Generate configuration file for a table"""
        
        # Create config filename
        table_simple_name = table_name.replace('.', '_')
        config_filename = f"tbc_{table_simple_name}.conf"
        config_path = os.path.join(self.system_config.config_dir, config_filename)
        
        # Ensure config directory exists
        os.makedirs(self.system_config.config_dir, exist_ok=True)
        
        # Get frequent columns
        frequent_columns = [col for col, count in analysis_result['top_columns'] 
                          if count >= LOG_ANALYSIS_MIN_USAGE]
        
        # Create configuration content
        config = configparser.ConfigParser()
        config.optionxform = str  # Preserve case
        
        section_name = "transcation"
        config.add_section(section_name)
        
        # Basic settings
        config.set(section_name, 'BASEDIRECTORY', BASE_OUTPUT_DIR)
        config.set(section_name, 'USERID', user_id)
        config.set(section_name, 'PASSWORD', password)
        config.set(section_name, 'OUTPUT_DIRECTORY', f"{BASE_OUTPUT_DIR}/transcation")
        config.set(section_name, 'START_DATE', DEFAULT_START_DATE)
        
        # Analysis settings
        config.set(section_name, 'ENABLE_LOG_ANALYSIS', 'true')
        config.set(section_name, 'LOG_ANALYSIS_MONTHS', str(LOG_ANALYSIS_MONTHS))
        config.set(section_name, 'LOG_ANALYSIS_MIN_USAGE', str(LOG_ANALYSIS_MIN_USAGE))
        config.set(section_name, 'MAX_LLM_CALLS_PER_TABLE', '1000')
        
        # Table configuration
        config.set(section_name, 'TABLES', table_name)
        
        # Column configuration
        columns_key = f"{table_name}_Columns"
        if frequent_columns:
            config.set(section_name, columns_key, ','.join(frequent_columns))
        else:
            # Fallback to common columns based on table type
            default_columns = self._get_default_columns_for_table(table_name)
            config.set(section_name, columns_key, ','.join(default_columns))
        
        # Date criteria
        where_col_key = f"{table_name}_WhereCol_daycriteria"
        config.set(section_name, where_col_key, 'dwh_business_date')
        
        # Enhanced metadata with LLM usage info
        config.set(section_name, f"{table_name}_Enhanced_Analysis_Date", datetime.now().strftime("%Y%m%d_%H%M%S"))
        config.set(section_name, f"{table_name}_TotalQueries", str(analysis_result['total_queries']))
        config.set(section_name, f"{table_name}_ValidQueries", str(analysis_result['valid_queries']))
        config.set(section_name, f"{table_name}_ColumnsFound", str(len(analysis_result['column_usage_frequency'])))
        config.set(section_name, f"{table_name}_AnalysisMethod", analysis_result['usage_statistics'].get('analysis_method', 'rule_based'))
        
        if 'llm_calls_made' in analysis_result['usage_statistics']:
            config.set(section_name, f"{table_name}_LLMCallsMade", str(analysis_result['usage_statistics']['llm_calls_made']))
        
        # LLM configuration info
        config.set(section_name, f"{table_name}_LLMEnabled", str(ENABLE_LLM_ANALYSIS and LLM_CLIENT_AVAILABLE))
        config.set(section_name, f"{table_name}_HybridAnalysis", str(USE_HYBRID_APPROACH))
        
        # Save configuration file
        try:
            with open(config_path, 'w') as configfile:
                config.write(configfile)
            
            self.logger.info(f"Configuration file generated: {config_path}")
            return config_path
            
        except Exception as e:
            self.logger.error(f"Error saving configuration file: {e}")
            raise
    
    def _get_default_columns_for_table(self, table_name: str) -> List[str]:
        """Get default columns based on table name patterns"""
        table_lower = table_name.lower()
        
        if 'trade' in table_lower:
            return [
                'trade_id', 'trade_date', 'settlement_date', 'quantity', 'price',
                'amount', 'currency', 'counterparty', 'trader_id', 'status'
            ]
        elif 'market' in table_lower or 'quote' in table_lower:
            return [
                'quote_timestamp', 'ticker_symbol', 'bid_price', 'ask_price',
                'quote_currency', 'volume', 'exchange_code', 'dwh_business_date'
            ]
        elif 'position' in table_lower:
            return [
                'position_id', 'account_id', 'instrument_id', 'quantity',
                'market_value', 'currency', 'as_of_date', 'portfolio_id'
            ]
        else:
            return [
                'id', 'name', 'type', 'status', 'date', 'amount',
                'currency', 'description', 'dwh_business_date'
            ]

# =============================================================================
# MAIN PROCESSING CLASS
# =============================================================================

class FrequentColumnProcessor:
    """Main processor for discovering frequent columns"""
    
    def __init__(self, user_id: str, password: str):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize configurations
        db_config = DatabaseConfig(
            host=DB_HOST,
            user_id=user_id,
            password=password,
            environment=DB_ENVIRONMENT
        )
        
        system_config = SystemConfig(
            java_path=WINDOWS_JAVA_PATH if os.name == 'nt' else LINUX_JAVA_PATH,
            jar_path=WINDOWS_JAR_PATH if os.name == 'nt' else LINUX_JAR_PATH,
            output_dir=BASE_OUTPUT_DIR,
            config_dir=CONFIG_OUTPUT_DIR
        )
        
        # Initialize components
        self.db_client = DatabaseClient(db_config, system_config)
        self.query_analyzer = QueryLogAnalyzer(self.db_client)
        self.config_generator = ConfigGenerator(system_config)
        self.user_id = user_id
        self.password = password
        
        # Create output directories
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        os.makedirs(CONFIG_OUTPUT_DIR, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('frequent_columns.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def process_table(self, table_name: str) -> str:
        """Process a single table and generate config file"""
        try:
            self.logger.info(f"Processing table: {table_name}")
            
            # Test database connection
            self.logger.info("Testing database connection...")
            test_query = "SELECT 1 as test"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.test', delete=False) as temp_file:
                temp_output = temp_file.name
            
            try:
                if not self.db_client.execute_query(test_query, temp_output):
                    raise Exception("Database connection test failed")
                self.logger.info("✅ Database connection successful!")
            finally:
                if os.path.exists(temp_output):
                    os.unlink(temp_output)
            
            # Get table schema
            self.logger.info(f"Loading schema for table: {table_name}")
            schema = self.db_client.get_table_schema(table_name)
            
            if not schema:
                self.logger.warning(f"Could not load schema for table: {table_name}")
            else:
                self.logger.info(f"Loaded schema with {len(schema)} columns")
            
            # Analyze query logs
            self.logger.info(f"Analyzing query logs for table: {table_name}")
            analysis_result = self.query_analyzer.analyze_table_usage(table_name)
            
            # Generate configuration file
            self.logger.info(f"Generating configuration file for table: {table_name}")
            config_path = self.config_generator.generate_config_file(
                table_name, analysis_result, self.user_id, self.password
            )
            
            # Log results with LLM usage info
            self.logger.info(f"✅ Processing completed for table: {table_name}")
            self.logger.info(f"   - Total queries analyzed: {analysis_result['total_queries']}")
            self.logger.info(f"   - Valid SQL queries: {analysis_result['valid_queries']}")
            self.logger.info(f"   - Unique columns found: {len(analysis_result['column_usage_frequency'])}")
            self.logger.info(f"   - Top frequent columns: {len(analysis_result['top_columns'])}")
            self.logger.info(f"   - Analysis method: {analysis_result['usage_statistics'].get('analysis_method', 'rule_based')}")
            
            if 'llm_calls_made' in analysis_result['usage_statistics']:
                self.logger.info(f"   - LLM calls made: {analysis_result['usage_statistics']['llm_calls_made']}")
            
            self.logger.info(f"   - Configuration file: {config_path}")
            
            return config_path
            
        except Exception as e:
            self.logger.error(f"Error processing table {table_name}: {e}")
            raise
    
    def process_multiple_tables(self, table_names: List[str]) -> List[str]:
        """Process multiple tables and generate config files"""
        config_files = []
        
        for table_name in table_names:
            try:
                config_path = self.process_table(table_name)
                config_files.append(config_path)
            except Exception as e:
                self.logger.error(f"Failed to process table {table_name}: {e}")
                continue
        
        return config_files

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate frequent columns configuration with hybrid analysis')
    parser.add_argument('--table', '-t', type=str, help='Table name to analyze')
    parser.add_argument('--tables', '-T', nargs='+', help='Multiple table names to analyze')
    parser.add_argument('--user', '-u', type=str, required=True, help='Database user ID')
    parser.add_argument('--password', '-p', type=str, required=True, help='Database password')
    parser.add_argument('--output-dir', '-o', type=str, default=CONFIG_OUTPUT_DIR, 
                       help='Output directory for config files')
    parser.add_argument('--disable-llm', action='store_true', help='Disable LLM analysis (use rule-based only)')
    
    args = parser.parse_args()
    
    # Update LLM settings based on arguments
    global ENABLE_LLM_ANALYSIS
    if args.disable_llm:
        ENABLE_LLM_ANALYSIS = False
        print("LLM analysis disabled by user request")
    
    # Update output directory if specified
    global CONFIG_OUTPUT_DIR
    CONFIG_OUTPUT_DIR = args.output_dir
    
    # Determine tables to process
    tables_to_process = []
    if args.table:
        tables_to_process.append(args.table)
    if args.tables:
        tables_to_process.extend(args.tables)
    
    if not tables_to_process:
        print("Error: No tables specified. Use --table or --tables option.")
        return 1
    
    # Display analysis method
    analysis_method = "Hybrid (Rule-based + LLM)" if (ENABLE_LLM_ANALYSIS and LLM_CLIENT_AVAILABLE) else "Rule-based only"
    print(f"Analysis method: {analysis_method}")
    if ENABLE_LLM_ANALYSIS and LLM_CLIENT_AVAILABLE:
        print(f"Maximum LLM calls per table: {MAX_LLM_CALLS_PER_TABLE}")
    
    try:
        # Initialize processor
        processor = FrequentColumnProcessor(args.user, args.password)
        
        # Process tables
        print(f"Processing {len(tables_to_process)} table(s)...")
        config_files = processor.process_multiple_tables(tables_to_process)
        
        # Print results
        print(f"\n✅ Successfully generated {len(config_files)} configuration file(s):")
        for config_file in config_files:
            print(f"   - {config_file}")
        
        return 0
        
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())