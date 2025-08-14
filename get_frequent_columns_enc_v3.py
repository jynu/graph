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
import io

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

# ENCRYPTED_PASS_FILE="password.enc"
# KEY_VALUE_FILE="keyfile.key"

os.environ['ENCRYPTED_PASS_FILE']="password.enc"
os.environ['KEY_VALUE_FILE']="keyfile.key"

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

class CredentialManager:
    """Handles secure credential management across platforms"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def _get_openssl_path(self) -> str:
        """Get OpenSSL executable path based on platform"""
        if os.name == 'nt':  # Windows
            # Common Windows OpenSSL locations
            possible_paths = [
                "C:/Program Files/OpenSSL-Win64/bin/openssl.exe",
                "C:/Program Files (x86)/OpenSSL-Win32/bin/openssl.exe",
                "C:/OpenSSL-Win64/bin/openssl.exe",
                "C:/OpenSSL-Win32/bin/openssl.exe",
                "openssl.exe",  # If in PATH
                "openssl"       # Fallback
            ]
        else:  # Linux/Unix
            possible_paths = [
                "/usr/local/bin/openssl",
                "/usr/bin/openssl",
                "/bin/openssl",
                "openssl"  # If in PATH
            ]
        
        # Check which OpenSSL is available
        for path in possible_paths:
            try:
                result = subprocess.run([path, "version"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    self.logger.debug(f"Found OpenSSL at: {path}")
                    return path
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                continue
        
        raise FileNotFoundError("OpenSSL executable not found on system")
    
    def decrypt_password(self) -> str:
        """Decrypt password using OpenSSL (cross-platform)"""
        encrypted_file = os.environ.get('ENCRYPTED_PASS_FILE')
        keyvalue_file = os.environ.get('KEY_VALUE_FILE')
        
        if not encrypted_file or not keyvalue_file:
            raise ValueError("Password decryption environment variables not set")
        
        if not os.path.exists(encrypted_file):
            raise FileNotFoundError(f"Encrypted password file not found: {encrypted_file}")
        
        if not os.path.exists(keyvalue_file):
            raise FileNotFoundError(f"Key value file not found: {keyvalue_file}")
        
        try:
            openssl_path = self._get_openssl_path()
            
            # OpenSSL command for decryption
            cmd = [
                openssl_path, "enc", "-aes-128-cbc", "-pbkdf2", 
                "-a", "-d", "-in", encrypted_file, "-pass", f"file:{keyvalue_file}"
            ]
            
            self.logger.debug(f"Executing decryption command on {os.name}")
            
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = proc.communicate()
            
            if proc.returncode != 0:
                error_msg = stderr.decode('utf-8').strip()
                raise RuntimeError(f"Password decryption failed: {error_msg}")
            
            decrypted_password = stdout.decode('utf-8').strip()
            
            if not decrypted_password:
                raise RuntimeError("Decrypted password is empty")
            
            self.logger.info(f"Password successfully decrypted on {os.name}")
            return decrypted_password
            
        except Exception as e:
            self.logger.error(f"Error decrypting password: {e}")
            raise
    
    def get_secure_password(self, provided_password: str = None) -> str:
        """Get password securely based on availability (cross-platform)"""
        
        # Priority 1: If password provided directly, use it (for backward compatibility)
        if provided_password:
            self.logger.info("Using provided password parameter")
            return provided_password
        
        # Priority 2: Try to decrypt from encrypted file (both Windows and Linux)
        try:
            decrypted_password = self.decrypt_password()
            if decrypted_password:
                self.logger.info(f"Using decrypted password from encrypted file ({os.name})")
                return decrypted_password
        except Exception as e:
            self.logger.warning(f"Could not decrypt password on {os.name}: {e}")
        
        # Priority 3: Try Windows Credential Manager (Windows only)
        if os.name == 'nt':
            try:
                windows_password = self._get_windows_credential()
                if windows_password:
                    self.logger.info("Using password from Windows Credential Manager")
                    return windows_password
            except Exception as e:
                self.logger.warning(f"Could not retrieve from Windows Credential Manager: {e}")
        
        # Priority 4: Fallback to interactive prompt
        import getpass
        self.logger.info("Prompting for password interactively")
        return getpass.getpass("Enter database password: ")
    
    def _get_windows_credential(self) -> str:
        """Get password from Windows Credential Manager (optional enhancement)"""
        try:
            import keyring
            service_name = "training_data_processor"
            username = os.environ.get('USERNAME', 'default_user')
            password = keyring.get_password(service_name, username)
            return password if password else ""
        except ImportError:
            self.logger.debug("keyring module not available for Windows Credential Manager")
            return ""
        except Exception as e:
            self.logger.debug(f"Windows Credential Manager access failed: {e}")
            return ""

def validate_environment():
    """Validate environment setup for encrypted password (cross-platform)"""
    encrypted_file = os.environ.get('ENCRYPTED_PASS_FILE')
    keyvalue_file = os.environ.get('KEY_VALUE_FILE')
    
    print(f"Platform: {os.name} ({'Windows' if os.name == 'nt' else 'Linux/Unix'})")
    
    if encrypted_file and keyvalue_file:
        print("Encrypted password configuration detected:")
        print(f"  - Encrypted file: {encrypted_file}")
        print(f"  - Key file: {keyvalue_file}")
        
        # Check file existence
        if not os.path.exists(encrypted_file):
            print(f"  ⚠️  Warning: Encrypted password file not found: {encrypted_file}")
        else:
            print(f"  ✓ Encrypted password file found")
        
        if not os.path.exists(keyvalue_file):
            print(f"  ⚠️  Warning: Key value file not found: {keyvalue_file}")
        else:
            print(f"  ✓ Key file found")
        
        # Check OpenSSL availability
        try:
            credential_manager = CredentialManager()
            openssl_path = credential_manager._get_openssl_path()
            print(f"  ✓ OpenSSL found at: {openssl_path}")
        except FileNotFoundError:
            print(f"  ⚠️  Warning: OpenSSL not found on system")
    else:
        print("No encrypted password configuration detected")
        if not encrypted_file:
            print("  - ENCRYPTED_PASS_FILE environment variable not set")
        if not keyvalue_file:
            print("  - KEY_VALUE_FILE environment variable not set")
        print("  - Will fall back to interactive password prompt")

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
        start_date = end_date - timedelta(days=60)  # Use 7 days for reliability
        
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
            AND user_action = 'Original Query'
            AND client_id NOT LIKE '%fid%'
        LIMIT 30
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
            # Try different encodings to handle binary data
            content = None
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'utf-8-sig']:
                try:
                    with open(log_file, 'r', encoding=encoding) as f:
                        content = f.read()
                    self.logger.debug(f"File read successfully with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    self.logger.debug(f"Failed to read with {encoding}, trying next...")
                    continue
            
            if not content:
                self.logger.error("Could not decode file with any encoding")
                return []
            
            # Clean any problematic characters
            import re
            content = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', ' ', content)
            
            lines = content.strip().split('\n')
            
            processed_count = 0
            for i, line in enumerate(lines):
                if i == 0:  # Skip header
                    continue
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Split CSV line with better error handling
                    parts = self._split_csv_line(line)
                    
                    # Ensure we have at least 3 parts
                    if len(parts) >= 3:
                        # Clean the parts to remove any problematic characters
                        log_entry = {
                            'dwh_business_date': self._clean_text(parts[0]),
                            'client_id': self._clean_text(parts[1]),
                            'query_text': self._clean_text(parts[2])
                        }
                        
                        # Only include if query_text is not empty and looks valid
                        if log_entry['query_text'] and len(log_entry['query_text']) > 5:
                            log_data.append(log_entry)
                            processed_count += 1
                            
                            # Debug first few entries
                            if processed_count <= 3:
                                self.logger.debug(f"Parsed entry {processed_count}: {log_entry['query_text'][:50]}...")
                    
                except Exception as parse_error:
                    self.logger.debug(f"Failed to parse line {i}: {line[:50]}... - Error: {parse_error}")
                    continue
            
            self.logger.info(f"Successfully parsed {len(log_data)} valid log entries from {len(lines)} total lines")
            
            return log_data
            
        except Exception as e:
            self.logger.error(f"Error parsing log file: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean text to remove problematic characters"""
        if not text:
            return ""
        
        # Remove non-printable characters except newlines and tabs
        import re
        cleaned = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', ' ', str(text))
        
        # Replace multiple spaces with single space
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()
    
    def _split_csv_line(self, line: str) -> List[str]:
        """Split CSV line handling quoted fields and problematic characters"""
        parts = []
        current_part = ""
        in_quotes = False
        
        try:
            for char in line:
                if char == '"':
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
                    parts.append(current_part.strip())
                    current_part = ""
                else:
                    # Only add printable characters
                    if ord(char) >= 32 or char in ['\t', '\n']:
                        current_part += char
            
            if current_part:
                parts.append(current_part.strip())
            
            return parts
            
        except Exception as e:
            self.logger.debug(f"Error splitting CSV line: {e}")
            # Fallback: simple split by comma
            return line.split(',')
    
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

**IMPORTANT: Respond with ONLY the JSON object, no additional text or formatting:**"""

            # Call LLM
            print(extraction_prompt)
            response = await client_manager.ask_vertexai(extraction_prompt)
            print(response)
            cleaned_response = self._clean_llm_response(response)
            result = json.loads(cleaned_response)
            # print(result)
            # Validate response
            if result.get('is_valid_sql', False) and result.get('confidence', 0) >= LLM_CONFIDENCE_THRESHOLD:
                extracted_columns = set(result.get('extracted_columns', []))
                self.logger.debug(f"LLM extracted {len(extracted_columns)} columns with confidence {result.get('confidence', 0):.2f}")
                return extracted_columns
            else:
                self.logger.debug(f"LLM extraction below confidence threshold or invalid SQL")
                return set()
        
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM JSON response: {e}")
            self.logger.debug(f"Raw LLM response: {response[:200]}...")
            return set()
        
        except Exception as e:
            self.logger.error(f"LLM extraction failed: {e}")
            return set()
    
    def _clean_llm_response(self, response: str) -> str:
        """Clean LLM response to extract valid JSON"""
        if not response:
            raise ValueError("Empty response from LLM")
        
        # Remove markdown code blocks if present
        import re
        
        # Pattern to match ```json ... ``` or ``` ... ```
        json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        match = re.search(json_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if match:
            # Extract JSON from code blocks
            json_content = match.group(1).strip()
            self.logger.debug("Extracted JSON from markdown code blocks")
            return json_content
        
        # If no code blocks, try to find JSON object directly
        # Look for { ... } pattern
        json_object_pattern = r'\{.*\}'
        match = re.search(json_object_pattern, response, re.DOTALL)
        
        if match:
            json_content = match.group(0).strip()
            self.logger.debug("Extracted JSON object directly")
            return json_content
        
        # If still no JSON found, return the response as-is and let json.loads handle the error
        self.logger.warning("Could not extract JSON from LLM response, returning raw response")
        return response.strip()
    
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
        self.logger.info(f"Starting config generation for table: {table_name}")
        self.logger.debug(f"Analysis result keys: {list(analysis_result.keys())}")
        self.logger.debug(f"Analysis result structure: {analysis_result}")
        
        
        # Create config filename
        table_simple_name = table_name.replace('.', '_')
        config_filename = f"tbc_{table_simple_name}.conf"
        config_path = os.path.join(self.system_config.config_dir, config_filename)
        self.logger.info(f"Config file path: {config_path}")
        
        # Ensure config directory exists
        os.makedirs(self.system_config.config_dir, exist_ok=True)
        
        # Get frequent columns
        top_columns = analysis_result.get('top_columns', [])
        self.logger.debug(f"Top columns data: {top_columns}")

        frequent_columns = []
        try:
            for item in top_columns:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    col, count = item[0], item[1]
                    if isinstance(count, (int, float)) and count >= LOG_ANALYSIS_MIN_USAGE:
                        frequent_columns.append(str(col))
                else:
                    self.logger.warning(f"Unexpected top_columns format: {item}")
        except Exception as e:
            self.logger.error(f"Error processing top_columns: {e}")
            frequent_columns = []

        self.logger.info(f"Frequent columns found: {frequent_columns}")
        
        # Create configuration content
        config = configparser.ConfigParser()
        config.optionxform = str  # Preserve case
        
        section_name = "transcation"
        config.add_section(section_name)
        
        
        # Sanitize table key for option names (avoid dots)
        def _sanitize(name: str) -> str:
            return name.replace('.', '_')
        safe_table_key = _sanitize(table_name)
# Basic settings
        config.set(section_name, 'BASEDIRECTORY', BASE_OUTPUT_DIR)
        config.set(section_name, 'USERID', str(user_id))
        config.set(section_name, 'PASSWORD', str(password or ""))
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
        columns_key = f"{safe_table_key}_Columns"
        try:
            if frequent_columns:
                # Ensure all column names are strings
                string_columns = [str(col) for col in frequent_columns]
                config.set(section_name, columns_key, ','.join(string_columns))
                self.logger.info(f"Set {len(string_columns)} frequent columns for {table_name}")
            else:
                # Fallback to common columns based on table type
                default_columns = self._get_default_columns_for_table(table_name)
                string_default_columns = [str(col) for col in default_columns]
                config.set(section_name, columns_key, ','.join(string_default_columns))
                self.logger.info(f"Set {len(string_default_columns)} default columns for {table_name}")
                
        except Exception as column_error:
            self.logger.error(f"Error setting column configuration: {column_error}")
            # Set minimal fallback
            config.set(section_name, columns_key, "id,name,date")
        
        # Date criteria
        where_col_key = f"{safe_table_key}_WhereCol_daycriteria"
        config.set(section_name, where_col_key, 'dwh_business_date')
        
        # Enhanced metadata with LLM usage info
        try:
            
            usage_stats = analysis_result.get('usage_statistics', {})
config.set(section_name, f"{safe_table_key}_Enhanced_Analysis_Date", str(datetime.now().strftime("%Y%m%d_%H%M%S")))
            config.set(section_name, f"{safe_table_key}_TotalQueries", str(analysis_result.get('total_queries', 0)))
            config.set(section_name, f"{safe_table_key}_ValidQueries", str(analysis_result.get('valid_queries', 0)))
            config.set(section_name, f"{safe_table_key}_ColumnsFound", str(len(analysis_result.get('column_usage_frequency', {}))))
            
            # Safely get analysis method
            analysis_method = usage_stats.get('analysis_method', 'rule_based')
            config.set(section_name, f"{safe_table_key}_AnalysisMethod", str(analysis_method))
            
            # Convert specific numeric values to strings with explicit type checking
            llm_calls = usage_stats.get('llm_calls_made', 0)
            config.set(section_name, f"{safe_table_key}_LLMCallsMade", str(llm_calls))
            
            total_unique = usage_stats.get('total_unique_columns', 0)
            config.set(section_name, f"{safe_table_key}_TotalUniqueColumns", str(total_unique))
            
            total_refs = usage_stats.get('total_column_references', 0)
            config.set(section_name, f"{safe_table_key}_TotalColumnReferences", str(total_refs))
            
            # Handle average with safe conversion
            avg_usage = usage_stats.get('average_usage_per_column', 0.0)
            if isinstance(avg_usage, (int, float)):
                config.set(section_name, f"{safe_table_key}_AverageUsagePerColumn", str(round(float(avg_usage), 2)))
            else:
                config.set(section_name, f"{safe_table_key}_AverageUsagePerColumn", "0.0")
            
            unique_clients = usage_stats.get('unique_clients', 0)
            config.set(section_name, f"{safe_table_key}_UniqueClients", str(unique_clients))
            
            # Handle most_used_column (could be tuple or None)
            most_used = usage_stats.get('most_used_column', None)
            if most_used and isinstance(most_used, (list, tuple)) and len(most_used) >= 2:
                config.set(section_name, f"{safe_table_key}_MostUsedColumn", f"{str(most_used[0])}:{str(most_used[1])}")
            else:
                config.set(section_name, f"{safe_table_key}_MostUsedColumn", "None")
            
            # LLM configuration info (ensure boolean values are converted to strings)
            config.set(section_name, f"{safe_table_key}_LLMEnabled", str(bool(ENABLE_LLM_ANALYSIS and LLM_CLIENT_AVAILABLE)))
            config.set(section_name, f"{safe_table_key}_HybridAnalysis", str(bool(USE_HYBRID_APPROACH)))
            
            self.logger.info("Successfully set all configuration metadata")
            
        except Exception as metadata_error:
            self.logger.error(f"Error setting configuration metadata: {metadata_error}")
            self.logger.error(f"Usage statistics that caused error: {usage_stats}")
        
        try:
            self.logger.info("Validating configuration before saving...")
            
            # Check all sections and options
            for section in config.sections():
                self.logger.debug(f"Section: {section}")
                for option in config.options(section):
                    value = config.get(section, option)
                    self.logger.debug(f"  {option} = {value} (type: {type(value)})")
                    
                    # Ensure value is string
                    if not isinstance(value, str):
                        self.logger.warning(f"Converting non-string value to string: {option} = {value}")
                        config.set(section, option, str(value))
            
            self.logger.info("Configuration validation completed successfully")
            
        except Exception as validation_error:
            self.logger.error(f"Configuration validation failed: {validation_error}")
            raise
        
        try:
            self.logger.info("Performing final validation of configuration values...")
            
            validation_errors = []
            for section_name_check in config.sections():
                for option_name in config.options(section_name_check):
                    try:
                        value = config.get(section_name_check, option_name)
                        if not isinstance(value, str):
                            self.logger.warning(f"Converting non-string value: {option_name} = {value} ({type(value)})")
                            config.set(section_name_check, option_name, str(value))
                    except Exception as val_error:
                        validation_errors.append(f"{option_name}: {val_error}")
            
            if validation_errors:
                self.logger.warning(f"Validation issues found: {validation_errors}")
            else:
                self.logger.info("Configuration validation completed successfully")
                    
        except Exception as validation_error:
            self.logger.error(f"Configuration validation failed: {validation_error}")
        
        # Save configuration file
        try:
            # First, try to validate that all values can be written
            test_validation_passed = True
            for section_name_check in config.sections():
                for option_name in config.options(section_name_check):
                    try:
                        value = config.get(section_name_check, option_name)
                        if not isinstance(value, str):
                            config.set(section_name_check, option_name, str(value))
                    except Exception as test_error:
                        self.logger.error(f"Cannot convert option {option_name}: {test_error}")
                        test_validation_passed = False
            
            if not test_validation_passed:
                raise Exception("Configuration validation failed - cannot write file")
            
            # Now try to write the file
            with open(config_path, 'w', encoding='utf-8') as configfile:
                config.write(configfile)
            
            # Verify file was created and has content
            if os.path.exists(config_path) and os.path.getsize(config_path) > 0:
                self.logger.info(f"Configuration file generated successfully: {config_path}")
                return config_path
            else:
                raise Exception("Configuration file was not created properly")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration file: {e}")
            self.logger.error(f"Config sections: {config.sections()}")
            for section in config.sections():
                self.logger.error(f"Section {section} options: {config.options(section)}")
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


    def generate_json_files(self, table_name: str, analysis_result: dict) -> dict:
        try:
            os.makedirs(self.system_config.output_dir, exist_ok=True)
        except Exception:
            pass
        safe = table_name.replace('.', '_')
        out_summary = os.path.join(self.system_config.output_dir, f"{safe}_frequent_columns.json")
        out_full    = os.path.join(self.system_config.output_dir, f"{safe}_analysis.json")

        payload_summary = {
            "table_name": table_name,
            "generated_at": datetime.now().isoformat(),
            "min_usage_threshold": LOG_ANALYSIS_MIN_USAGE,
            "frequent_columns": [c for c, n in analysis_result.get("top_columns", []) if isinstance(n, (int, float)) and n >= LOG_ANALYSIS_MIN_USAGE],
            "top_columns": analysis_result.get("top_columns", [])
        }
        with open(out_summary, "w", encoding="utf-8") as f:
            json.dump(payload_summary, f, ensure_ascii=False, indent=2)

        def _default(obj):
            try:
                return obj.isoformat()
            except Exception:
                return str(obj)

        with open(out_full, "w", encoding="utf-8") as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2, default=_default)

        self.logger.info(f"JSON summary written: {out_summary}")
        self.logger.info(f"JSON full written: {out_full}")
        return {"summary_json": out_summary, "full_json": out_full}
# =============================================================================
# MAIN PROCESSING CLASS
# =============================================================================

class FrequentColumnProcessor:
    """Main processor for discovering frequent columns"""
    
    def __init__(self, user_id: str, password: str):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.credential_manager = CredentialManager()
        secure_password = self.credential_manager.get_secure_password(password)
        
        
        self.secure_password = secure_password
# Initialize configurations
        db_config = DatabaseConfig(
            host=DB_HOST,
            user_id=user_id,
            password=secure_password,
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
        self.password = (self.secure_password or "")
        
        
        
        # Create output directories
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        os.makedirs(CONFIG_OUTPUT_DIR, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('frequent_columns.log', encoding = 'utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def debug_analysis_result(analysis_result: Dict[str, Any], logger):
        """Debug function to inspect analysis result structure"""
        logger.info("=== ANALYSIS RESULT DEBUG ===")
        for key, value in analysis_result.items():
            logger.info(f"{key}: {type(value)} = {value}")
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    logger.info(f"  {sub_key}: {type(sub_value)} = {sub_value}")
        logger.info("=== END DEBUG ===")
    
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
            
            FrequentColumnProcessor.debug_analysis_result(analysis_result, self.logger)
            
            # Generate configuration file
            self.logger.info(f"Generating configuration file for table: {table_name}")
            self.logger.debug(f"Analysis result before config generation: {json.dumps(analysis_result, indent=2, default=str)}")

            try:
                config_path = self.config_generator.generate_config_file(
                    table_name, analysis_result, self.user_id, (self.secure_password or self.password or "")
                )
                _json_paths = self.config_generator.generate_json_files(table_name, analysis_result)
                self.logger.info(f"Configuration file generated successfully: {config_path}")
            except Exception as config_error:
                self.logger.error(f"Failed to generate configuration file: {config_error}")
                self.logger.error(f"Analysis result that caused the error: {analysis_result}")
                raise
            
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
    parser.add_argument('--password', '-p', type=str, required=False, help='Database password (optional, will use encrypted file by default)')
    parser.add_argument('--output-dir', '-o', type=str, default='./data_output', 
                       help='Output directory for config files')
    parser.add_argument('--disable-llm', action='store_true', help='Disable LLM analysis (use rule-based only)')
    
    args = parser.parse_args()
    
    # Update LLM settings based on arguments
    global ENABLE_LLM_ANALYSIS
    if args.disable_llm:
        ENABLE_LLM_ANALYSIS = False
        print("LLM analysis disabled by user request")
    
    # Update output directory if specified
    #global CONFIG_OUTPUT_DIR
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
    
    validate_environment()
    
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