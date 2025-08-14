# 2_get_distinct_values.py
"""
Core Module 2: Get Distinct Values
==================================
Input: Configuration file (tbc_XXX.conf) from Step 1
Output: JSON file with distinct values and metadata

This module processes the configuration file generated in Step 1, extracts distinct values
for each column, applies business rules, and generates the final JSON output for ORASS.
"""

import os
import sys
import json
import logging
import configparser
import subprocess
import tempfile
import copy
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

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

# Business Rules Settings
MAX_DISTINCT_VALUES_FOR_EVAL = 100          # Columns with <= 100 distinct values need BA evaluation
MAX_DISTINCT_VALUES_FOR_PARTIAL = 10000     # Columns with <= 10000 distinct values need partial evaluation
MAX_DISTINCT_VALUES_TO_FETCH = 100          # Maximum distinct values to fetch per column

# Output Settings
DEFAULT_OUTPUT_DIR = "./output"
JSON_OUTPUT_FILE = "all_final_output.json"
ENUM100_OUTPUT_FILE = "enum100_output.json"
ENUM10000_OUTPUT_FILE = "enum10000_output.json"

# Date Settings
DEFAULT_DAYS_BACK = 90  # Default number of days for data analysis

# =============================================================================
# CORE CLASSES AND ENUMS
# =============================================================================

class Decision(Enum):
    """Business rule decisions for columns"""
    COLUMNS_NEEDBA_EVAL = "COLUMNS_NEEDBA_EVAL"                    # <= 100 distinct values
    COLUMNS_NOTREJALLPHASE_NEEDBA_EVAL = "COLUMNS_NOTREJALLPHASE_NEEDBA_EVAL"  # <= 10000 distinct values
    COLUMNS_REJECTED = "COLUMNS_REJECTED"                          # > 10000 distinct values

@dataclass
class RuleDecision:
    """Business rule decision result"""
    enum: Decision
    reject_reason: str

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

class DatabaseClient:
    """Handles database operations and query execution"""
    
    def __init__(self, db_config: DatabaseConfig, system_config: SystemConfig):
        self.db_config = db_config
        self.system_config = system_config
        self.logger = logging.getLogger(__name__)
    
    def execute_query(self, query: str, output_file: str, format_type: str = "TXT") -> bool:
        """Execute a database query and save results to file"""
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
                self.logger.error(f"Query execution failed: {result.stderr}")
                return False
            
            # Check if output file was created and has content
            if not os.path.exists(output_file):
                self.logger.error(f"Output file was not created: {output_file}")
                return False
            
            file_size = os.path.getsize(output_file)
            self.logger.debug(f"Output file size: {file_size} bytes")
            
            return file_size > 0
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Query execution timed out: {query}")
            return False
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
        """Parse schema file and return column type mappings"""
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
                    column_type = parts[1].strip().rstrip(',')
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

class MetadataManager:
    """Manages table metadata and column statistics"""
    
    def __init__(self, db_client: DatabaseClient, config_data: Dict[str, Any]):
        self.db_client = db_client
        self.config_data = config_data
        self.table_schemas = {}
        self.logger = logging.getLogger(__name__)
    
    def load_table_metadata(self, table_name: str) -> bool:
        """Load metadata for a specific table"""
        try:
            self.logger.info(f"Loading metadata for table: {table_name}")
            schema = self.db_client.get_table_schema(table_name)
            
            if schema:
                self.table_schemas[table_name] = schema
                self.logger.info(f"Loaded metadata for table: {table_name} ({len(schema)} columns)")
                return True
            else:
                self.logger.error(f"Failed to load metadata for table: {table_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading metadata for table {table_name}: {e}")
            return False
    
    def get_column_distinct_count(self, table_name: str, column_name: str) -> Optional[int]:
        """Get distinct count for a specific column"""
        try:
            # Build date conditions
            safe_table_key = table_name.replace('.', '_')
            where_clause_col = (self.config_data.get(f"{table_name}_WhereCol_daycriteria") or self.config_data.get(f"{safe_table_key}_WhereCol_daycriteria") or "dwh_business_date").lower()
            no_of_days = DEFAULT_DAYS_BACK
            
            start_date = datetime.strptime(self.config_data.get('START_DATE', '20250311'), '%Y%m%d')
            n_days_ago = start_date - timedelta(days=no_of_days)
            
            where_conditions = self._build_date_conditions(
                table_name, where_clause_col, n_days_ago, start_date
            )
            
            # Determine query type based on column type
            column_info = self.table_schemas.get(table_name, {}).get(column_name, "string|string")
            mapped_type = column_info.split("|")[1]
            
            if mapped_type == "nestedtype":
                query = f"select count({column_name}) from {table_name} where {where_conditions}"
            else:
                query = f"select count(distinct {column_name}) from {table_name} where {where_conditions}"
            
            # Execute query
            with tempfile.NamedTemporaryFile(mode='w', suffix='.count', delete=False) as temp_file:
                output_file = temp_file.name
            
            try:
                if self.db_client.execute_query(query, output_file):
                    result = self._parse_count_result(output_file)
                    return int(result) if result.isdigit() else None
                return None
            finally:
                if os.path.exists(output_file):
                    os.unlink(output_file)
                    
        except Exception as e:
            self.logger.error(f"Error getting distinct count for {table_name}.{column_name}: {e}")
            return None
    
    def get_distinct_values(self, table_name: str, column_name: str) -> List[str]:
        """Get distinct values for a specific column"""
        try:
            # Build date conditions
            safe_table_key = table_name.replace('.', '_')
            where_clause_col = (self.config_data.get(f"{table_name}_WhereCol_daycriteria") or self.config_data.get(f"{safe_table_key}_WhereCol_daycriteria") or "dwh_business_date").lower()
            no_of_days = DEFAULT_DAYS_BACK
            
            start_date = datetime.strptime(self.config_data.get('START_DATE', '20250311'), '%Y%m%d')
            n_days_ago = start_date - timedelta(days=no_of_days)
            
            where_conditions = self._build_date_conditions(
                table_name, where_clause_col, n_days_ago, start_date
            )
            
            # Determine query type based on column type
            column_info = self.table_schemas.get(table_name, {}).get(column_name, "string|string")
            mapped_type = column_info.split("|")[1]
            
            if mapped_type == "nestedtype":
                query = f"select {column_name} from {table_name} where {where_conditions} limit {MAX_DISTINCT_VALUES_TO_FETCH}"
            else:
                query = f"select distinct {column_name} from {table_name} where {where_conditions} limit {MAX_DISTINCT_VALUES_TO_FETCH}"
            
            # Execute query
            with tempfile.NamedTemporaryFile(mode='w', suffix='.values', delete=False) as temp_file:
                output_file = temp_file.name
            
            try:
                if self.db_client.execute_query(query, output_file):
                    return self._parse_distinct_values(output_file)
                return []
            finally:
                if os.path.exists(output_file):
                    os.unlink(output_file)
                    
        except Exception as e:
            self.logger.error(f"Error getting distinct values for {table_name}.{column_name}: {e}")
            return []
    
    def _build_date_conditions(self, table_name: str, where_col: str, 
                             start_date: datetime, end_date: datetime) -> str:
        """Build date condition clause based on column type"""
        column_info = self.table_schemas.get(table_name, {}).get(where_col, "string|string")
        mapped_type = column_info.split("|")[1]
        
        if mapped_type == "string":
            start_str = f"'{start_date.strftime('%Y%m%d')}'"
            end_str = f"'{end_date.strftime('%Y%m%d')}'"
        elif mapped_type == "timestamp":
            start_str = f"CAST('{start_date.strftime('%Y-%m-%d %H:%M:%S')}' AS timestamp)"
            end_str = f"CAST('{end_date.strftime('%Y-%m-%d %H:%M:%S')}' AS timestamp)"
        else:
            start_str = start_date.strftime('%Y%m%d')
            end_str = end_date.strftime('%Y%m%d')
        
        return f"{where_col} > {start_str} and {where_col} < {end_str}"
    
    def _parse_count_result(self, output_file: str) -> str:
        """Parse count result from output file"""
        try:
            with open(output_file, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    return lines[1].strip()
        except Exception as e:
            self.logger.error(f"Error parsing count result: {e}")
        return "0"
    
    def _parse_distinct_values(self, output_file: str) -> List[str]:
        """Parse distinct values from output file"""
        values = []
        try:
            with open(output_file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if i == 0:  # Skip header
                        continue
                    value = line.strip()
                    if value and value.lower() != 'null':
                        values.append(value)
        except Exception as e:
            self.logger.error(f"Error parsing distinct values: {e}")
        return values

class RulesEngine:
    """Applies business rules to determine column processing decisions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def evaluate_column(self, table_name: str, column_name: str, 
                       distinct_count: int, column_type: str) -> RuleDecision:
        """Evaluate a column against business rules"""
        
        # Rule 1: Check distinct count thresholds
        if distinct_count <= MAX_DISTINCT_VALUES_FOR_EVAL:
            return RuleDecision(
                enum=Decision.COLUMNS_NEEDBA_EVAL,
                reject_reason=""
            )
        elif distinct_count <= MAX_DISTINCT_VALUES_FOR_PARTIAL:
            return RuleDecision(
                enum=Decision.COLUMNS_NOTREJALLPHASE_NEEDBA_EVAL,
                reject_reason=""
            )
        else:
            return RuleDecision(
                enum=Decision.COLUMNS_REJECTED,
                reject_reason=f"Too many distinct values: {distinct_count}"
            )
    
    def apply_rules(self, column_stats: Dict[str, Dict[str, int]], 
                   metadata: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, RuleDecision]]:
        """Apply all rules to column statistics"""
        decisions = {}
        
        for table_name, columns in column_stats.items():
            decisions[table_name] = {}
            
            for column_name, distinct_count in columns.items():
                try:
                    column_info = metadata.get(table_name, {}).get(column_name, "string|string")
                    column_type = column_info.split("|")[1]
                    
                    decision = self.evaluate_column(
                        table_name, column_name, distinct_count, column_type
                    )
                    decisions[table_name][column_name] = decision
                    
                    self.logger.debug(f"Column {table_name}.{column_name}: {distinct_count} values -> {decision.enum.value}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing column {table_name}.{column_name}: {e}")
                    decisions[table_name][column_name] = RuleDecision(
                        enum=Decision.COLUMNS_REJECTED,
                        reject_reason=f"Processing error: {e}"
                    )
        
        return decisions

class JSONGenerator:
    """Generates final JSON output for ORASS"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_json_output(self, master_dict: Dict[str, Any]) -> Dict[str, str]:
        """Generate all JSON output files"""
        try:
            # Extract data from master dictionary
            config_data = master_dict.get('APPLICATION_CONFIG', {})
            metadata = master_dict.get('META_DATA', {})
            column_stats = master_dict.get('COLUMN_DISTINCT_VALUES', {})
            decisions = master_dict.get('DISTINCT_VALUE_DECISION', {})
            distinct_values = master_dict.get('DISTINCT_VALUES', {})
            
            # Get tables list
            tables_str = config_data.get('TABLES', '')
            tables_list = [t.strip() for t in tables_str.split(',') if t.strip()]
            
            # Generate JSON structures
            final_json = []
            final_json_enum100 = []
            final_json_enum10000 = []
            
            for table_name in tables_list:
                table_structures = self._create_table_structures(
                    table_name, config_data, metadata, column_stats, decisions, distinct_values
                )
                
                if table_structures['main']:
                    final_json.append(table_structures['main'])
                if table_structures['enum100']:
                    final_json_enum100.append(table_structures['enum100'])
                if table_structures['enum10000']:
                    final_json_enum10000.append(table_structures['enum10000'])
            
            # Save output files
            output_files = {}
            
            # Main output
            main_output = {"trainingdata": final_json}
            main_file = os.path.join(self.output_dir, JSON_OUTPUT_FILE)
            with open(main_file, 'w') as f:
                json.dump(main_output, f, indent=4)
            output_files['main'] = main_file
            self.logger.info(f"Generated main JSON: {main_file}")
            
            # Enum100 output
            if final_json_enum100:
                enum100_output = {"trainingdata": final_json_enum100}
                enum100_file = os.path.join(self.output_dir, ENUM100_OUTPUT_FILE)
                with open(enum100_file, 'w') as f:
                    json.dump(enum100_output, f, indent=4)
                output_files['enum100'] = enum100_file
                self.logger.info(f"Generated enum100 JSON: {enum100_file}")
            
            # Enum10000 output
            if final_json_enum10000:
                enum10000_output = {"trainingdata": final_json_enum10000}
                enum10000_file = os.path.join(self.output_dir, ENUM10000_OUTPUT_FILE)
                with open(enum10000_file, 'w') as f:
                    json.dump(enum10000_output, f, indent=4)
                output_files['enum10000'] = enum10000_file
                self.logger.info(f"Generated enum10000 JSON: {enum10000_file}")
            
            return output_files
            
        except Exception as e:
            self.logger.error(f"Error generating JSON output: {e}")
            raise
    
    def _create_table_structures(self, table_name: str, config_data: Dict[str, Any],
                               metadata: Dict[str, Dict[str, str]], 
                               column_stats: Dict[str, Dict[str, int]],
                               decisions: Dict[str, Dict[str, RuleDecision]],
                               distinct_values: Dict[str, Dict[str, List[str]]]) -> Dict[str, Any]:
        """Create table structures for different JSON outputs"""
        
        # Get columns for this table
        columns_key = f"{table_name}_Columns"
safe_table_key = table_name.replace('.', '_')
        columns_str = config_data.get(columns_key) or config_data.get(f"{safe_table_key}_Columns", "")
        if not columns_str:
            return {'main': None, 'enum100': None, 'enum10000': None}
        
        columns_list = [col.strip().lower() for col in columns_str.split(",")]
        
        # Base table structure
        base_table = {
            "tablename": table_name,
            "tableDescription": "",
            "tableAlias": [],
            "tableSpecificRules": "",
            "columns": []
        }
        
        # Create copies for different outputs
        main_table = copy.deepcopy(base_table)
        enum100_table = copy.deepcopy(base_table)
        enum10000_table = copy.deepcopy(base_table)
        
        # Process each column
        for column_name in columns_list:
            column_structures = self._create_column_structures(
                table_name, column_name, metadata, column_stats, decisions, distinct_values
            )
            
            if column_structures['main']:
                main_table['columns'].append(column_structures['main'])
            if column_structures['enum100']:
                enum100_table['columns'].append(column_structures['enum100'])
            if column_structures['enum10000']:
                enum10000_table['columns'].append(column_structures['enum10000'])
        
        return {
            'main': main_table if main_table['columns'] else None,
            'enum100': enum100_table if enum100_table['columns'] else None,
            'enum10000': enum10000_table if enum10000_table['columns'] else None
        }
    
    def _create_column_structures(self, table_name: str, column_name: str,
                                metadata: Dict[str, Dict[str, str]],
                                column_stats: Dict[str, Dict[str, int]],
                                decisions: Dict[str, Dict[str, RuleDecision]],
                                distinct_values: Dict[str, Dict[str, List[str]]]) -> Dict[str, Any]:
        """Create column structures for different output types"""
        
        # Base column structure
        base_column = {
            "columnname": column_name,
            "columnDescription": "",
            "columnSpecificRules": "",
            "columnAlias": [],
            "update_date": datetime.now().strftime("%Y-%m-%d"),
            "update_soeid": "system"
        }
        
        # Add metadata if available
        if table_name in metadata and column_name in metadata[table_name]:
            meta_data = metadata[table_name][column_name].split("|")
            base_column["actual_col_type"] = meta_data[0]
            base_column["mapped_col_type"] = meta_data[1]
        else:
            base_column["actual_col_type"] = "string"
            base_column["mapped_col_type"] = "string"
        
        # Add distinct count
        if table_name in column_stats and column_name in column_stats[table_name]:
            base_column["distinct_count"] = str(column_stats[table_name][column_name])
        else:
            base_column["distinct_count"] = "0"
        
        # Add decision information
        decision = None
        if table_name in decisions and column_name in decisions[table_name]:
            decision = decisions[table_name][column_name]
            base_column["enum"] = decision.enum.value
            base_column["RejectReason"] = decision.reject_reason
            base_column["provide_distinct"] = "YES" if decision.enum in [
                Decision.COLUMNS_NEEDBA_EVAL, 
                Decision.COLUMNS_NOTREJALLPHASE_NEEDBA_EVAL
            ] else "NO"
        else:
            base_column["enum"] = Decision.COLUMNS_REJECTED.value
            base_column["RejectReason"] = "No decision available"
            base_column["provide_distinct"] = "NO"
        
        # Add distinct values if available
        base_column["distinct_values"] = []
        base_column["distinct_value_map"] = {}
        
        if (table_name in distinct_values and column_name in distinct_values[table_name]):
            values = [v for v in distinct_values[table_name][column_name] if v]
            base_column["distinct_values"] = values
            if decision and decision.enum == Decision.COLUMNS_NEEDBA_EVAL:
                base_column["distinct_value_map"] = {v: "" for v in values}
        
        # Create different versions based on decision
        result = {'main': copy.deepcopy(base_column), 'enum100': None, 'enum10000': None}
        
        if decision:
            if decision.enum == Decision.COLUMNS_NEEDBA_EVAL:
                enum100_col = copy.deepcopy(base_column)
                # Remove certain fields for enum100 output
                for key in ['provide_distinct', 'RejectReason']:
                    enum100_col.pop(key, None)
                result['enum100'] = enum100_col
                
            elif decision.enum == Decision.COLUMNS_NOTREJALLPHASE_NEEDBA_EVAL:
                enum10000_col = copy.deepcopy(base_column)
                # Remove certain fields for enum10000 output
                for key in ['provide_distinct', 'RejectReason']:
                    enum10000_col.pop(key, None)
                result['enum10000'] = enum10000_col
        
        return result

# =============================================================================
# MAIN PROCESSING CLASS
# =============================================================================

class DistinctValueProcessor:
    """Main processor for extracting distinct values and generating JSON"""
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.config_data = {}
        self.db_client = None
        self.metadata_manager = None
        self.rules_engine = RulesEngine()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('distinct_values.log', encoding = 'utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def load_config_file(self, config_file_path: str) -> bool:
        """Load configuration file generated in Step 1"""
        try:
            if not os.path.exists(config_file_path):
                raise FileNotFoundError(f"Configuration file not found: {config_file_path}")
            
            config = configparser.ConfigParser()
            config.optionxform = str  # Preserve case
            config.read(config_file_path)
            
            # Get the main section (usually 'transcation')
            section_name = config.sections()[0] if config.sections() else 'transcation'
            self.config_data = dict(config.items(section_name))
            
            self.logger.info(f"Loaded configuration from: {config_file_path}")
            self.logger.info(f"Configuration section: {section_name}")
            self.logger.info(f"Tables to process: {self.config_data.get('TABLES', 'None')}")
            
            # Initialize database client
            db_config = DatabaseConfig(
                host=DB_HOST,
                user_id=self.config_data.get('USERID', ''),
                password=self.config_data.get('PASSWORD', ''),
                environment=DB_ENVIRONMENT
            )
            
            system_config = SystemConfig(
                java_path=WINDOWS_JAVA_PATH if os.name == 'nt' else LINUX_JAVA_PATH,
                jar_path=WINDOWS_JAR_PATH if os.name == 'nt' else LINUX_JAR_PATH,
                output_dir=self.config_data.get('OUTPUT_DIRECTORY', DEFAULT_OUTPUT_DIR)
            )
            
            self.db_client = DatabaseClient(db_config, system_config)
            self.metadata_manager = MetadataManager(self.db_client, self.config_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading configuration file: {e}")
            return False
    
    def process_config(self, config_file_path: str, output_dir: str = None) -> Dict[str, str]:
        """Process configuration file and generate JSON outputs"""
        try:
            # Load configuration
            if not self.load_config_file(config_file_path):
                raise Exception("Failed to load configuration file")
            
            # Set output directory
            if output_dir:
                self.config_data['OUTPUT_DIRECTORY'] = output_dir
            
            output_directory = self.config_data.get('OUTPUT_DIRECTORY', DEFAULT_OUTPUT_DIR)
            os.makedirs(output_directory, exist_ok=True)
            
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
            
            # Initialize master dictionary
            master_dict = {
                'APPLICATION_CONFIG': self.config_data,
                'META_DATA': {},
                'COLUMN_DISTINCT_VALUES': {},
                'DISTINCT_VALUE_DECISION': {},
                'DISTINCT_VALUES': {}
            }
            
            # Get tables list
            tables_str = self.config_data.get('TABLES', '')
            tables_list = [t.strip() for t in tables_str.split(',') if t.strip()]
            
            if not tables_list:
                raise Exception("No tables specified in configuration")
            
            # Step 1: Load table metadata
            self.logger.info("Loading table metadata...")
            for table_name in tables_list:
                if not self.metadata_manager.load_table_metadata(table_name):
                    self.logger.warning(f"Failed to load metadata for table: {table_name}")
                    continue
            
            master_dict['META_DATA'] = self.metadata_manager.table_schemas
            
            # Step 2: Get distinct counts for all columns
            self.logger.info("Getting distinct counts for columns...")
            for table_name in tables_list:
                columns_key = f"{table_name}_Columns"
                safe_table_key = table_name.replace('.', '_')
                columns_str = self.config_data.get(columns_key) or self.config_data.get(f"{safe_table_key}_Columns", "")
                
                if not columns_str:
                    self.logger.warning(f"No columns specified for table: {table_name}")
                    continue
                
                columns_list = [col.strip().lower() for col in columns_str.split(",")]
                
                for column_name in columns_list:
                    self.logger.info(f"Processing column: {table_name}.{column_name}")
                    
                    distinct_count = self.metadata_manager.get_column_distinct_count(table_name, column_name)
                    
                    if distinct_count is not None:
                        if table_name not in master_dict['COLUMN_DISTINCT_VALUES']:
                            master_dict['COLUMN_DISTINCT_VALUES'][table_name] = {}
                        master_dict['COLUMN_DISTINCT_VALUES'][table_name][column_name] = distinct_count
                        self.logger.info(f"Column {table_name}.{column_name}: {distinct_count} distinct values")
                    else:
                        self.logger.warning(f"Could not get distinct count for {table_name}.{column_name}")
            
            # Step 3: Apply business rules
            self.logger.info("Applying business rules...")
            decisions = self.rules_engine.apply_rules(
                master_dict['COLUMN_DISTINCT_VALUES'], 
                master_dict['META_DATA']
            )
            master_dict['DISTINCT_VALUE_DECISION'] = decisions
            
            # Step 4: Get distinct values for approved columns
            self.logger.info("Fetching distinct values for approved columns...")
            for table_name, columns in decisions.items():
                for column_name, decision in columns.items():
                    if decision.enum in [Decision.COLUMNS_NEEDBA_EVAL, Decision.COLUMNS_NOTREJALLPHASE_NEEDBA_EVAL]:
                        self.logger.info(f"Fetching distinct values for {table_name}.{column_name}")
                        
                        distinct_values = self.metadata_manager.get_distinct_values(table_name, column_name)
                        
                        if table_name not in master_dict['DISTINCT_VALUES']:
                            master_dict['DISTINCT_VALUES'][table_name] = {}
                        master_dict['DISTINCT_VALUES'][table_name][column_name] = distinct_values
                        
                        self.logger.info(f"Retrieved {len(distinct_values)} distinct values for {table_name}.{column_name}")
            
            # Step 5: Generate JSON outputs
            self.logger.info("Generating JSON outputs...")
            json_generator = JSONGenerator(output_directory)
            output_files = json_generator.generate_json_output(master_dict)
            
            # Log summary
            self._log_processing_summary(master_dict, output_files)
            
            return output_files
            
        except Exception as e:
            self.logger.error(f"Error processing configuration: {e}")
            raise
    
    def _log_processing_summary(self, master_dict: Dict[str, Any], output_files: Dict[str, str]):
        """Log processing summary"""
        self.logger.info("=" * 60)
        self.logger.info("PROCESSING SUMMARY")
        self.logger.info("=" * 60)
        
        # Tables processed
        tables_processed = len(master_dict.get('META_DATA', {}))
        self.logger.info(f"Tables processed: {tables_processed}")
        
        # Columns analyzed
        total_columns = sum(len(cols) for cols in master_dict.get('COLUMN_DISTINCT_VALUES', {}).values())
        self.logger.info(f"Total columns analyzed: {total_columns}")
        
        # Decision breakdown
        decisions = master_dict.get('DISTINCT_VALUE_DECISION', {})
        needba_count = 0
        partial_count = 0
        rejected_count = 0
        
        for table_decisions in decisions.values():
            for decision in table_decisions.values():
                if decision.enum == Decision.COLUMNS_NEEDBA_EVAL:
                    needba_count += 1
                elif decision.enum == Decision.COLUMNS_NOTREJALLPHASE_NEEDBA_EVAL:
                    partial_count += 1
                else:
                    rejected_count += 1
        
        self.logger.info(f"Columns needing BA evaluation (≤{MAX_DISTINCT_VALUES_FOR_EVAL} values): {needba_count}")
        self.logger.info(f"Columns needing partial evaluation (≤{MAX_DISTINCT_VALUES_FOR_PARTIAL} values): {partial_count}")
        self.logger.info(f"Columns rejected (>{MAX_DISTINCT_VALUES_FOR_PARTIAL} values): {rejected_count}")
        
        # Output files
        self.logger.info("Output files generated:")
        for output_type, file_path in output_files.items():
            self.logger.info(f"  {output_type}: {file_path}")
        
        self.logger.info("=" * 60)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_config_file(config_file_path: str) -> bool:
    """Validate configuration file format and required fields"""
    try:
        if not os.path.exists(config_file_path):
            print(f"Error: Configuration file not found: {config_file_path}")
            return False
        
        config = configparser.ConfigParser()
        config.read(config_file_path)
        
        if not config.sections():
            print("Error: No sections found in configuration file")
            return False
        
        section = config.sections()[0]
        required_fields = ['USERID', 'PASSWORD', 'TABLES']
        
        for field in required_fields:
            if not config.has_option(section, field):
                print(f"Error: Required field '{field}' missing from configuration")
                return False
        
        # Check if tables have column definitions
        tables_str = config.get(section, 'TABLES')
        tables_list = [t.strip() for t in tables_str.split(',') if t.strip()]
        
        for table_name in tables_list:
            columns_key = f"{table_name}_Columns"
            safe_table_key = table_name.replace('.', '_')
            if not (config.has_option(section, columns_key) or config.has_option(section, f"{safe_table_key}_Columns")):
                print(f"Warning: No columns defined for table '{table_name}' (keys tried: {columns_key}, {safe_table_key}_Columns)")
        
        print(f"✅ Configuration file validation passed")
        print(f"   - Section: {section}")
        print(f"   - Tables: {len(tables_list)}")
        print(f"   - User: {config.get(section, 'USERID')}")
        
        return True
        
    except Exception as e:
        print(f"Error validating configuration file: {e}")
        return False

def preview_config_processing(config_file_path: str):
    """Preview what will be processed without executing queries"""
    try:
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(config_file_path)
        
        section_name = config.sections()[0]
        config_data = dict(config.items(section_name))
        
        print("=" * 60)
        print("CONFIGURATION PREVIEW")
        print("=" * 60)
        
        print(f"Section: {section_name}")
        print(f"User: {config_data.get('USERID', 'Not specified')}")
        print(f"Output Directory: {config_data.get('OUTPUT_DIRECTORY', DEFAULT_OUTPUT_DIR)}")
        print(f"Start Date: {config_data.get('START_DATE', 'Not specified')}")
        
        # Tables and columns
        tables_str = config_data.get('TABLES', '')
        tables_list = [t.strip() for t in tables_str.split(',') if t.strip()]
        
        print(f"\nTables to process: {len(tables_list)}")
        for table_name in tables_list:
            columns_key = f"{table_name}_Columns"
safe_table_key = table_name.replace('.', '_')
        columns_str = config_data.get(columns_key) or config_data.get(f"{safe_table_key}_Columns", "")
            columns_list = [col.strip() for col in columns_str.split(",") if col.strip()]
            
            print(f"\n  Table: {table_name}")
            print(f"    Columns ({len(columns_list)}): {', '.join(columns_list[:5])}")
            if len(columns_list) > 5:
                print(f"    ... and {len(columns_list) - 5} more")
            
            safe_table_key = table_name.replace('.', '_')
            where_col = (config_data.get(f"{table_name}_WhereCol_daycriteria") or config_data.get(f"{safe_table_key}_WhereCol_daycriteria") or "dwh_business_date")
            print(f"    Date column: {where_col}")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"Error previewing configuration: {e}")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract distinct values and generate JSON output')
    parser.add_argument('--config_file', type=str, help='Configuration file path (tbc_XXX.conf)')
    parser.add_argument('--output-dir', '-o', type=str, help='Output directory (overrides config)')
    parser.add_argument('--validate', '-v', action='store_true', help='Validate configuration file only')
    parser.add_argument('--preview', '-p', action='store_true', help='Preview configuration without processing')
    parser.add_argument('--dry-run', action='store_true', help='Dry run - validate and preview only')
    
    args = parser.parse_args()
    
    # Validate configuration file
    if not validate_config_file(args.config_file):
        return 1
    
    if args.validate:
        print("Configuration validation completed successfully!")
        return 0
    
    # Preview configuration
    if args.preview or args.dry_run:
        preview_config_processing(args.config_file)
        if args.dry_run:
            return 0
    
    try:
        # Initialize processor
        processor = DistinctValueProcessor()
        
        # Process configuration
        print(f"Processing configuration file: {args.config_file}")
        output_files = processor.process_config(args.config_file, args.output_dir)
        
        # Print results
        print(f"\n✅ Processing completed successfully!")
        print(f"Output files generated:")
        for output_type, file_path in output_files.items():
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            print(f"  {output_type}: {file_path} ({file_size:,} bytes)")
        
        return 0
        
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())