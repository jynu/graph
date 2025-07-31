# training_data_main.py - Complete main file with all components
import io
import os
import sys
import subprocess
import json
import configparser
import copy
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Import the frequent column function
from getFrequentColumn import getFrequentColumn

# =============================================================================
# CONFIGURATION CLASSES AND MANAGER
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
    """System-wide configuration"""
    base_directory: str
    output_directory: str
    start_date: datetime
    jdk_path: str
    jar_path: str
    date_type: str = "transcation"

class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_files: list):
        self.config = configparser.RawConfigParser()
        self.config._interpolation = configparser.ExtendedInterpolation()
        self.config.optionxform = str
        self.config_data = {}
        
        for config_file in config_files:
            if os.path.exists(config_file):
                self.config.read(config_file)
            else:
                raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    def get_section_config(self, section: str) -> Dict[str, Any]:
        """Get configuration for a specific section"""
        if not self.config.has_section(section):
            raise ValueError(f"Section '{section}' not found in configuration")
        
        return dict(self.config.items(section))
    
    def get_database_config(self, section: str = "transcation") -> DatabaseConfig:
        """Get database configuration"""
        config_data = self.get_section_config(section)
        
        # Handle Windows authentication differently
        if os.name == 'nt':  # Windows
            user_id = config_data.get('USERID', '')
            password = config_data.get('PASSWORD', '')
        else:
            # For Linux, use environment variables
            user_id = os.environ.get('USER_FOR_TRAINING', '')
            password = self._decrypt_password()
        
        return DatabaseConfig(
            host=self._get_host_from_config(config_data),
            user_id=user_id,
            password=password
        )
    
    def get_system_config(self, section: str = "transcation") -> SystemConfig:
        """Get system configuration"""
        config_data = self.get_section_config(section)
        
        return SystemConfig(
            base_directory=config_data['BASEDIRECTORY'],
            output_directory=config_data['OUTPUT_DIRECTORY'],
            start_date=datetime.strptime(config_data['START_DATE'], "%Y%m%d"),
            jdk_path=self._get_jdk_path(),
            jar_path=self._get_jar_path()
        )
    
    def get_tables_list(self, section: str = "transcation") -> list:
        """Get list of tables to process"""
        config_data = self.get_section_config(section)
        return config_data['TABLES'].split(",")
    
    def _get_host_from_config(self, config_data: dict) -> str:
        """Extract host configuration"""
        return "--application.server.host=ws://olympus-high-volume-api-icg-isg-olympus-high-volume-api-167969.apps.namicggtd34d.ecs.dyn.nsroot.net"
    
    def _get_jdk_path(self) -> str:
        """Get JDK path based on OS"""
        if os.name == 'nt':  # Windows
            return r"C:/Users/AK06306/AppData/Local/CitiSoftware/CTC2174129_JDK_17.0_15W64/bin/java.exe"
        else:  # Linux
            return "/opt/jdk/17.0_9l64/bin/java"
    
    def _get_jar_path(self) -> str:
        """Get JAR path based on OS"""
        if os.name == 'nt':  # Windows
            return r"C:/Users/AK06306/Downloads/mktdata-report-hvapi-commandline-client-1.0.30-20250519.150013-4.jar"
        else:  # Linux
            return "/home/bj33244/mktdata-report-hvapi-commandline-client-1.0.27-SNAPSHOT.jar"
    
    def _decrypt_password(self) -> str:
        """Decrypt password for Linux systems"""
        if os.name == 'nt':
            return ""  # Handle Windows authentication differently
        
        encrypted_file = os.environ.get('ENCRYPTED_PASS_FILE')
        keyvalue_file = os.environ.get('KEY_VALUE_FILE')
        
        if not encrypted_file or not keyvalue_file:
            raise ValueError("Password decryption environment variables not set")
        
        proc = subprocess.Popen([
            "/usr/local/bin/openssl", "enc", "-aes-128-cbc", "-pbkdf2", 
            "-a", "-d", "-in", encrypted_file, "-pass", f"file:{keyvalue_file}"
        ], stdout=subprocess.PIPE)
        
        return io.TextIOWrapper(proc.stdout, encoding="utf-8").readline().strip()


# =============================================================================
# DATABASE CLIENT
# =============================================================================

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
                self.system_config.jdk_path,
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
            result = subprocess.run(command, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                self.logger.error(f"Query execution failed: {result.stderr}")
                return False
            
            return os.path.exists(output_file)
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Query execution timed out: {query}")
            return False
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            return False
    
    def get_table_schema(self, table_name: str, schema_file: str) -> Dict[str, str]:
        """Get table schema information"""
        if os.path.exists(schema_file):
            return self._parse_schema_file(schema_file)
        
        query = f"describe {table_name}"
        if self.execute_query(query, schema_file):
            return self._parse_schema_file(schema_file)
        
        return {}
    
    def _parse_schema_file(self, schema_file: str) -> Dict[str, str]:
        """Parse schema file and return column type mappings"""
        schema_dict = {}
        
        with open(schema_file, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            if i == 0:  # Skip header
                continue
                
            parts = line.split(',', 1)
            if len(parts) >= 2:
                column_name = parts[0].strip().lower()
                column_type = parts[1].strip().rstrip(',')
                mapped_type = self._map_column_type(column_type)
                schema_dict[column_name] = f"{column_type}|{mapped_type}"
        
        return schema_dict
    
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


# =============================================================================
# METADATA MANAGER
# =============================================================================

class MetadataManager:
    """Manages table metadata and column statistics"""
    
    def __init__(self, db_client: DatabaseClient, system_config: SystemConfig):
        self.db_client = db_client
        self.system_config = system_config
        self.table_schemas = {}
        self.column_stats = {}
        self.logger = logging.getLogger(__name__)
    
    def load_table_metadata(self, table_name: str) -> bool:
        """Load metadata for a specific table"""
        table_dir = os.path.join(
            self.system_config.base_directory, 
            self.system_config.date_type, 
            "table", 
            table_name
        )
        os.makedirs(table_dir, exist_ok=True)
        
        schema_file = os.path.join(table_dir, f"{table_name}.schema")
        schema = self.db_client.get_table_schema(table_name, schema_file)
        
        if schema:
            self.table_schemas[table_name] = schema
            self.logger.info(f"Loaded metadata for table: {table_name}")
            return True
        
        self.logger.error(f"Failed to load metadata for table: {table_name}")
        return False
    
    def get_column_distinct_count(self, table_name: str, column_name: str, 
                                config_data: dict) -> Optional[str]:
        """Get distinct count for a specific column"""
        # Create query based on column type and date range
        where_clause_col = config_data[f"{table_name}_WhereCol_daycriteria"].lower()
        no_of_days = int(config_data[f"{table_name}_NoOfDays"])
        
        # Build date range
        n_days_ago = self.system_config.start_date - timedelta(days=no_of_days)
        where_conditions = self._build_date_conditions(
            table_name, where_clause_col, n_days_ago, self.system_config.start_date
        )
        
        # Determine query type based on column type
        column_info = self.table_schemas[table_name][column_name]
        mapped_type = column_info.split("|")[1]
        
        if mapped_type == "nestedtype":
            query = f"select count({column_name}) from {table_name} where {where_conditions}"
        else:
            query = f"select count(distinct {column_name}) from {table_name} where {where_conditions}"
        
        # Execute query
        output_file = os.path.join(
            self.system_config.base_directory,
            self.system_config.date_type,
            "table",
            table_name,
            f"{column_name}.json"
        )
        
        if self.db_client.execute_query(query, output_file):
            return self._parse_count_result(output_file)
        
        return None
    
    def get_distinct_values(self, table_name: str, column_name: str, 
                          config_data: dict) -> List[str]:
        """Get distinct values for a specific column"""
        where_clause_col = config_data[f"{table_name}_WhereCol_daycriteria"].lower()
        no_of_days = int(config_data[f"{table_name}_NoOfDays"])
        
        n_days_ago = self.system_config.start_date - timedelta(days=no_of_days)
        where_conditions = self._build_date_conditions(
            table_name, where_clause_col, n_days_ago, self.system_config.start_date
        )
        
        column_info = self.table_schemas[table_name][column_name]
        mapped_type = column_info.split("|")[1]
        
        if mapped_type == "nestedtype":
            query = f"select {column_name} from {table_name} where {where_conditions}"
        else:
            query = f"select distinct {column_name} from {table_name} where {where_conditions}"
        
        output_dir = os.path.join(self.system_config.output_directory, table_name)
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"{column_name}_distinctvalue.json")
        
        if self.db_client.execute_query(query, output_file):
            return self._parse_distinct_values(output_file)
        
        return []
    
    def _build_date_conditions(self, table_name: str, where_col: str, 
                             start_date: datetime, end_date: datetime) -> str:
        """Build date condition clause based on column type"""
        column_info = self.table_schemas[table_name][where_col]
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
                    if value:
                        values.append(value)
        except Exception as e:
            self.logger.error(f"Error parsing distinct values: {e}")
        return values


# =============================================================================
# RULES ENGINE
# =============================================================================

class Decision(Enum):
    COLUMNS_NEEDBA_EVAL = "COLUMNS_NEEDBA_EVAL"
    COLUMNS_NOTREJALLPHASE_NEEDBA_EVAL = "COLUMNS_NOTREJALLPHASE_NEEDBA_EVAL"
    COLUMNS_REJECTED = "COLUMNS_REJECTED"

@dataclass
class RuleDecision:
    enum: Decision
    reject_reason: str

class RulesEngine:
    """Applies business rules to determine column processing decisions"""
    
    def __init__(self, rules_config_path: str):
        self.rules_config = self._load_rules_config(rules_config_path)
        self.logger = logging.getLogger(__name__)
    
    def _load_rules_config(self, config_path: str) -> Dict[str, Any]:
        """Load rules configuration from JSON file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def evaluate_column(self, table_name: str, column_name: str, 
                       distinct_count: int, column_type: str) -> RuleDecision:
        """Evaluate a column against business rules"""
        
        # Rule 1: Check distinct count thresholds
        if distinct_count <= 100:
            return RuleDecision(
                enum=Decision.COLUMNS_NEEDBA_EVAL,
                reject_reason=""
            )
        elif distinct_count <= 10000:
            return RuleDecision(
                enum=Decision.COLUMNS_NOTREJALLPHASE_NEEDBA_EVAL,
                reject_reason=""
            )
        else:
            return RuleDecision(
                enum=Decision.COLUMNS_REJECTED,
                reject_reason=f"Too many distinct values: {distinct_count}"
            )
    
    def apply_rules(self, master_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all rules to the master dictionary"""
        decisions = {}
        
        column_stats = master_dict.get('COLUMN_DISTINCT_VALUES', {})
        metadata = master_dict.get('META_DATA', {})
        
        for table_name, columns in column_stats.items():
            decisions[table_name] = {}
            
            for column_name, distinct_count_str in columns.items():
                try:
                    distinct_count = int(distinct_count_str)
                    column_info = metadata.get(table_name, {}).get(column_name, "string|string")
                    column_type = column_info.split("|")[1]
                    
                    decision = self.evaluate_column(
                        table_name, column_name, distinct_count, column_type
                    )
                    decisions[table_name][column_name] = decision
                    
                except (ValueError, KeyError) as e:
                    self.logger.error(f"Error processing column {table_name}.{column_name}: {e}")
                    decisions[table_name][column_name] = RuleDecision(
                        enum=Decision.COLUMNS_REJECTED,
                        reject_reason=f"Processing error: {e}"
                    )
        
        return decisions


# =============================================================================
# OUTPUT GENERATOR
# =============================================================================

class OutputGenerator:
    """Generates final JSON output based on processed data and rules"""
    
    def __init__(self, system_config: SystemConfig):
        self.system_config = system_config
        self.logger = logging.getLogger(__name__)
    
    def generate_final_json(self, master_dict: Dict[str, Any], 
                          tables_list: List[str], config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the final JSON structure"""
        
        final_json = []
        final_json_enum100 = []
        final_json_enum10000 = []
        
        for table_name in tables_list:
            table_dict = self._create_table_structure(table_name, master_dict, config_data)
            if table_dict['main']:
                final_json.append(table_dict['main'])
            
            if table_dict['enum100']:
                final_json_enum100.append(table_dict['enum100'])
            if table_dict['enum10000']:
                final_json_enum10000.append(table_dict['enum10000'])
        
        return {
            'FINAL_JSON': final_json,
            'FINAL_JSON_ENUM100': final_json_enum100,
            'FINAL_JSON_ENUM10000': final_json_enum10000
        }
    
    def _create_table_structure(self, table_name: str, master_dict: Dict[str, Any], 
                               config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create table structure for JSON output"""
        
        # Get columns for this table
        table_columns_key = f"{table_name}_Columns"
        columns_str = config_data.get(table_columns_key, "")
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
            column_structs = self._create_column_structures(
                table_name, column_name, master_dict
            )
            
            if column_structs['main']:
                main_table['columns'].append(column_structs['main'])
            if column_structs['enum100']:
                enum100_table['columns'].append(column_structs['enum100'])
            if column_structs['enum10000']:
                enum10000_table['columns'].append(column_structs['enum10000'])
        
        return {
            'main': main_table,
            'enum100': enum100_table if enum100_table['columns'] else None,
            'enum10000': enum10000_table if enum10000_table['columns'] else None
        }
    
    def _create_column_structures(self, table_name: str, column_name: str, 
                                master_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Create column structures for different output types"""
        
        # Get column metadata
        metadata = master_dict.get('META_DATA', {})
        column_stats = master_dict.get('COLUMN_DISTINCT_VALUES', {})
        decisions = master_dict.get('DISTINCT_VALUE_DECISION', {})
        distinct_values = master_dict.get('DISTINCT_VALUES', {})
        
        # Base column structure
        base_column = {
            "columnname": column_name,
            "columnDescription": "",
            "columnSpecificRules": "",
            "columnAlias": [],
            "update_date": "",
            "update_soeid": ""
        }
        
        # Add metadata if available
        if table_name in metadata and column_name in metadata[table_name]:
            meta_data = metadata[table_name][column_name].split("|")
            base_column["actual_col_type"] = meta_data[0]
            base_column["mapped_col_type"] = meta_data[1]
        
        # Add distinct count
        if table_name in column_stats and column_name in column_stats[table_name]:
            base_column["distinct_count"] = column_stats[table_name][column_name]
        
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
        
        # Add distinct values if available
        base_column["distinct_values"] = []
        base_column["distinct_value_map"] = {}
        
        if (table_name in distinct_values and column_name in distinct_values[table_name]):
            values = [v for v in distinct_values[table_name][column_name] if v]
            base_column["distinct_values"] = values
            if decision and decision.enum == Decision.COLUMNS_NEEDBA_EVAL:
                base_column["distinct_value_map"] = {v: "" for v in values}
        
        # Create different versions
        result = {'main': copy.deepcopy(base_column), 'enum100': None, 'enum10000': None}
        
        if decision:
            if decision.enum == Decision.COLUMNS_NEEDBA_EVAL:
                enum100_col = copy.deepcopy(base_column)
                for key in ['provide_distinct', 'RejectReason', 'mapped_col_type']:
                    enum100_col.pop(key, None)
                result['enum100'] = enum100_col
                
            elif decision.enum == Decision.COLUMNS_NOTREJALLPHASE_NEEDBA_EVAL:
                enum10000_col = copy.deepcopy(base_column)
                for key in ['provide_distinct', 'RejectReason', 'mapped_col_type']:
                    enum10000_col.pop(key, None)
                result['enum10000'] = enum10000_col
        
        return result
    
    def save_outputs(self, final_data: Dict[str, Any]) -> bool:
        """Save all output files"""
        try:
            output_dir = self.system_config.output_directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Save main output
            main_output = {
                "trainingdata": final_data['FINAL_JSON']
            }
            
            main_file = os.path.join(output_dir, "all_final_output.json")
            with open(main_file, 'w') as f:
                json.dump(main_output, f, indent=4)
            
            self.logger.info(f"Saved main output to: {main_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving outputs: {e}")
            return False


# =============================================================================
# MAIN PROCESSOR
# =============================================================================

class TrainingDataProcessor:
    """Main processor that orchestrates the entire training data selection workflow"""
    
    def __init__(self, config_files: List[str]):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        try:
            self.config_manager = ConfigManager(config_files)
            self.db_config = self.config_manager.get_database_config()
            self.system_config = self.config_manager.get_system_config()
            self.db_client = DatabaseClient(self.db_config, self.system_config)
            self.metadata_manager = MetadataManager(self.db_client, self.system_config)
            
            # Load rules engine
            rules_path = os.path.join(os.path.dirname(__file__), 'rules.json')
            self.rules_engine = RulesEngine(rules_path)
            self.output_generator = OutputGenerator(self.system_config)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize processor: {e}")
            raise
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training_data_processor.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def process(self, section: str = "transcation") -> bool:
        """Main processing workflow"""
        try:
            self.logger.info("Starting training data processing workflow")
            
            # Get configuration data
            config_data = self.config_manager.get_section_config(section)
            tables_list = self.config_manager.get_tables_list(section)
            
            # Initialize master dictionary
            master_dict = {
                'APPLICATION_CONFIG': config_data,
                'META_DATA': {},
                'COLUMN_DISTINCT_VALUES': {},
                'DISTINCT_VALUE_DECISION': {},
                'DISTINCT_VALUES': {},
                'WHERE_CLAUSE': {}
            }
            
            # Step 1: Load metadata for all tables
            self.logger.info("Loading table metadata...")
            for table_name in tables_list:
                if not self.metadata_manager.load_table_metadata(table_name):
                    self.logger.warning(f"Failed to load metadata for table: {table_name}")
                    continue
            
            master_dict['META_DATA'] = self.metadata_manager.table_schemas
            
            # Step 2: Set number of days for processing
            self.logger.info("Setting processing day ranges...")
            for table_name in tables_list:
                config_data[f"{table_name}_NoOfDays"] = 90  # Default to 90 days
            
            # Step 3: Process columns and get statistics
            self.logger.info("Processing column statistics...")
            for table_name in tables_list:
                columns_key = f"{table_name}_Columns"
                columns_value = config_data.get(columns_key, "")
                
                # Handle auto-generated columns
                if columns_value == "[generated]":
                    try:
                        column_list = getFrequentColumn(
                            table_name, 
                            self.db_config.host, 
                            self.db_config.user_id, 
                            self.db_config.password
                        )
                        config_data[columns_key] = ",".join(column_list)
                        self.logger.info(f"Generated columns for {table_name}: {column_list}")
                    except Exception as e:
                        self.logger.error(f"Failed to generate columns for {table_name}: {e}")
                        continue
                else:
                    column_list = [col.strip().lower() for col in columns_value.split(",")]
                
                # Get distinct counts for each column
                for column_name in column_list:
                    distinct_count = self.metadata_manager.get_column_distinct_count(
                        table_name, column_name, config_data
                    )
                    
                    if distinct_count is not None:
                        if table_name not in master_dict['COLUMN_DISTINCT_VALUES']:
                            master_dict['COLUMN_DISTINCT_VALUES'][table_name] = {}
                        master_dict['COLUMN_DISTINCT_VALUES'][table_name][column_name] = distinct_count
                        self.logger.info(f"Column {table_name}.{column_name}: {distinct_count} distinct values")
            
            # Step 4: Apply business rules
            self.logger.info("Applying business rules...")
            decisions = self.rules_engine.apply_rules(master_dict)
            master_dict['DISTINCT_VALUE_DECISION'] = decisions
            
            # Step 5: Get distinct values for approved columns
            self.logger.info("Fetching distinct values for approved columns...")
            for table_name, columns in decisions.items():
                for column_name, decision in columns.items():
                    if decision.enum in [Decision.COLUMNS_NEEDBA_EVAL, Decision.COLUMNS_NOTREJALLPHASE_NEEDBA_EVAL]:
                        distinct_values = self.metadata_manager.get_distinct_values(
                            table_name, column_name, config_data
                        )
                        
                        if table_name not in master_dict['DISTINCT_VALUES']:
                            master_dict['DISTINCT_VALUES'][table_name] = {}
                        master_dict['DISTINCT_VALUES'][table_name][column_name] = distinct_values
                        
                        self.logger.info(f"Retrieved {len(distinct_values)} distinct values for {table_name}.{column_name}")
            
            # Step 6: Generate final JSON output
            self.logger.info("Generating final JSON output...")
            final_data = self.output_generator.generate_final_json(master_dict, tables_list, config_data)
            
            # Step 7: Save outputs
            if self.output_generator.save_outputs(final_data):
                self.logger.info("Training data processing completed successfully")
                return True
            else:
                self.logger.error("Failed to save outputs")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in processing workflow: {e}")
            return False


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point"""
    # Configuration files (adjust paths as needed)
    config_files = [
        r"C:/Users/AK06306/Downloads/rsocket_workspace/olympus-dc-server/data_preprocessing/Tables_Selection_Middleoffice.conf",
        r"C:/Users/AK06306/Downloads/rsocket_workspace/olympus-dc-server/data_preprocessing/Tables_Selection.conf"
    ]
    
    try:
        processor = TrainingDataProcessor(config_files)
        success = processor.process()
        
        if success:
            print("Training data processing completed successfully!")
            return 0
        else:
            print("Training data processing failed!")
            return 1
            
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())