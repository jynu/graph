# 3_load_to_orass.py
"""
Core Module 3: Load to ORASS
=============================
Input: JSON file from Step 2 (all_final_output.json)
Output: Data loaded into ORASS database

This module loads the processed training data JSON into the ORASS database
for further analysis and business usage.
"""

import os
import sys
import json
import logging
import getpass
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
import oracledb

# =============================================================================
# CONFIGURATION SETTINGS - Modify these as needed
# =============================================================================

# ORASS Database Connection Settings
# Development Environment
DEV_ORASS_DSN = "OLY2DEV.oraas.dyn.nsroot.net:8889/haOLY2DEV"
DEV_ORASS_USERNAME = "OLYMMQA"
DEV_ORASS_PASSWORD = "NaYm0Y3D"

# UAT Environment  
UAT_ORASS_DSN = "OLY2UAT.oraas.dyn.nsroot.net:8889/haOLY2UAT"
UAT_ORASS_USERNAME = "GFOLYMQA"
UAT_ORASS_PASSWORD = "ou70cJ5z"

# Production Environment (add as needed)
PROD_ORASS_DSN = ""
PROD_ORASS_USERNAME = ""
PROD_ORASS_PASSWORD = ""

# Default Environment
DEFAULT_ENVIRONMENT = "dev"

# Database Table Names
TABLE_METADATA_TABLE = "TABLE_METADATA"
COLUMN_METADATA_TABLE = "COLUMN_METADATA" 
TABLE_AKA_TABLE = "TABLE_AKA"
COLUMN_AKA_TABLE = "COLUMN_AKA"
DISTINCT_VALUE_TABLE = "DISTINCT_VALUE_MAPPING"

# Processing Settings
BATCH_SIZE = 1000  # Number of records to process in one batch
MAX_DESCRIPTION_LENGTH = 255  # Maximum length for description fields
MAX_VALUE_LENGTH = 255  # Maximum length for value fields
DEFAULT_EDITOR = "system"  # Default editor name

# File Settings
DEFAULT_JSON_FILE = "all_final_output.json"
BACKUP_DIRECTORY = "./backups"

# =============================================================================
# DATABASE CONNECTION MANAGER
# =============================================================================

class ORASSDatabaseManager:
    """Manages ORASS database connections and operations"""
    
    def __init__(self, environment: str = DEFAULT_ENVIRONMENT):
        self.environment = environment.lower()
        self.connection = None
        self.cursor = None
        self.logger = logging.getLogger(__name__)
        
        # Set connection parameters based on environment
        self._set_connection_params()
    
    def _set_connection_params(self):
        """Set database connection parameters based on environment"""
        if self.environment == "dev":
            self.dsn = DEV_ORASS_DSN
            self.username = DEV_ORASS_USERNAME
            self.password = DEV_ORASS_PASSWORD
        elif self.environment == "uat":
            self.dsn = UAT_ORASS_DSN
            self.username = UAT_ORASS_USERNAME
            self.password = UAT_ORASS_PASSWORD
        elif self.environment == "prod":
            self.dsn = PROD_ORASS_DSN
            self.username = PROD_ORASS_USERNAME
            self.password = PROD_ORASS_PASSWORD
        else:
            raise ValueError(f"Unknown environment: {self.environment}")
        
        self.logger.info(f"Database configuration set for environment: {self.environment}")
    
    def connect(self) -> bool:
        """Establish database connection"""
        try:
            self.logger.info(f"Connecting to ORASS database ({self.environment})...")
            self.connection = oracledb.connect(
                user=self.username, 
                password=self.password, 
                dsn=self.dsn
            )
            self.cursor = self.connection.cursor()
            self.logger.info("✅ Database connection established successfully")
            return True
            
        except oracledb.DatabaseError as e:
            error, = e.args
            self.logger.error(f"Database connection failed: {error.code} - {error.message}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected connection error: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            self.logger.info("Database connection closed")
        except Exception as e:
            self.logger.error(f"Error closing database connection: {e}")
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            if not self.connection:
                return False
            
            self.cursor.execute("SELECT 1 FROM DUAL")
            result = self.cursor.fetchone()
            return result is not None
            
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def execute_query(self, query: str, params: tuple = None) -> bool:
        """Execute a query with parameters"""
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            return True
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            self.logger.error(f"Query: {query}")
            if params:
                self.logger.error(f"Parameters: {params}")
            return False
    
    def commit(self):
        """Commit transaction"""
        try:
            self.connection.commit()
        except Exception as e:
            self.logger.error(f"Commit failed: {e}")
            raise
    
    def rollback(self):
        """Rollback transaction"""
        try:
            self.connection.rollback()
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")

# =============================================================================
# DATA PROCESSING UTILITIES
# =============================================================================

def clean_text(text: str, max_length: int = MAX_DESCRIPTION_LENGTH) -> str:
    """Clean and truncate text for database storage"""
    if not text:
        return ""
    
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', str(text)).strip()
    
    # Truncate if necessary
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
    
    return cleaned

def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to integer"""
    try:
        if value is None or value == '':
            return default
        return int(value)
    except (ValueError, TypeError):
        return default

def safe_date(date_str: str) -> Optional[datetime]:
    """Safely parse date string"""
    if not date_str:
        return None
    
    try:
        # Try different date formats
        for fmt in ['%Y-%m-%d', '%Y%m%d', '%Y-%m-%d %H:%M:%S']:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None
    except Exception:
        return None

# =============================================================================
# ORASS DATA LOADER
# =============================================================================

class ORASSSDataLoader:
    """Loads training data into ORASS database"""
    
    def __init__(self, db_manager: ORASSDatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self.editor = getpass.getuser() if getpass.getuser() else DEFAULT_EDITOR
    
    def load_json_file(self, json_file_path: str) -> bool:
        """Load JSON file and process all tables"""
        try:
            self.logger.info(f"Loading JSON file: {json_file_path}")
            
            if not os.path.exists(json_file_path):
                raise FileNotFoundError(f"JSON file not found: {json_file_path}")
            
            with open(json_file_path, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)
            
            # Extract training data
            training_data = data.get("trainingdata", [])
            if not training_data:
                self.logger.warning("No training data found in JSON file")
                return False
            
            self.logger.info(f"Found {len(training_data)} tables to process")
            
            # Process each table
            success_count = 0
            for table_data in training_data:
                try:
                    if self.process_table(table_data):
                        success_count += 1
                    else:
                        self.logger.warning(f"Failed to process table: {table_data.get('tablename', 'Unknown')}")
                except Exception as e:
                    self.logger.error(f"Error processing table {table_data.get('tablename', 'Unknown')}: {e}")
                    continue
            
            self.logger.info(f"Successfully processed {success_count}/{len(training_data)} tables")
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Error loading JSON file: {e}")
            return False
    
    def process_table(self, table_data: Dict[str, Any]) -> bool:
        """Process a single table and its metadata"""
        try:
            table_name = table_data["tablename"]
            self.logger.info(f"Processing table: {table_name}")
            
            # Parse table name
            schema_name, simple_table_name = self._parse_table_name(table_name)
            
            # Insert/update table metadata
            table_description = clean_text(table_data.get("tableDescription", ""))
            table_rules = clean_text(table_data.get("tableSpecificRules", ""))
            
            table_id = self.insert_table_metadata(
                schema_name, simple_table_name, table_description, table_rules
            )
            
            if not table_id:
                self.logger.error(f"Failed to insert table metadata for: {table_name}")
                return False
            
            # Process table aliases
            table_aliases = table_data.get("tableAlias", [])
            self.process_table_aliases(table_id, table_aliases)
            
            # Process columns
            columns = table_data.get("columns", [])
            self.logger.info(f"Processing {len(columns)} columns for table: {table_name}")
            
            for column_data in columns:
                try:
                    self.process_column(table_id, column_data)
                except Exception as e:
                    self.logger.error(f"Error processing column {column_data.get('columnname', 'Unknown')}: {e}")
                    continue
            
            # Commit table processing
            self.db_manager.commit()
            self.logger.info(f"✅ Successfully processed table: {table_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing table: {e}")
            self.db_manager.rollback()
            return False
    
    def process_column(self, table_id: int, column_data: Dict[str, Any]) -> bool:
        """Process a single column and its metadata"""
        try:
            column_name = column_data["columnname"].lower()
            description = clean_text(column_data.get("columnDescription", ""))
            column_rules = clean_text(column_data.get("columnSpecificRules", ""))
            
            # Column type information
            actual_col_type = column_data.get("actual_col_type", "string")
            mapped_col_type = column_data.get("mapped_col_type", "string")
            
            # Statistics
            distinct_count = safe_int(column_data.get("distinct_count", 0))
            provide_distinct = column_data.get("provide_distinct", "NO")
            reject_reason = clean_text(column_data.get("RejectReason", ""))
            
            # Audit information
            last_editor = column_data.get("update_soeid", self.editor)
            update_date_str = column_data.get("update_date", "")
            last_time = safe_date(update_date_str) if update_date_str else datetime.now()
            
            # Insert column metadata
            column_id = self.insert_column_metadata(
                table_id, column_name, description, actual_col_type, mapped_col_type,
                distinct_count, provide_distinct, last_editor, last_time, reject_reason, 
                "", column_rules
            )
            
            if not column_id:
                self.logger.error(f"Failed to insert column metadata for: {column_name}")
                return False
            
            # Process column aliases
            column_aliases = column_data.get("columnAlias", [])
            self.process_column_aliases(column_id, column_aliases)
            
            # Process distinct values
            distinct_values = column_data.get("distinct_values", [])
            distinct_value_map = column_data.get("distinct_value_map", {})
            
            if distinct_values:
                self.process_distinct_values(column_id, distinct_values, distinct_value_map)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing column: {e}")
            return False
    
    def _parse_table_name(self, table_name: str) -> tuple:
        """Parse table name into schema and table components"""
        if '.' in table_name:
            parts = table_name.split('.')
            return parts[0], parts[1]
        else:
            return "", table_name
    
    def insert_table_metadata(self, schema_name: str, table_name: str, 
                            description: str, rules: str) -> Optional[int]:
        """Insert or update table metadata"""
        try:
            # Check if table exists
            check_query = f"""
                SELECT table_id FROM {TABLE_METADATA_TABLE} 
                WHERE schema_name = :1 AND table_name = :2
            """
            
            self.db_manager.execute_query(check_query, (schema_name, table_name))
            result = self.db_manager.cursor.fetchone()
            
            if result:
                # Update existing table
                table_id = result[0]
                update_query = f"""
                    UPDATE {TABLE_METADATA_TABLE} 
                    SET description = :1, rules = :2, last_updated = :3
                    WHERE table_id = :4
                """
                self.db_manager.execute_query(
                    update_query, (description, rules, datetime.now(), table_id)
                )
                self.logger.debug(f"Updated existing table: {schema_name}.{table_name}")
            else:
                # Insert new table
                insert_query = f"""
                    INSERT INTO {TABLE_METADATA_TABLE} 
                    (schema_name, table_name, description, rules, created_date, last_updated)
                    VALUES (:1, :2, :3, :4, :5, :6)
                    RETURNING table_id INTO :7
                """
                
                table_id_var = self.db_manager.cursor.var(oracledb.NUMBER)
                self.db_manager.execute_query(
                    insert_query, 
                    (schema_name, table_name, description, rules, datetime.now(), datetime.now(), table_id_var)
                )
                table_id = int(table_id_var.getvalue()[0])
                self.logger.debug(f"Inserted new table: {schema_name}.{table_name} (ID: {table_id})")
            
            return table_id
            
        except Exception as e:
            self.logger.error(f"Error inserting table metadata: {e}")
            return None
    
    def insert_column_metadata(self, table_id: int, column_name: str, description: str,
                             format1: str, format2: str, count: int, flag: str,
                             last_editor: str, last_time: datetime, reason: str,
                             exclude: str, column_rules: str) -> Optional[int]:
        """Insert or update column metadata"""
        try:
            # Check if column exists
            check_query = f"""
                SELECT column_id FROM {COLUMN_METADATA_TABLE}
                WHERE table_id = :1 AND column_name = :2
            """
            
            self.db_manager.execute_query(check_query, (table_id, column_name))
            result = self.db_manager.cursor.fetchone()
            
            if result:
                # Update existing column
                column_id = result[0]
                update_query = f"""
                    UPDATE {COLUMN_METADATA_TABLE}
                    SET description = :1, format1 = :2, format2 = :3, count = :4,
                        flag = :5, last_editor = :6, last_time = :7, reason = :8,
                        exclude = :9, column_rules = :10
                    WHERE column_id = :11
                """
                self.db_manager.execute_query(
                    update_query, 
                    (description, format1, format2, count, flag, last_editor, 
                     last_time, reason, exclude, column_rules, column_id)
                )
                self.logger.debug(f"Updated existing column: {column_name}")
            else:
                # Insert new column
                insert_query = f"""
                    INSERT INTO {COLUMN_METADATA_TABLE}
                    (table_id, column_name, description, format1, format2, count,
                     flag, last_editor, last_time, reason, exclude, column_rules)
                    VALUES (:1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12)
                    RETURNING column_id INTO :13
                """
                
                column_id_var = self.db_manager.cursor.var(oracledb.NUMBER)
                self.db_manager.execute_query(
                    insert_query,
                    (table_id, column_name, description, format1, format2, count,
                     flag, last_editor, last_time, reason, exclude, column_rules, column_id_var)
                )
                column_id = int(column_id_var.getvalue()[0])
                self.logger.debug(f"Inserted new column: {column_name} (ID: {column_id})")
            
            return column_id
            
        except Exception as e:
            self.logger.error(f"Error inserting column metadata: {e}")
            return None
    
    def process_table_aliases(self, table_id: int, aliases: List[str]):
        """Process table aliases"""
        try:
            # Delete existing aliases
            self.drop_table_aliases(table_id)
            
            # Insert new aliases
            for alias in aliases:
                if alias and alias.strip():
                    self.insert_table_alias(table_id, alias.strip())
            
        except Exception as e:
            self.logger.error(f"Error processing table aliases: {e}")
    
    def process_column_aliases(self, column_id: int, aliases: List[str]):
        """Process column aliases"""
        try:
            # Delete existing aliases
            self.drop_column_aliases(column_id)
            
            # Insert new aliases
            for alias in aliases:
                if alias and alias.strip():
                    self.insert_column_alias(column_id, alias.strip())
            
        except Exception as e:
            self.logger.error(f"Error processing column aliases: {e}")
    
    def process_distinct_values(self, column_id: int, values: List[str], 
                              value_map: Dict[str, str]):
        """Process distinct values and their mappings"""
        try:
            # Delete existing distinct values
            self.drop_distinct_values(column_id)
            
            # Insert new distinct values
            for value in values:
                if value:
                    cleaned_value = clean_text(value, MAX_VALUE_LENGTH)
                    description = clean_text(value_map.get(value, ""), MAX_DESCRIPTION_LENGTH)
                    self.insert_distinct_value(column_id, cleaned_value, description)
            
        except Exception as e:
            self.logger.error(f"Error processing distinct values: {e}")
    
    def insert_table_alias(self, table_id: int, alias: str) -> bool:
        """Insert table alias"""
        try:
            insert_query = f"""
                INSERT INTO {TABLE_AKA_TABLE} (table_id, alias)
                VALUES (:1, :2)
            """
            return self.db_manager.execute_query(insert_query, (table_id, alias))
        except Exception as e:
            self.logger.error(f"Error inserting table alias: {e}")
            return False
    
    def insert_column_alias(self, column_id: int, alias: str) -> bool:
        """Insert column alias"""
        try:
            insert_query = f"""
                INSERT INTO {COLUMN_AKA_TABLE} (column_id, alias)
                VALUES (:1, :2)
            """
            return self.db_manager.execute_query(insert_query, (column_id, alias))
        except Exception as e:
            self.logger.error(f"Error inserting column alias: {e}")
            return False
    
    def insert_distinct_value(self, column_id: int, value: str, description: str) -> bool:
        """Insert distinct value mapping"""
        try:
            insert_query = f"""
                INSERT INTO {DISTINCT_VALUE_TABLE} (column_id, value, description)
                VALUES (:1, :2, :3)
            """
            return self.db_manager.execute_query(insert_query, (column_id, value, description))
        except Exception as e:
            self.logger.error(f"Error inserting distinct value: {e}")
            return False
    
    def drop_table_aliases(self, table_id: int) -> bool:
        """Delete existing table aliases"""
        try:
            delete_query = f"DELETE FROM {TABLE_AKA_TABLE} WHERE table_id = :1"
            return self.db_manager.execute_query(delete_query, (table_id,))
        except Exception as e:
            self.logger.error(f"Error dropping table aliases: {e}")
            return False
    
    def drop_column_aliases(self, column_id: int) -> bool:
        """Delete existing column aliases"""
        try:
            delete_query = f"DELETE FROM {COLUMN_AKA_TABLE} WHERE column_id = :1"
            return self.db_manager.execute_query(delete_query, (column_id,))
        except Exception as e:
            self.logger.error(f"Error dropping column aliases: {e}")
            return False
    
    def drop_distinct_values(self, column_id: int) -> bool:
        """Delete existing distinct value mappings"""
        try:
            delete_query = f"DELETE FROM {DISTINCT_VALUE_TABLE} WHERE column_id = :1"
            return self.db_manager.execute_query(delete_query, (column_id,))
        except Exception as e:
            self.logger.error(f"Error dropping distinct values: {e}")
            return False

# =============================================================================
# BACKUP AND VALIDATION UTILITIES
# =============================================================================

class DataValidator:
    """Validates data before loading into ORASS"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_json_file(self, json_file_path: str) -> bool:
        """Validate JSON file structure and content"""
        try:
            self.logger.info(f"Validating JSON file: {json_file_path}")
            
            if not os.path.exists(json_file_path):
                self.logger.error(f"JSON file not found: {json_file_path}")
                return False
            
            # Check file size
            file_size = os.path.getsize(json_file_path)
            if file_size == 0:
                self.logger.error("JSON file is empty")
                return False
            
            self.logger.info(f"File size: {file_size:,} bytes")
            
            # Parse JSON
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate structure
            if 'trainingdata' not in data:
                self.logger.error("Missing 'trainingdata' key in JSON")
                return False
            
            training_data = data['trainingdata']
            if not isinstance(training_data, list):
                self.logger.error("'trainingdata' should be a list")
                return False
            
            if len(training_data) == 0:
                self.logger.warning("No training data found")
                return False
            
            # Validate each table
            valid_tables = 0
            for i, table_data in enumerate(training_data):
                if self.validate_table_data(table_data, i):
                    valid_tables += 1
            
            self.logger.info(f"Validation completed: {valid_tables}/{len(training_data)} tables are valid")
            return valid_tables > 0
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON format: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return False
    
    def validate_table_data(self, table_data: Dict[str, Any], index: int) -> bool:
        """Validate individual table data"""
        try:
            # Required fields
            required_fields = ['tablename', 'columns']
            for field in required_fields:
                if field not in table_data:
                    self.logger.error(f"Table {index}: Missing required field '{field}'")
                    return False
            
            table_name = table_data['tablename']
            if not table_name or not isinstance(table_name, str):
                self.logger.error(f"Table {index}: Invalid table name")
                return False
            
            # Validate columns
            columns = table_data['columns']
            if not isinstance(columns, list):
                self.logger.error(f"Table {table_name}: Columns should be a list")
                return False
            
            if len(columns) == 0:
                self.logger.warning(f"Table {table_name}: No columns found")
            
            # Validate each column
            valid_columns = 0
            for column_data in columns:
                if self.validate_column_data(column_data, table_name):
                    valid_columns += 1
            
            self.logger.debug(f"Table {table_name}: {valid_columns}/{len(columns)} columns are valid")
            return valid_columns > 0
            
        except Exception as e:
            self.logger.error(f"Error validating table {index}: {e}")
            return False
    
    def validate_column_data(self, column_data: Dict[str, Any], table_name: str) -> bool:
        """Validate individual column data"""
        try:
            if 'columnname' not in column_data:
                self.logger.error(f"Table {table_name}: Column missing 'columnname'")
                return False
            
            column_name = column_data['columnname']
            if not column_name or not isinstance(column_name, str):
                self.logger.error(f"Table {table_name}: Invalid column name")
                return False
            
            # Check distinct count if present
            if 'distinct_count' in column_data:
                try:
                    distinct_count = int(column_data['distinct_count'])
                    if distinct_count < 0:
                        self.logger.warning(f"Table {table_name}, Column {column_name}: Negative distinct count")
                except (ValueError, TypeError):
                    self.logger.warning(f"Table {table_name}, Column {column_name}: Invalid distinct count format")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating column in table {table_name}: {e}")
            return False

class BackupManager:
    """Manages data backups before loading"""
    
    def __init__(self, backup_dir: str = BACKUP_DIRECTORY):
        self.backup_dir = backup_dir
        self.logger = logging.getLogger(__name__)
        os.makedirs(backup_dir, exist_ok=True)
    
    def create_backup(self, json_file_path: str) -> str:
        """Create backup of JSON file before processing"""
        try:
            if not os.path.exists(json_file_path):
                raise FileNotFoundError(f"Source file not found: {json_file_path}")
            
            # Generate backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.basename(json_file_path)
            name, ext = os.path.splitext(filename)
            backup_filename = f"{name}_backup_{timestamp}{ext}"
            backup_path = os.path.join(self.backup_dir, backup_filename)
            
            # Copy file
            import shutil
            shutil.copy2(json_file_path, backup_path)
            
            self.logger.info(f"Backup created: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            raise

# =============================================================================
# MAIN PROCESSING CLASS
# =============================================================================

class ORASSSLoader:
    """Main class for loading training data into ORASS"""
    
    def __init__(self, environment: str = DEFAULT_ENVIRONMENT):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.environment = environment
        
        # Initialize components
        self.db_manager = ORASSDatabaseManager(environment)
        self.data_loader = None
        self.validator = DataValidator()
        self.backup_manager = BackupManager()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('orass_loader.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def load_json_to_orass(self, json_file_path: str, create_backup: bool = True,
                          validate_first: bool = True) -> bool:
        """Main method to load JSON data into ORASS"""
        try:
            self.logger.info("=" * 60)
            self.logger.info("ORASS DATA LOADING PROCESS STARTED")
            self.logger.info("=" * 60)
            self.logger.info(f"Environment: {self.environment}")
            self.logger.info(f"JSON file: {json_file_path}")
            
            # Step 1: Validate JSON file
            if validate_first:
                self.logger.info("Step 1: Validating JSON file...")
                if not self.validator.validate_json_file(json_file_path):
                    self.logger.error("JSON validation failed. Aborting load process.")
                    return False
                self.logger.info("✅ JSON validation passed")
            
            # Step 2: Create backup
            if create_backup:
                self.logger.info("Step 2: Creating backup...")
                backup_path = self.backup_manager.create_backup(json_file_path)
                self.logger.info(f"✅ Backup created: {backup_path}")
            
            # Step 3: Connect to database
            self.logger.info("Step 3: Connecting to ORASS database...")
            if not self.db_manager.connect():
                self.logger.error("Database connection failed. Aborting load process.")
                return False
            
            # Test connection
            if not self.db_manager.test_connection():
                self.logger.error("Database connection test failed. Aborting load process.")
                return False
            
            self.logger.info("✅ Database connection established")
            
            # Step 4: Initialize data loader
            self.data_loader = ORASSSDataLoader(self.db_manager)
            
            # Step 5: Load data
            self.logger.info("Step 4: Loading data into ORASS...")
            success = self.data_loader.load_json_file(json_file_path)
            
            if success:
                self.logger.info("✅ Data loading completed successfully!")
                return True
            else:
                self.logger.error("❌ Data loading failed!")
                return False
            
        except Exception as e:
            self.logger.error(f"Fatal error during loading process: {e}")
            return False
        finally:
            # Always close database connection
            if self.db_manager:
                self.db_manager.disconnect()
    
    def get_loading_summary(self, json_file_path: str) -> Dict[str, Any]:
        """Get summary of what would be loaded without actually loading"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            training_data = data.get('trainingdata', [])
            
            summary = {
                'total_tables': len(training_data),
                'total_columns': 0,
                'total_distinct_values': 0,
                'tables': []
            }
            
            for table_data in training_data:
                table_name = table_data.get('tablename', 'Unknown')
                columns = table_data.get('columns', [])
                
                table_summary = {
                    'name': table_name,
                    'column_count': len(columns),
                    'distinct_value_count': 0
                }
                
                for column_data in columns:
                    distinct_values = column_data.get('distinct_values', [])
                    table_summary['distinct_value_count'] += len(distinct_values)
                
                summary['total_columns'] += table_summary['column_count']
                summary['total_distinct_values'] += table_summary['distinct_value_count']
                summary['tables'].append(table_summary)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return {}

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Load training data JSON into ORASS database')
    parser.add_argument('json_file', type=str, help='JSON file path (all_final_output.json)')
    parser.add_argument('--environment', '-e', type=str, default=DEFAULT_ENVIRONMENT,
                       choices=['dev', 'uat', 'prod'], help='Target environment')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup creation')
    parser.add_argument('--no-validation', action='store_true', help='Skip JSON validation')
    parser.add_argument('--summary', '-s', action='store_true', help='Show loading summary only')
    parser.add_argument('--validate-only', '-v', action='store_true', help='Validate JSON file only')
    
    args = parser.parse_args()
    
    # Validate JSON file exists
    if not os.path.exists(args.json_file):
        print(f"Error: JSON file not found: {args.json_file}")
        return 1
    
    try:
        # Initialize loader
        loader = ORASSSLoader(args.environment)
        
        # Handle different modes
        if args.validate_only:
            print("Validating JSON file...")
            validator = DataValidator()
            if validator.validate_json_file(args.json_file):
                print("✅ JSON validation passed!")
                return 0
            else:
                print("❌ JSON validation failed!")
                return 1
        
        elif args.summary:
            print("Generating loading summary...")
            summary = loader.get_loading_summary(args.json_file)
            
            print(f"\nLoading Summary:")
            print(f"Total tables: {summary.get('total_tables', 0)}")
            print(f"Total columns: {summary.get('total_columns', 0)}")
            print(f"Total distinct values: {summary.get('total_distinct_values', 0)}")
            
            print(f"\nTable breakdown:")
            for table in summary.get('tables', []):
                print(f"  {table['name']}: {table['column_count']} columns, {table['distinct_value_count']} distinct values")
            
            return 0
        
        else:
            # Full loading process
            print(f"Loading JSON data into ORASS ({args.environment} environment)...")
            success = loader.load_json_to_orass(
                args.json_file,
                create_backup=not args.no_backup,
                validate_first=not args.no_validation
            )
            
            if success:
                print("✅ Data loading completed successfully!")
                return 0
            else:
                print("❌ Data loading failed!")
                return 1
        
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())