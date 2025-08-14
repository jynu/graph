# training_data_processor.py
"""
Training Data Processor
=======================
Main orchestrator script that combines frequent column discovery and distinct value extraction
to generate and save training data for text-to-SQL models.

This script:
1. Discovers frequent columns for a specified table
2. Extracts distinct values for those columns
3. Calls the saveTable API to save the training data

Prerequisites:
- get_frequent_columns_enc_v2.py (in same directory)
- get_distinct_values.py (rename from "get_distinct_values (1).py" if needed)
- Required dependencies: requests, configparser, etc.

Usage:
    python training_data_processor.py --table TABLE_NAME --user USER_ID [options]
"""

import os
import sys
import json
import logging
import argparse
import requests
import tempfile
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Import classes and functions from the existing modules
try:
    # From get_frequent_columns_enc_v2.py
    from get_frequent_columns_enc_v2 import (
        FrequentColumnProcessor,
        validate_environment,
        CredentialManager,
        DatabaseClient as FreqColumnDatabaseClient,
        QueryLogAnalyzer,
        ConfigGenerator
    )
    
    # From get_distinct_values.py (note the filename has spaces and parentheses)
    from get_distinct_values import (
        DistinctValueProcessor,
        validate_config_file,
        DatabaseClient as DistinctValueDatabaseClient,
        MetadataManager,
        RulesEngine,
        JSONGenerator,
        Decision,
        RuleDecision
    )
    
    print("✅ Successfully imported required modules")
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Please ensure both scripts are in the same directory:")
    print("  - get_frequent_columns_enc_v2.py")
    print("  - get_distinct_values.py (note: filename contains spaces and parentheses)")
    print("You may need to rename 'get_distinct_values (1).py' to 'get_distinct_values.py'")
    sys.exit(1)

# =============================================================================
# CONFIGURATION SETTINGS
# =============================================================================

# API Configuration
DEFAULT_API_BASE_URL = "https://mktdata-ai-training-service-icg-isg-olympusmqa-167969.apps.namicggtd35d.ecs.dyn.nsroot.net"
SAVE_TABLE_ENDPOINT = "/api/intelligenceTraining/saveTable"

# Default paths and settings
DEFAULT_CONFIG_DIR = "./configs"
DEFAULT_OUTPUT_DIR = "./output"
DEFAULT_TEMP_DIR = "./temp"

# =============================================================================
# CORE CLASSES
# =============================================================================

@dataclass
class ProcessingConfig:
    """Configuration for the training data processing"""
    table_name: str
    user_id: str
    password: str = None
    encrypted_file: str = None
    key_file: str = None
    api_base_url: str = DEFAULT_API_BASE_URL
    config_dir: str = DEFAULT_CONFIG_DIR
    output_dir: str = DEFAULT_OUTPUT_DIR
    temp_dir: str = DEFAULT_TEMP_DIR
    table_id: str = None
    schema_name: str = None
    updated_by: str = None
    version: str = "1"
    table_description: str = ""
    table_specific_rules: str = ""
    table_alias: List[str] = None

class APIClient:
    """Handles API communication for saving training data"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.logger = logging.getLogger(__name__)
        
    def save_table_data(self, table_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save table training data via API"""
        try:
            url = f"{self.base_url}{SAVE_TABLE_ENDPOINT}"
            
            self.logger.info(f"Calling saveTable API: {url}")
            self.logger.debug(f"Request payload keys: {list(table_data.keys())}")
            
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            response = requests.post(url, json=table_data, headers=headers, timeout=30)
            
            self.logger.info(f"API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                response_data = response.json()
                self.logger.info(f"✅ Table data saved successfully")
                self.logger.info(f"   - Table ID: {response_data.get('tableId', 'N/A')}")
                self.logger.info(f"   - Version: {response_data.get('version', 'N/A')}")
                return response_data
            else:
                error_msg = f"API call failed with status {response.status_code}: {response.text}"
                self.logger.error(error_msg)
                raise Exception(error_msg)
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error calling API: {e}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Error calling saveTable API: {e}"
            self.logger.error(error_msg)
            raise

class DataTransformer:
    """Transforms JSON output to API format"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def transform_to_api_format(self, json_data: Dict[str, Any], 
                              config: ProcessingConfig) -> Dict[str, Any]:
        """Transform JSON output to saveTable API format"""
        try:
            # Extract training data from JSON
            training_data = json_data.get('trainingdata', [])
            
            if not training_data:
                raise ValueError("No training data found in JSON output")
            
            # Get the first table (assuming single table processing)
            source_table = training_data[0]
            
            # Transform columns to API format
            api_columns = []
            for col in source_table.get('columns', []):
                api_column = {
                    "tableColumnId": None,  # Will be assigned by API
                    "columnName": col.get('columnname', ''),
                    "columnDescription": col.get('columnDescription', ''),
                    "columnAlias": col.get('columnAlias', []),
                    "rejectReason": col.get('RejectReason'),
                    "actual_col_type": col.get('actual_col_type', ''),
                    "mapped_col_type": col.get('mapped_col_type', ''),
                    "distinct_count": col.get('distinct_count', 0),
                    "provide_distinct": col.get('provide_distinct', 'NO'),
                    "distinct_value_map": self._convert_distinct_values_to_map(col),
                    "enum": col.get('enum')
                }
                api_columns.append(api_column)
            
            # Build API payload
            api_payload = {
                "data": {
                    "trainingdata": [
                        {
                            "tableId": config.table_id,
                            "schemaName": config.schema_name or self._extract_schema_name(config.table_name),
                            "tableName": self._extract_table_name(config.table_name),
                            "tableDescription": config.table_description or source_table.get('tableDescription', ''),
                            "tableSpecificRules": config.table_specific_rules or source_table.get('tableSpecificRules', ''),
                            "updatedBy": config.updated_by or config.user_id,
                            "version": config.version,
                            "tableAlias": config.table_alias or source_table.get('tableAlias', []),
                            "columns": api_columns
                        }
                    ]
                }
            }
            
            self.logger.info(f"Transformed data for table: {config.table_name}")
            self.logger.info(f"   - Columns: {len(api_columns)}")
            self.logger.info(f"   - Schema: {config.schema_name or self._extract_schema_name(config.table_name)}")
            
            return api_payload
            
        except Exception as e:
            self.logger.error(f"Error transforming data to API format: {e}")
            raise
    
    def _convert_distinct_values_to_map(self, column: Dict[str, Any]) -> List[str]:
        """Convert distinct values to the format expected by API"""
        distinct_value_map = column.get('distinct_value_map', {})
        distinct_values = column.get('distinct_values', [])
        
        # If we have a map with descriptions, use it
        if distinct_value_map and isinstance(distinct_value_map, dict):
            return [f'"{key}": "{value}"' for key, value in distinct_value_map.items()]
        
        # Otherwise, create empty descriptions for distinct values
        if distinct_values:
            return [f'"{value}": ""' for value in distinct_values[:10]]  # Limit to 10 values
        
        return []
    
    def _extract_schema_name(self, full_table_name: str) -> str:
        """Extract schema name from full table name"""
        if '.' in full_table_name:
            return full_table_name.split('.')[0]
        return "default_schema"
    
    def _extract_table_name(self, full_table_name: str) -> str:
        """Extract table name from full table name"""
        if '.' in full_table_name:
            return full_table_name.split('.')[-1]
        return full_table_name

class TrainingDataProcessor:
    """Main orchestrator for the entire training data processing pipeline"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.api_client = APIClient(config.api_base_url)
        self.data_transformer = DataTransformer()
        
        # Create directories
        for directory in [config.config_dir, config.output_dir, config.temp_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training_data_processor.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def process_table(self) -> Dict[str, Any]:
        """Process a single table through the complete pipeline"""
        try:
            self.logger.info("=" * 60)
            self.logger.info("TRAINING DATA PROCESSING PIPELINE")
            self.logger.info("=" * 60)
            self.logger.info(f"Table: {self.config.table_name}")
            self.logger.info(f"User: {self.config.user_id}")
            self.logger.info(f"API URL: {self.config.api_base_url}")
            
            # Step 1: Discover frequent columns
            self.logger.info("\n🔍 STEP 1: Discovering frequent columns...")
            config_file_path = self._discover_frequent_columns()
            
            # Step 2: Extract distinct values
            self.logger.info("\n📊 STEP 2: Extracting distinct values...")
            json_output_files = self._extract_distinct_values(config_file_path)
            
            # Step 3: Transform and save data
            self.logger.info("\n💾 STEP 3: Transforming and saving data...")
            api_response = self._transform_and_save_data(json_output_files)
            
            # Step 4: Cleanup temporary files
            self.logger.info("\n🧹 STEP 4: Cleaning up temporary files...")
            self._cleanup_temp_files(config_file_path)
            
            self.logger.info("\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
            return api_response
            
        except Exception as e:
            self.logger.error(f"\n❌ PIPELINE FAILED: {e}")
            raise
    
    def _discover_frequent_columns(self) -> str:
        """Step 1: Discover frequent columns using the enhanced analyzer"""
        try:
            # Initialize frequent column processor
            processor = FrequentColumnProcessor(self.config.user_id, self.config.password)
            
            # Process the table
            config_file_path = processor.process_table(self.config.table_name)
            
            self.logger.info(f"✅ Frequent columns configuration generated: {config_file_path}")
            return config_file_path
            
        except Exception as e:
            self.logger.error(f"Error in frequent column discovery: {e}")
            raise
    
    def _extract_distinct_values(self, config_file_path: str) -> Dict[str, str]:
        """Step 2: Extract distinct values using the configuration file"""
        try:
            # Validate configuration file
            if not validate_config_file(config_file_path):
                raise Exception("Configuration file validation failed")
            
            # Initialize distinct value processor
            processor = DistinctValueProcessor()
            
            # Process configuration and extract distinct values
            output_files = processor.process_config(config_file_path, self.config.output_dir)
            
            self.logger.info(f"✅ Distinct values extracted successfully")
            for output_type, file_path in output_files.items():
                self.logger.info(f"   - {output_type}: {file_path}")
            
            return output_files
            
        except Exception as e:
            self.logger.error(f"Error in distinct value extraction: {e}")
            raise
    
    def _transform_and_save_data(self, json_output_files: Dict[str, str]) -> Dict[str, Any]:
        """Step 3: Transform JSON data and save via API"""
        try:
            # Use the main JSON output file
            main_json_file = json_output_files.get('main')
            if not main_json_file or not os.path.exists(main_json_file):
                raise Exception("Main JSON output file not found")
            
            # Load JSON data
            with open(main_json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            self.logger.info(f"Loaded JSON data from: {main_json_file}")
            
            # Transform to API format
            api_payload = self.data_transformer.transform_to_api_format(json_data, self.config)
            
            # Save payload for debugging
            debug_file = os.path.join(self.config.temp_dir, f"api_payload_{self.config.table_name.replace('.', '_')}.json")
            with open(debug_file, 'w', encoding='utf-8') as f:
                json.dump(api_payload, f, indent=2)
            self.logger.info(f"API payload saved for debugging: {debug_file}")
            
            # Call saveTable API
            api_response = self.api_client.save_table_data(api_payload)
            
            self.logger.info("✅ Data saved successfully via API")
            return api_response
            
        except Exception as e:
            self.logger.error(f"Error in data transformation and saving: {e}")
            raise
    
    def _cleanup_temp_files(self, config_file_path: str):
        """Step 4: Clean up temporary files (optional)"""
        try:
            # Optionally remove temporary config file
            # Uncomment the following lines if you want to clean up config files
            # if os.path.exists(config_file_path):
            #     os.remove(config_file_path)
            #     self.logger.info(f"Cleaned up config file: {config_file_path}")
            
            self.logger.info("✅ Cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Warning during cleanup: {e}")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_inputs(config: ProcessingConfig) -> bool:
    """Validate input parameters"""
    errors = []
    
    if not config.table_name:
        errors.append("Table name is required")
    
    if not config.user_id:
        errors.append("User ID is required")
    
    if not config.password and not (config.encrypted_file and config.key_file):
        errors.append("Either password or encrypted file + key file must be provided")
    
    if config.encrypted_file and not os.path.exists(config.encrypted_file):
        errors.append(f"Encrypted file not found: {config.encrypted_file}")
    
    if config.key_file and not os.path.exists(config.key_file):
        errors.append(f"Key file not found: {config.key_file}")
    
    if errors:
        for error in errors:
            print(f"❌ Validation Error: {error}")
        return False
    
    return True

def setup_environment(config: ProcessingConfig):
    """Setup environment variables for encrypted password files"""
    if config.encrypted_file:
        os.environ['ENCRYPTED_PASS_FILE'] = config.encrypted_file
    
    if config.key_file:
        os.environ['KEY_VALUE_FILE'] = config.key_file

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Process training data for text-to-SQL models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using encrypted password files
  python training_data_processor.py --table SCHEMA.TABLE_NAME --user USER123 --encrypted-file password.enc --key-file keyfile.key
  
  # Using direct password (not recommended for production)
  python training_data_processor.py --table SCHEMA.TABLE_NAME --user USER123 --password PASSWORD
  
  # With custom API URL and metadata
  python training_data_processor.py --table SCHEMA.TABLE_NAME --user USER123 --encrypted-file password.enc --key-file keyfile.key --api-url https://custom-api.com --table-id 123 --schema-name CUSTOM_SCHEMA
        """
    )
    
    # Required arguments
    parser.add_argument('--table', '-t', type=str, required=True,
                       help='Table name to process (e.g., SCHEMA.TABLE_NAME)')
    parser.add_argument('--user', '-u', type=str, required=True,
                       help='Database user ID')
    
    # Authentication arguments
    parser.add_argument('--password', '-p', type=str,
                       help='Database password (not recommended for production)')
    parser.add_argument('--encrypted-file', '-e', type=str,
                       help='Path to encrypted password file')
    parser.add_argument('--key-file', '-k', type=str,
                       help='Path to encryption key file')
    
    # API configuration
    parser.add_argument('--api-url', type=str, default=DEFAULT_API_BASE_URL,
                       help=f'API base URL (default: {DEFAULT_API_BASE_URL})')
    
    # Directory configuration
    parser.add_argument('--config-dir', type=str, default=DEFAULT_CONFIG_DIR,
                       help=f'Configuration files directory (default: {DEFAULT_CONFIG_DIR})')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                       help=f'Output files directory (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--temp-dir', type=str, default=DEFAULT_TEMP_DIR,
                       help=f'Temporary files directory (default: {DEFAULT_TEMP_DIR})')
    
    # Metadata arguments
    parser.add_argument('--table-id', type=str,
                       help='Table ID for API (optional)')
    parser.add_argument('--schema-name', type=str,
                       help='Schema name override (optional)')
    parser.add_argument('--updated-by', type=str,
                       help='Updated by user ID (defaults to user ID)')
    parser.add_argument('--version', type=str, default='1',
                       help='Version number (default: 1)')
    parser.add_argument('--table-description', type=str, default='',
                       help='Table description')
    parser.add_argument('--table-rules', type=str, default='',
                       help='Table specific rules')
    parser.add_argument('--table-alias', nargs='+',
                       help='Table aliases (space-separated)')
    
    # Processing options
    parser.add_argument('--disable-llm', action='store_true',
                       help='Disable LLM analysis for frequent columns')
    parser.add_argument('--dry-run', action='store_true',
                       help='Perform dry run without calling API')
    
    args = parser.parse_args()
    
    # Create processing configuration
    config = ProcessingConfig(
        table_name=args.table,
        user_id=args.user,
        password=args.password,
        encrypted_file=args.encrypted_file,
        key_file=args.key_file,
        api_base_url=args.api_url,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        temp_dir=args.temp_dir,
        table_id=args.table_id,
        schema_name=args.schema_name,
        updated_by=args.updated_by,
        version=args.version,
        table_description=args.table_description,
        table_specific_rules=args.table_rules,
        table_alias=args.table_alias or []
    )
    
    # Validate inputs
    if not validate_inputs(config):
        return 1
    
    # Setup environment
    setup_environment(config)
    
    # Validate environment
    print("🔍 Validating environment...")
    validate_environment()
    
    # Disable LLM if requested
    if args.disable_llm:
        import get_frequent_columns_enc_v2
        get_frequent_columns_enc_v2.ENABLE_LLM_ANALYSIS = False
        print("🚫 LLM analysis disabled")
    
    try:
        # Initialize processor
        processor = TrainingDataProcessor(config)
        
        if args.dry_run:
            print("🧪 DRY RUN MODE - API will not be called")
            # Perform steps 1 and 2 only
            config_file = processor._discover_frequent_columns()
            json_files = processor._extract_distinct_values(config_file)
            
            # Transform data but don't save
            main_json_file = json_files.get('main')
            if main_json_file and os.path.exists(main_json_file):
                with open(main_json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                api_payload = processor.data_transformer.transform_to_api_format(json_data, config)
                
                print("✅ Dry run completed successfully")
                print(f"   - Config file: {config_file}")
                print(f"   - JSON files: {len(json_files)}")
                print(f"   - API payload ready (not sent)")
            else:
                print("❌ Dry run failed - no JSON output generated")
                return 1
        else:
            # Full processing
            result = processor.process_table()
            
            print("🎉 Training data processing completed successfully!")
            print(f"   - Table: {config.table_name}")
            print(f"   - Table ID: {result.get('tableId', 'N/A')}")
            print(f"   - Version: {result.get('version', 'N/A')}")
            print(f"   - Status: {result.get('status', 'N/A')}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())