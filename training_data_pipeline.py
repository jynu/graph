# training_data_pipeline.py
"""
Training Data Processing Pipeline
=================================
Main orchestrator script that executes the complete training data processing workflow:

1. Get Frequent Columns (1_get_frequent_columns.py)
2. Get Distinct Values (2_get_distinct_values.py) 
3. Load to ORASS (3_load_to_orass.py)

This script can run the complete pipeline or individual steps as needed.
"""

import os
import sys
import logging
import argparse
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Optional

# =============================================================================
# CONFIGURATION SETTINGS - Modify these as needed
# =============================================================================

# Module Paths
MODULE_1_PATH = "1_get_frequent_columns.py"
MODULE_2_PATH = "2_get_distinct_values.py"
MODULE_3_PATH = "3_load_to_orass.py"

# Default Settings
DEFAULT_USER_ID = ""  # Set your default user ID
DEFAULT_PASSWORD = ""  # Set your default password
DEFAULT_ENVIRONMENT = "dev"
DEFAULT_OUTPUT_DIR = "./output"
DEFAULT_CONFIG_DIR = "./configs"

# Pipeline Settings
PIPELINE_LOG_FILE = "training_data_pipeline.log"
STEP_TIMEOUT = 3600  # 1 hour timeout per step

# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

class TrainingDataPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, user_id: str, password: str, environment: str = DEFAULT_ENVIRONMENT):
        self.user_id = user_id
        self.password = password
        self.environment = environment
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Pipeline state
        self.step_results = {}
        self.current_step = 0
        self.total_steps = 3
        
    def setup_logging(self):
        """Setup pipeline logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(PIPELINE_LOG_FILE),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def run_complete_pipeline(self, table_names: List[str], output_dir: str = DEFAULT_OUTPUT_DIR,
                            config_dir: str = DEFAULT_CONFIG_DIR) -> bool:
        """Run the complete 3-step pipeline"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("TRAINING DATA PROCESSING PIPELINE STARTED")
            self.logger.info("=" * 80)
            self.logger.info(f"Tables to process: {table_names}")
            self.logger.info(f"Output directory: {output_dir}")
            self.logger.info(f"Config directory: {config_dir}")
            self.logger.info(f"Environment: {self.environment}")
            self.logger.info(f"User: {self.user_id}")
            
            # Create directories
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(config_dir, exist_ok=True)
            
            # Step 1: Get Frequent Columns
            self.current_step = 1
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"STEP {self.current_step}/{self.total_steps}: GETTING FREQUENT COLUMNS")
            self.logger.info(f"{'='*60}")
            
            config_files = self.run_step_1(table_names, config_dir)
            if not config_files:
                self.logger.error("Step 1 failed: No config files generated")
                return False
            
            self.step_results['step_1'] = {
                'status': 'success',
                'config_files': config_files,
                'timestamp': datetime.now()
            }
            self.logger.info(f"✅ Step 1 completed: {len(config_files)} config files generated")
            
            # Step 2: Get Distinct Values
            self.current_step = 2
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"STEP {self.current_step}/{self.total_steps}: GETTING DISTINCT VALUES")
            self.logger.info(f"{'='*60}")
            
            json_files = []
            for config_file in config_files:
                json_file = self.run_step_2(config_file, output_dir)
                if json_file:
                    json_files.extend(json_file.values()) if isinstance(json_file, dict) else json_files.append(json_file)
            
            if not json_files:
                self.logger.error("Step 2 failed: No JSON files generated")
                return False
            
            self.step_results['step_2'] = {
                'status': 'success',
                'json_files': json_files,
                'timestamp': datetime.now()
            }
            self.logger.info(f"✅ Step 2 completed: {len(json_files)} JSON files generated")
            
            # Step 3: Load to ORASS
            self.current_step = 3
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"STEP {self.current_step}/{self.total_steps}: LOADING TO ORASS")
            self.logger.info(f"{'='*60}")
            
            # Load main JSON file (all_final_output.json)
            main_json_file = None
            for json_file in json_files:
                if 'all_final_output.json' in json_file:
                    main_json_file = json_file
                    break
            
            if not main_json_file:
                self.logger.error("Step 3 failed: Main JSON file (all_final_output.json) not found")
                return False
            
            success = self.run_step_3(main_json_file)
            if not success:
                self.logger.error("Step 3 failed: Data loading to ORASS failed")
                return False
            
            self.step_results['step_3'] = {
                'status': 'success',
                'loaded_file': main_json_file,
                'timestamp': datetime.now()
            }
            self.logger.info(f"✅ Step 3 completed: Data loaded to ORASS")
            
            # Pipeline completion
            self.logger.info(f"\n{'='*80}")
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info(f"{'='*80}")
            self._log_pipeline_summary()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed at step {self.current_step}: {e}")
            return False
    
    def run_step_1(self, table_names: List[str], config_dir: str) -> List[str]:
        """Run Step 1: Get Frequent Columns"""
        try:
            config_files = []
            
            for table_name in table_names:
                self.logger.info(f"Processing table: {table_name}")
                
                # Build command
                cmd = [
                    sys.executable, MODULE_1_PATH,
                    "--table", table_name,
                    "--user", self.user_id,
                    "--password", self.password,
                    "--output-dir", config_dir
                ]
                
                # Execute command
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=STEP_TIMEOUT)
                
                if result.returncode == 0:
                    # Find generated config file
                    table_safe_name = table_name.replace('.', '_')
                    config_file = os.path.join(config_dir, f"tbc_{table_safe_name}.conf")
                    
                    if os.path.exists(config_file):
                        config_files.append(config_file)
                        self.logger.info(f"✅ Config generated for {table_name}: {config_file}")
                    else:
                        self.logger.warning(f"Config file not found for {table_name}")
                else:
                    self.logger.error(f"Failed to process {table_name}: {result.stderr}")
            
            return config_files
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Step 1 timed out after {STEP_TIMEOUT} seconds")
            return []
        except Exception as e:
            self.logger.error(f"Error in step 1: {e}")
            return []
    
    def run_step_2(self, config_file: str, output_dir: str) -> Optional[Dict[str, str]]:
        """Run Step 2: Get Distinct Values"""
        try:
            self.logger.info(f"Processing config file: {config_file}")
            
            # Build command
            cmd = [
                sys.executable, MODULE_2_PATH,
                config_file,
                "--output-dir", output_dir
            ]
            
            # Execute command
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=STEP_TIMEOUT)
            
            if result.returncode == 0:
                # Find generated JSON files
                json_files = {}
                for filename in ["all_final_output.json", "enum100_output.json", "enum10000_output.json"]:
                    json_path = os.path.join(output_dir, filename)
                    if os.path.exists(json_path):
                        json_files[filename] = json_path
                
                if json_files:
                    self.logger.info(f"✅ JSON files generated: {list(json_files.keys())}")
                    return json_files
                else:
                    self.logger.warning(f"No JSON files found in {output_dir}")
                    return None
            else:
                self.logger.error(f"Failed to process {config_file}: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Step 2 timed out after {STEP_TIMEOUT} seconds")
            return None
        except Exception as e:
            self.logger.error(f"Error in step 2: {e}")
            return None
    
    def run_step_3(self, json_file: str) -> bool:
        """Run Step 3: Load to ORASS"""
        try:
            self.logger.info(f"Loading JSON file to ORASS: {json_file}")
            
            # Build command
            cmd = [
                sys.executable, MODULE_3_PATH,
                json_file,
                "--environment", self.environment
            ]
            
            # Execute command
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=STEP_TIMEOUT)
            
            if result.returncode == 0:
                self.logger.info(f"✅ Data loaded to ORASS successfully")
                return True
            else:
                self.logger.error(f"Failed to load to ORASS: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Step 3 timed out after {STEP_TIMEOUT} seconds")
            return False
        except Exception as e:
            self.logger.error(f"Error in step 3: {e}")
            return False
    
    def run_individual_step(self, step_number: int, **kwargs) -> bool:
        """Run an individual step"""
        try:
            if step_number == 1:
                table_names = kwargs.get('table_names', [])
                config_dir = kwargs.get('config_dir', DEFAULT_CONFIG_DIR)
                result = self.run_step_1(table_names, config_dir)
                return len(result) > 0
                
            elif step_number == 2:
                config_file = kwargs.get('config_file', '')
                output_dir = kwargs.get('output_dir', DEFAULT_OUTPUT_DIR)
                if not config_file:
                    self.logger.error("Config file required for step 2")
                    return False
                result = self.run_step_2(config_file, output_dir)
                return result is not None
                
            elif step_number == 3:
                json_file = kwargs.get('json_file', '')
                if not json_file:
                    self.logger.error("JSON file required for step 3")
                    return False
                return self.run_step_3(json_file)
                
            else:
                self.logger.error(f"Invalid step number: {step_number}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error running step {step_number}: {e}")
            return False
    
    def _log_pipeline_summary(self):
        """Log pipeline execution summary"""
        self.logger.info("\nPIPELINE EXECUTION SUMMARY:")
        self.logger.info("-" * 40)
        
        for step_name, step_data in self.step_results.items():
            step_num = step_name.replace('step_', '')
            status = step_data['status']
            timestamp = step_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            
            self.logger.info(f"Step {step_num}: {status.upper()} at {timestamp}")
            
            if step_name == 'step_1':
                config_files = step_data.get('config_files', [])
                self.logger.info(f"  Generated {len(config_files)} config files")
                for config_file in config_files:
                    self.logger.info(f"    - {config_file}")
                    
            elif step_name == 'step_2':
                json_files = step_data.get('json_files', [])
                self.logger.info(f"  Generated {len(json_files)} JSON files")
                for json_file in json_files:
                    file_size = os.path.getsize(json_file) if os.path.exists(json_file) else 0
                    self.logger.info(f"    - {json_file} ({file_size:,} bytes)")
                    
            elif step_name == 'step_3':
                loaded_file = step_data.get('loaded_file', '')
                self.logger.info(f"  Loaded file: {loaded_file}")
        
        self.logger.info("-" * 40)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_module_files() -> bool:
    """Validate that all required module files exist"""
    required_modules = [MODULE_1_PATH, MODULE_2_PATH, MODULE_3_PATH]
    missing_modules = []
    
    for module in required_modules:
        if not os.path.exists(module):
            missing_modules.append(module)
    
    if missing_modules:
        print(f"Error: Missing required module files:")
        for module in missing_modules:
            print(f"  - {module}")
        return False
    
    return True

def parse_table_list(table_input: str) -> List[str]:
    """Parse table names from comma-separated string"""
    if not table_input:
        return []
    
    # Split by comma and clean whitespace
    tables = [table.strip() for table in table_input.split(',')]
    
    # Remove empty strings
    tables = [table for table in tables if table]
    
    return tables

def get_user_credentials() -> tuple:
    """Get user credentials interactively if not provided"""
    import getpass
    
    user_id = input("Enter User ID: ").strip()
    if not user_id:
        raise ValueError("User ID is required")
    
    password = getpass.getpass("Enter Password: ").strip()
    if not password:
        raise ValueError("Password is required")
    
    return user_id, password

def preview_pipeline_execution(table_names: List[str], output_dir: str, config_dir: str, environment: str):
    """Preview what the pipeline will do without executing"""
    print("=" * 60)
    print("PIPELINE EXECUTION PREVIEW")
    print("=" * 60)
    
    print(f"Tables to process ({len(table_names)}):")
    for i, table in enumerate(table_names, 1):
        print(f"  {i}. {table}")
    
    print(f"\nDirectories:")
    print(f"  Config directory: {config_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Environment: {environment}")
    
    print(f"\nPipeline steps:")
    print(f"  Step 1: Generate {len(table_names)} config file(s)")
    for table in table_names:
        table_safe = table.replace('.', '_')
        config_file = os.path.join(config_dir, f"tbc_{table_safe}.conf")
        print(f"    - {config_file}")
    
    print(f"  Step 2: Process config files to generate JSON")
    print(f"    - {os.path.join(output_dir, 'all_final_output.json')}")
    print(f"    - {os.path.join(output_dir, 'enum100_output.json')} (if applicable)")
    print(f"    - {os.path.join(output_dir, 'enum10000_output.json')} (if applicable)")
    
    print(f"  Step 3: Load main JSON to ORASS ({environment} environment)")
    
    print("=" * 60)

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Training Data Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline for one table
  python training_data_pipeline.py --tables "GFOLYMKT_MANAGED_ALL.OM_EQ_QUOTE_MARKET_DATA" --user myuser --password mypass

  # Run complete pipeline for multiple tables
  python training_data_pipeline.py --tables "table1,table2,table3" --user myuser --password mypass --env uat

  # Run only step 1 (get frequent columns)
  python training_data_pipeline.py --step 1 --tables "table1" --user myuser --password mypass

  # Run only step 2 (get distinct values)
  python training_data_pipeline.py --step 2 --config-file "./configs/tbc_table1.conf"

  # Run only step 3 (load to ORASS)
  python training_data_pipeline.py --step 3 --json-file "./output/all_final_output.json" --env dev
        """
    )
    
    # Main arguments
    parser.add_argument('--tables', '-t', type=str, help='Comma-separated list of table names')
    parser.add_argument('--user', '-u', type=str, help='Database user ID')
    parser.add_argument('--password', '-p', type=str, help='Database password')
    parser.add_argument('--environment', '--env', '-e', type=str, default=DEFAULT_ENVIRONMENT,
                       choices=['dev', 'uat', 'prod'], help='Target environment')
    
    # Step control
    parser.add_argument('--step', '-s', type=int, choices=[1, 2, 3], 
                       help='Run specific step only (1=frequent columns, 2=distinct values, 3=load ORASS)')
    
    # Step-specific arguments
    parser.add_argument('--config-file', type=str, help='Config file for step 2')
    parser.add_argument('--json-file', type=str, help='JSON file for step 3')
    
    # Directory settings
    parser.add_argument('--output-dir', '-o', type=str, default=DEFAULT_OUTPUT_DIR,
                       help='Output directory for JSON files')
    parser.add_argument('--config-dir', '-c', type=str, default=DEFAULT_CONFIG_DIR,
                       help='Config directory for .conf files')
    
    # Control flags
    parser.add_argument('--preview', action='store_true', help='Preview execution without running')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode for credentials')
    parser.add_argument('--validate-modules', action='store_true', help='Validate module files only')
    
    args = parser.parse_args()
    
    # Validate module files
    if args.validate_modules or not validate_module_files():
        return 1 if not validate_module_files() else 0
    
    try:
        # Get credentials
        if args.interactive:
            user_id, password = get_user_credentials()
        else:
            user_id = args.user or DEFAULT_USER_ID
            password = args.password or DEFAULT_PASSWORD
            
            if not user_id or not password:
                print("Error: User ID and password are required. Use --interactive or provide --user and --password")
                return 1
        
        # Parse table names for steps that need them
        table_names = []
        if args.tables:
            table_names = parse_table_list(args.tables)
        
        # Preview mode
        if args.preview:
            if not table_names and not args.step:
                print("Error: Table names required for preview")
                return 1
            
            if not args.step:  # Complete pipeline preview
                preview_pipeline_execution(table_names, args.output_dir, args.config_dir, args.environment)
            else:
                print(f"Preview for Step {args.step} only")
                if args.step == 1 and table_names:
                    print(f"Will process {len(table_names)} table(s): {', '.join(table_names)}")
                elif args.step == 2 and args.config_file:
                    print(f"Will process config file: {args.config_file}")
                elif args.step == 3 and args.json_file:
                    print(f"Will load JSON file: {args.json_file}")
            return 0
        
        # Initialize pipeline
        pipeline = TrainingDataPipeline(user_id, password, args.environment)
        
        # Run specific step or complete pipeline
        if args.step:
            print(f"Running Step {args.step} only...")
            
            if args.step == 1:
                if not table_names:
                    print("Error: Table names required for step 1")
                    return 1
                success = pipeline.run_individual_step(1, table_names=table_names, config_dir=args.config_dir)
                
            elif args.step == 2:
                if not args.config_file:
                    print("Error: Config file required for step 2")
                    return 1
                success = pipeline.run_individual_step(2, config_file=args.config_file, output_dir=args.output_dir)
                
            elif args.step == 3:
                if not args.json_file:
                    print("Error: JSON file required for step 3")
                    return 1
                success = pipeline.run_individual_step(3, json_file=args.json_file)
            
            if success:
                print(f"✅ Step {args.step} completed successfully!")
                return 0
            else:
                print(f"❌ Step {args.step} failed!")
                return 1
        
        else:
            # Run complete pipeline
            if not table_names:
                print("Error: Table names required for complete pipeline")
                return 1
            
            print("Running complete pipeline...")
            success = pipeline.run_complete_pipeline(table_names, args.output_dir, args.config_dir)
            
            if success:
                print("✅ Complete pipeline executed successfully!")
                return 0
            else:
                print("❌ Pipeline execution failed!")
                return 1
    
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 