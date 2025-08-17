Required Changes for Existing Scripts
1. get_frequent_columns.py Changes:
python# At the top, add this import (around line 25):
from database_manager import (
    MultiDatabaseManager, DatabaseType, Environment, 
    get_database_client_from_string, EnhancedCredentialManager
)

# Replace the DatabaseClient class (lines ~200-300) with this:
class DatabaseClient:
    """Wrapper for the new multi-database client"""
    
    def __init__(self, db_config, system_config):
        # For backward compatibility, extract database info
        self.db_config = db_config
        self.system_config = system_config
        self.logger = logging.getLogger(__name__)
        
        # Default to Impala prod for backward compatibility
        self.db_type = DatabaseType.IMPALA
        self.environment = Environment.PROD
        
        # Create the new database client
        manager = MultiDatabaseManager()
        self.db_client = manager.get_database_client(
            self.db_type, self.environment, 
            db_config.user_id, db_config.password
        )
    
    def execute_query(self, query: str, output_file: str, format_type: str = "TXT") -> bool:
        """Execute query using new database manager"""
        return self.db_client.execute_query(query, output_file, format_type)
    
    def get_table_schema(self, table_name: str) -> Dict[str, str]:
        """Get table schema using new database manager"""
        return self.db_client.get_table_schema(table_name)

# In FrequentColumnProcessor.__init__ method (around line 800), replace credential handling:
def __init__(self, user_id: str, password: str, database_spec: str = "impala:prod"):
    self.setup_logging()
    self.logger = logging.getLogger(__name__)
    
    # Parse database specification
    if ':' in database_spec:
        db_parts = database_spec.split(':')
        self.db_type = DatabaseType(db_parts[0].lower())
        self.environment = Environment(db_parts[1].lower())
    else:
        # Default to Impala prod
        self.db_type = DatabaseType.IMPALA
        self.environment = Environment.PROD
    
    # Get credentials using enhanced credential manager
    cred_manager = EnhancedCredentialManager()
    secure_user, secure_password = cred_manager.get_credentials(
        self.db_type, self.environment, user_id, password
    )
    
    # Create database client using new manager
    manager = MultiDatabaseManager()
    self.db_client = manager.get_database_client(
        self.db_type, self.environment, secure_user, secure_password
    )
    
    # Keep rest of initialization the same...
2. get_distinct_values.py Changes:
python# At the top, add this import:
from database_manager import (
    MultiDatabaseManager, DatabaseType, Environment,
    get_database_client_from_string, EnhancedCredentialManager
)

# Replace the DatabaseClient class (lines ~150-250) with:
class DatabaseClient:
    """Wrapper for the new multi-database client"""
    
    def __init__(self, db_config, system_config):
        self.db_config = db_config
        self.system_config = system_config
        self.logger = logging.getLogger(__name__)
        
        # Extract database type from db_config or default to Impala
        self.db_type = getattr(db_config, 'db_type', DatabaseType.IMPALA)
        self.environment = getattr(db_config, 'environment', Environment.PROD)
        
        # Create the new database client
        manager = MultiDatabaseManager()
        self.db_client = manager.get_database_client(
            self.db_type, self.environment,
            db_config.user_id, db_config.password
        )
    
    def execute_query(self, query: str, output_file: str, format_type: str = "TXT") -> bool:
        return self.db_client.execute_query(query, output_file, format_type)
    
    def get_table_schema(self, table_name: str) -> Dict[str, str]:
        return self.db_client.get_table_schema(table_name)

# In DistinctValueProcessor.load_config_file method (around line 450), update database client creation:
def load_config_file(self, config_file_path: str) -> bool:
    """Load configuration file generated in Step 1"""
    try:
        # ... existing config loading code ...
        
        # Extract database specification from config if available
        database_spec = self.config_data.get('DATABASE_SPEC', 'impala:prod')
        
        # Parse database specification
        if ':' in database_spec:
            db_parts = database_spec.split(':')
            db_type = DatabaseType(db_parts[0].lower())
            environment = Environment(db_parts[1].lower())
        else:
            db_type = DatabaseType.IMPALA
            environment = Environment.PROD
        
        # Get credentials
        cred_manager = EnhancedCredentialManager()
        user_id, password = cred_manager.get_credentials(
            db_type, environment,
            self.config_data.get('USERID', ''),
            self.config_data.get('PASSWORD', '')
        )
        
        # Create database client using new manager
        manager = MultiDatabaseManager()
        self.db_client = manager.get_database_client(db_type, environment, user_id, password)
        
        # ... rest of method stays the same ...
3. training_data_processor.py Changes:
python# At the top, add this import:
from database_manager import (
    MultiDatabaseManager, DatabaseType, Environment,
    get_database_client_from_string, validate_database_specification
)

# Update ProcessingConfig dataclass (around line 50):
@dataclass
class ProcessingConfig:
    """Configuration for the training data processing"""
    table_name: str
    user_id: str
    password: str = None
    database_spec: str = "impala:prod"  # Add this field
    encrypted_file: str = None
    key_file: str = None
    # ... rest of fields stay the same ...

# In main() function, add database specification argument:
def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(...)
    
    # Add database specification argument
    parser.add_argument('--database', '-db', type=str, default='impala:prod',
                       help='Database specification (format: type:environment, e.g., impala:prod, oracle:dev)')
    
    # ... existing arguments ...
    
    args = parser.parse_args()
    
    # Validate database specification
    if not validate_database_specification(args.database):
        print(f"Invalid database specification: {args.database}")
        print("Format should be: type:environment (e.g., impala:prod, oracle:dev)")
        return 1
    
    # Create processing configuration
    config = ProcessingConfig(
        table_name=args.table,
        user_id=args.user,
        password=args.password,
        database_spec=args.database,  # Add this
        # ... rest of config ...
    )

# Update TrainingDataProcessor.__init__ to pass database spec to modules:
def __init__(self, config: ProcessingConfig):
    # ... existing initialization ...
    
    # Store database specification for passing to other modules
    self.database_spec = config.database_spec
    
# Update _discover_frequent_columns method:
def _discover_frequent_columns(self) -> str:
    """Step 1: Discover frequent columns using the enhanced analyzer"""
    try:
        # Initialize frequent column processor with database specification
        processor = FrequentColumnProcessor(
            self.config.user_id, 
            self.config.password,
            self.database_spec  # Pass database spec
        )
        
        # ... rest stays the same ...
Usage Examples:

Environment-specific Credentials (Optional):
Create environment-specific credential files:

bash# For Oracle DEV
export ORACLE_DEV_USERNAME="OLYMMQA"
export ORACLE_DEV_ENCRYPTED_PASS_FILE="oracle_dev_password.enc"
export ORACLE_DEV_KEY_VALUE_FILE="oracle_dev_keyfile.key"

# For Oracle UAT
export ORACLE_UAT_USERNAME="GFOLYMQA"
export ORACLE_UAT_ENCRYPTED_PASS_FILE="oracle_uat_password.enc"
export ORACLE_UAT_KEY_VALUE_FILE="oracle_uat_keyfile.key"
Updated Changes for Existing Scripts:
1. get_frequent_columns.py - Additional Update:
python# Add this at the top imports section:
try:
    import oracledb
    ORACLE_CLIENT_AVAILABLE = True
except ImportError:
    ORACLE_CLIENT_AVAILABLE = False
    print("⚠️ Oracle client not available. Install with: pip install oracledb")

# Update the main() function to include database parameter:
def main():
    """Main entry point"""
    parser.add_argument('--database', '-db', type=str, default='impala:prod',
                       help='Database specification (impala:prod, oracle:dev, oracle:uat)')
    
    # ... existing args ...
    
    args = parser.parse_args()
    
    # Validate database specification
    from database_manager import validate_database_specification
    if not validate_database_specification(args.database):
        print(f"Invalid database specification: {args.database}")
        print("Available options: impala:prod, oracle:dev, oracle:uat")
        return 1
    
    # Check Oracle dependency if needed
    if 'oracle' in args.database and not ORACLE_CLIENT_AVAILABLE:
        print("Error: Oracle client library not installed. Run: pip install oracledb")
        return 1
2. get_distinct_values.py - Additional Update:
python# Add Oracle dependency check at the top:
try:
    import oracledb
    ORACLE_CLIENT_AVAILABLE = True
except ImportError:
    ORACLE_CLIENT_AVAILABLE = False

# In the load_config_file method, add database spec to config:
def load_config_file(self, config_file_path: str) -> bool:
    """Load configuration file generated in Step 1"""
    try:
        # ... existing config loading ...
        
        # Check if database specification is provided in config
        database_spec = self.config_data.get('DATABASE_SPEC', 'impala:prod')
        
        # Validate Oracle availability if needed
        if 'oracle' in database_spec and not ORACLE_CLIENT_AVAILABLE:
            self.logger.error("Oracle client library not available. Install with: pip install oracledb")
            return False
        
        # ... rest of method ...
3. training_data_processor.py - Additional Update:
python# Add Oracle dependency check:
try:
    import oracledb
    ORACLE_CLIENT_AVAILABLE = True
except ImportError:
    ORACLE_CLIENT_AVAILABLE = False

# Update validate_inputs function:
def validate_inputs(config: ProcessingConfig) -> bool:
    """Validate input parameters"""
    errors = []
    
    # ... existing validations ...
    
    # Validate Oracle availability if needed
    if 'oracle' in config.database_spec and not ORACLE_CLIENT_AVAILABLE:
        errors.append("Oracle client library not installed. Run: pip install oracledb")
    
    # Validate database specification
    from database_manager import validate_database_specification
    if not validate_database_specification(config.database_spec):
        errors.append(f"Invalid database specification: {config.database_spec}")
    
    # ... rest of validation ...
Enhanced Configuration File Support:
The system now supports configuration files that specify database types. Your existing .conf files can include: