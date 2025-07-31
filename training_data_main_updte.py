 troubleshooting:
1. Update DatabaseClient.execute_query() method:
Add detailed logging before and after the subprocess call:
pythondef execute_query(self, query: str, output_file: str, format_type: str = "TXT") -> bool:
    """Execute a database query and save results to file"""
    try:
        command = [
            self.system_config.jdk_path,
            "--add-opens=java.base/java.nio=ALL-UNNAMED",
            "-jar", self.system_config.jar_path,
            self.db_config.host,
            f"--query={query}",
            f"--user={self.db_config.user_id}",
            f"--pass={self.db_config.password}",  # Consider masking this in logs
            f"--env={self.db_config.environment}",
            f"--format={format_type}",
            f"--destination={output_file}"
        ]
        
        # ADD THESE LOGGING LINES:
        self.logger.info(f"Executing query: {query}")
        self.logger.info(f"Database host: {self.db_config.host}")
        self.logger.info(f"Database user: {self.db_config.user_id}")
        self.logger.info(f"Java path: {self.system_config.jdk_path}")
        self.logger.info(f"JAR path: {self.system_config.jar_path}")
        self.logger.info(f"Output file: {output_file}")
        
        # Check if Java and JAR files exist
        if not os.path.exists(self.system_config.jdk_path):
            self.logger.error(f"Java executable not found: {self.system_config.jdk_path}")
            return False
        
        if not os.path.exists(self.system_config.jar_path):
            self.logger.error(f"JAR file not found: {self.system_config.jar_path}")
            return False
        
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        
        # ADD DETAILED ERROR LOGGING:
        self.logger.info(f"Command return code: {result.returncode}")
        if result.stdout:
            self.logger.info(f"Command stdout: {result.stdout}")
        if result.stderr:
            self.logger.error(f"Command stderr: {result.stderr}")
        
        if result.returncode != 0:
            self.logger.error(f"Query execution failed with code {result.returncode}")
            self.logger.error(f"Error output: {result.stderr}")
            return False
        
        # Check if output file was created
        if not os.path.exists(output_file):
            self.logger.error(f"Output file was not created: {output_file}")
            return False
        
        # Check if output file has content
        file_size = os.path.getsize(output_file)
        self.logger.info(f"Output file size: {file_size} bytes")
        
        return True
        
    except subprocess.TimeoutExpired:
        self.logger.error(f"Query execution timed out: {query}")
        return False
    except Exception as e:
        self.logger.error(f"Error executing query: {e}")
        return False
2. Update ConfigManager.get_database_config() method:
Add validation logging:
pythondef get_database_config(self, section: str = "transcation") -> DatabaseConfig:
    """Get database configuration"""
    config_data = self.get_section_config(section)
    
    # ADD VALIDATION LOGGING:
    if 'USERID' not in config_data:
        print(f"WARNING: USERID not found in section [{section}]")
    if 'PASSWORD' not in config_data:
        print(f"WARNING: PASSWORD not found in section [{section}]")
    
    # Handle Windows authentication differently
    if os.name == 'nt':  # Windows
        user_id = config_data.get('USERID', '')
        password = config_data.get('PASSWORD', '')
    else:
        user_id = os.environ.get('USER_FOR_TRAINING', '')
        password = self._decrypt_password()
    
    # ADD CREDENTIAL VALIDATION:
    if not user_id:
        print(f"ERROR: No user ID configured for section [{section}]")
    if not password:
        print(f"ERROR: No password configured for section [{section}]")
    
    print(f"Database config - User: {user_id}, Password: {'*' * len(password) if password else 'NOT SET'}")
    
    return DatabaseConfig(
        host=self._get_host_from_config(config_data),
        user_id=user_id,
        password=password
    )
3. Add a new test method in DatabaseClient:
Add this new method to test database connectivity:
pythondef test_connection(self) -> bool:
    """Test database connection with a simple query"""
    try:
        # Create a temporary output file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.test', delete=False) as temp_file:
            temp_output = temp_file.name
        
        # Try a simple query
        test_query = "SELECT 1 as test_connection"
        
        self.logger.info("Testing database connection...")
        success = self.execute_query(test_query, temp_output, "TXT")
        
        if success and os.path.exists(temp_output):
            with open(temp_output, 'r') as f:
                content = f.read()
                self.logger.info(f"Connection test result: {content}")
            os.unlink(temp_output)
            return True
        else:
            self.logger.error("Connection test failed")
            return False
            
    except Exception as e:
        self.logger.error(f"Connection test error: {e}")
        return False
4. Update MetadataManager.load_table_metadata() method:
Add table existence check:
pythondef load_table_metadata(self, table_name: str) -> bool:
    """Load metadata for a specific table"""
    # ADD TABLE EXISTENCE CHECK:
    self.logger.info(f"Checking if table exists: {table_name}")
    
    table_dir = os.path.join(
        self.system_config.base_directory, 
        self.system_config.date_type, 
        "table", 
        table_name
    )
    os.makedirs(table_dir, exist_ok=True)
    
    schema_file = os.path.join(table_dir, f"{table_name}.schema")
    
    # ADD FILE PATH LOGGING:
    self.logger.info(f"Schema file path: {schema_file}")
    
    schema = self.db_client.get_table_schema(table_name, schema_file)
    
    if schema:
        self.table_schemas[table_name] = schema
        self.logger.info(f"Loaded metadata for table: {table_name} ({len(schema)} columns)")
        # ADD COLUMN DETAILS:
        for col_name, col_info in schema.items():
            self.logger.debug(f"  Column: {col_name} -> {col_info}")
        return True
    
    self.logger.error(f"Failed to load metadata for table: {table_name}")
    return False
5. Add connection test to main workflow:
In the TrainingDataProcessor.process() method, add this after creating the db_client:
pythondef process(self, section: str = "transcation") -> bool:
    """Main processing workflow"""
    try:
        self.logger.info("Starting training data processing workflow")
        
        # Get configuration data
        config_data = self.config_manager.get_section_config(section)
        tables_list = self.config_manager.get_tables_list(section)
        
        # ADD CONNECTION TEST HERE:
        self.logger.info("Testing database connection...")
        if hasattr(self.db_client, 'test_connection'):
            if not self.db_client.test_connection():
                self.logger.error("Database connection test failed. Check credentials and network connectivity.")
                return False
            else:
                self.logger.info("Database connection test successful!")
        
        # ... rest of the method
6. Quick debugging commands to add temporarily:
Add these at the start of the problematic query execution:
python# In DatabaseClient.execute_query(), add before subprocess.run():
print(f"DEBUG: Full command: {' '.join(command)}")
print(f"DEBUG: Working directory: {os.getcwd()}")
print(f"DEBUG: Java version check...")
java_version_result = subprocess.run([self.system_config.jdk_path, "-version"], 
                                   capture_output=True, text=True)
print(f"DEBUG: Java check result: {java_version_result.stderr}")
These updates will help you identify: