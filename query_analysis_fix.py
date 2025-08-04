Fix 1: Move query_analyzer creation to class level
Update your TrainingDataProcessor.__init__ method:
pythondef __init__(self, config_files: List[str]):
    self.setup_logging()
    self.logger = logging.getLogger(__name__)
    
    try:
        self.config_manager = ConfigManager(config_files)
        self.db_config = self.config_manager.get_database_config()
        self.system_config = self.config_manager.get_system_config()
        self.db_client = DatabaseClient(self.db_config, self.system_config)
        self.metadata_manager = MetadataManager(self.db_client, self.system_config)
        
        # ADD THIS: Initialize query analyzer at class level
        self.query_analyzer = QueryLogAnalyzer(self.db_client)
        
        # Load rules engine
        rules_path = os.path.join(os.path.dirname(__file__), 'rules.json')
        self.rules_engine = RulesEngine(rules_path)
        self.output_generator = OutputGenerator(self.system_config)
        
    except Exception as e:
        self.logger.error(f"Failed to initialize processor: {e}")
        raise
Fix 2: Update the process method
Update your process method:
python# NEW STEP 2.5: Analyze query logs for frequent columns
if self._should_analyze_query_logs(config_data):
    self.logger.info("Testing audit log access...")
    
    # Test audit log access using self.query_analyzer
    if hasattr(self.query_analyzer, 'test_audit_log_access'):
        if not self.query_analyzer.test_audit_log_access():
            self.logger.error("Cannot access audit log table - skipping log analysis")
        else:
            self.logger.info("Audit log access successful! Analyzing query logs for frequent columns...")
            self._analyze_and_update_frequent_columns(tables_list, config_data)
    else:
        self.logger.info("Analyzing query logs for frequent columns...")
        self._analyze_and_update_frequent_columns(tables_list, config_data)
Fix 3: Update the _analyze_and_update_frequent_columns method
Update your _analyze_and_update_frequent_columns method to use the class-level analyzer:
pythondef _analyze_and_update_frequent_columns(self, tables_list: List[str], config_data: dict):
    """Analyze query logs and update frequent columns"""
    
    self.logger.info("Using existing query log analyzer...")
    # Remove this line: query_analyzer = QueryLogAnalyzer(self.db_client)
    # Use self.query_analyzer instead
    
    months_back = int(config_data.get('LOG_ANALYSIS_MONTHS', '6'))
    min_usage = int(config_data.get('LOG_ANALYSIS_MIN_USAGE', '5'))
    
    # CREATE DEBUG OUTPUT
    output_dir = self.system_config.output_directory
    os.makedirs(output_dir, exist_ok=True)
    debug_output_file = os.path.join(output_dir, "debug_table_search.json")
    detailed_output_file = os.path.join(output_dir, "column_discovery_analysis.json")
    comparison_output_file = os.path.join(output_dir, "column_comparison_analysis.json")
    all_analysis_results = {}
    comparison_results = {}
    
    for table_name in tables_list:
        # ADD DEBUGGING STEP
        try:
            self.logger.info(f"🔍 DEBUG: Searching for table patterns in logs: {table_name}")
            debug_results = self.query_analyzer.debug_table_search(table_name, months_back)
            
            # Save debug results
            with open(debug_output_file, 'w') as f:
                json.dump(debug_results, f, indent=4)
            
            self.logger.info(f"🔍 DEBUG: Search results saved to {debug_output_file}")
            
            # Log debug results
            for pattern, result in debug_results.items():
                if 'error' not in result:
                    count = result.get('total_count', '0')
                    self.logger.info(f"🔍 {pattern}: {count} queries found")
                    if int(count) > 0:
                        self.logger.info(f"   Date range: {result.get('earliest_date')} to {result.get('latest_date')}")
                else:
                    self.logger.error(f"🔍 {pattern}: Error - {result['error']}")
                    
        except Exception as e:
            self.logger.error(f"Debug search failed: {e}")
        
        columns_key = f"{table_name}_Columns"
        columns_value = config_data.get(columns_key, "")
        
        try:
            self.logger.info(f"Analyzing query logs for table: {table_name}")
            print(f"DEBUG: Starting log analysis for {table_name}")
            print(f"DEBUG: Column setting: {columns_value[:100]}...")
            
            # Use self.query_analyzer instead of query_analyzer
            usage_data = self.query_analyzer.analyze_table_usage(table_name, months_back=months_back)
            all_analysis_results[table_name] = usage_data
            
            # ... rest of your existing code remains the same ...
Fix 4: Add the missing methods to query_log_analyzer.py
Add these methods to your QueryLogAnalyzer class in query_log_analyzer.py:
pythondef test_audit_log_access(self) -> bool:
    """Test if we can access the audit log table"""
    
    test_query = """
    SELECT COUNT(*) as total_records
    FROM gfolydal_managed.jdbc_centralized_audit_log 
    WHERE dwh_business_date >= 20240101
    LIMIT 1
    """
    
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.test', delete=False) as temp_file:
            temp_output = temp_file.name
        
        self.logger.info("🔍 Testing audit log table access...")
        success = self.db_client.execute_query(test_query, temp_output, "TXT")
        
        if success:
            with open(temp_output, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    count = lines[1].strip()
                    self.logger.info(f"✅ Audit log accessible: {count} total records found")
                    return True
        
        self.logger.error("❌ Cannot access audit log table")
        return False
        
    except Exception as e:
        self.logger.error(f"❌ Audit log access test failed: {e}")
        return False
    finally:
        if 'temp_output' in locals() and os.path.exists(temp_output):
            os.unlink(temp_output)

def debug_table_search(self, table_name: str, months_back: int = 6) -> Dict[str, Any]:
    """Debug method to find how the table appears in logs"""
    
    # Parse schema and table name
    if '.' in table_name:
        schema_name, table_only = table_name.split('.', 1)
    else:
        schema_name = ''
        table_only = table_name
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months_back * 30)
    
    # Try different search patterns
    search_patterns = [
        f"%{table_name}%",  # Full name with schema
        f"%{table_only}%",  # Just table name
        f"%{schema_name}%",  # Just schema name
        f"%OM_TRADE_FACT%",  # Partial table name
        f"%TRADE_FACT%",    # Even more partial
    ]
    
    results = {}
    
    for i, pattern in enumerate(search_patterns):
        debug_query = f"""
        SELECT 
            COUNT(*) as total_count,
            MIN(dwh_business_date) as earliest_date,
            MAX(dwh_business_date) as latest_date
        FROM gfolydal_managed.jdbc_centralized_audit_log AS dc_log
        WHERE 
            dc_log.dwh_business_date BETWEEN {start_date.strftime('%Y%m%d')} AND {end_date.strftime('%Y%m%d')}
            AND dc_log.query_text LIKE '{pattern}'
            AND dc_log.client_id NOT LIKE '%fid%'
            AND dc_log.exception_code IS NULL
        """
        
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.debug', delete=False) as temp_file:
                temp_output = temp_file.name
            
            success = self.db_client.execute_query(debug_query, temp_output, "TXT")
            
            if success:
                with open(temp_output, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 2:
                        data = lines[1].strip().split(',')
                        results[f"pattern_{i+1}_{pattern}"] = {
                            'total_count': data[0] if len(data) > 0 else '0',
                            'earliest_date': data[1] if len(data) > 1 else 'N/A',
                            'latest_date': data[2] if len(data) > 2 else 'N/A'
                        }
                
                # Cleanup
                if os.path.exists(temp_output):
                    os.unlink(temp_output)
                    
        except Exception as e:
            results[f"pattern_{i+1}_{pattern}"] = {'error': str(e)}
    
    return results
Summary of Changes: