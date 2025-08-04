1. Table Name Mismatch in Query Logs
The audit log might use different naming conventions. Let's debug this:
Add this debugging method to query_log_analyzer.py:
pythondef debug_table_search(self, table_name: str, months_back: int = 6) -> Dict[str, Any]:
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
Add this debugging call to your main code. Update _analyze_and_update_frequent_columns:
pythondef _analyze_and_update_frequent_columns(self, tables_list: List[str], config_data: dict):
    """Analyze query logs and update frequent columns"""
    
    self.logger.info("Initializing query log analyzer...")
    query_analyzer = QueryLogAnalyzer(self.db_client)
    
    months_back = int(config_data.get('LOG_ANALYSIS_MONTHS', '6'))
    min_usage = int(config_data.get('LOG_ANALYSIS_MIN_USAGE', '5'))
    
    # CREATE DEBUG OUTPUT
    output_dir = self.system_config.output_directory
    os.makedirs(output_dir, exist_ok=True)
    debug_output_file = os.path.join(output_dir, "debug_table_search.json")
    
    # ... existing code ...
    
    for table_name in tables_list:
        # ADD DEBUGGING STEP
        try:
            self.logger.info(f"🔍 DEBUG: Searching for table patterns in logs: {table_name}")
            debug_results = query_analyzer.debug_table_search(table_name, months_back)
            
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
        
        # Continue with existing analysis...
        columns_key = f"{table_name}_Columns"
        columns_value = config_data.get(columns_key, "")
        
        # ... rest of existing code ...
2. Date Range Issue
Your logs might be outside the 6-month window. Let's also check a broader range:
Update the extract_query_logs method in query_log_analyzer.py:
pythondef extract_query_logs(self, table_name: str, months_back: int) -> List[Dict[str, Any]]:
    """Extract query logs from the audit log table"""
    
    # Parse schema and table name
    if '.' in table_name:
        schema_name, table_only = table_name.split('.', 1)
    else:
        schema_name = ''
        table_only = table_name
    
    # Calculate date range - try broader range if needed
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months_back * 30)
    
    # Log the search parameters
    self.logger.info(f"🔍 Searching logs from {start_date.strftime('%Y%m%d')} to {end_date.strftime('%Y%m%d')}")
    self.logger.info(f"🔍 Looking for table: {table_name}")
    
    # Try multiple search patterns
    search_patterns = [
        table_name,                    # Full name: GFOLYNSD_STANDARDIZATION.OM_TRADE_FACT_ALL
        table_only,                    # Table only: OM_TRADE_FACT_ALL
        f"{schema_name}.{table_only}", # Explicit format
        "OM_TRADE_FACT",              # Partial name
    ]
    
    for pattern in search_patterns:
        self.logger.info(f"🔍 Trying search pattern: {pattern}")
        
        # Build the log extraction query
        log_query = f"""
        SELECT 
            dwh_business_date,
            '{schema_name}' AS schema_name,
            '{table_only}' AS table_name,
            client_id,
            CONCAT('|', dc_log.query_text, '|') AS query_text,
            status
        FROM gfolydal_managed.jdbc_centralized_audit_log AS dc_log
        WHERE 
            dc_log.dwh_business_date BETWEEN {start_date.strftime('%Y%m%d')} AND {end_date.strftime('%Y%m%d')}
            AND dc_log.query_text LIKE '%{pattern}%'
            AND dc_log.client_id NOT LIKE '%fid%'
            AND dc_log.exception_code IS NULL
        LIMIT 1000
        """
        
        # Execute query to get logs
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
                temp_output = temp_file.name
            
            self.logger.info(f"🔍 Executing log query with pattern: {pattern}")
            success = self.db_client.execute_query(log_query, temp_output, "TXT")
            
            if success:
                log_data = self._parse_log_file(temp_output)
                if log_data:
                    self.logger.info(f"✅ Found {len(log_data)} queries with pattern: {pattern}")
                    return log_data
                else:
                    self.logger.info(f"❌ No queries found with pattern: {pattern}")
            else:
                self.logger.error(f"❌ Query execution failed for pattern: {pattern}")
                
        except Exception as e:
            self.logger.error(f"❌ Error with pattern {pattern}: {e}")
    
    # If no patterns worked, return empty
    self.logger.warning(f"❌ No query logs found for table {table_name} with any search pattern")
    return []
3. Database Access Issue
The audit log table might not be accessible or might be named differently:
Add this method to test audit log access:
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
        if os.path.exists(temp_output):
            os.unlink(temp_output)
Quick Test - Run This First:
Add this to your main process method before the log analysis:
python# ADD THIS RIGHT BEFORE THE LOG ANALYSIS STEP
if self._should_analyze_query_logs(config_data):
    self.logger.info("Testing audit log access...")
    
    # Test audit log access
    if hasattr(query_analyzer, 'test_audit_log_access'):
        if not query_analyzer.test_audit_log_access():
            self.logger.error("Cannot access audit log table - skipping log analysis")
        else:
            self.logger.info("Analyzing query logs for frequent columns...")
            self._analyze_and_update_frequent_columns(tables_list, config_data)
    else:
        self.logger.info("Analyzing query logs for frequent columns...")
        self._analyze_and_update_frequent_columns(tables_list, config_data)
Most Likely Issues: