Specific Fixes Needed
1. Fix the extract_query_logs Method
UPDATE this method in query_log_analyzer.py around line 350:
pythondef extract_query_logs(self, table_name: str, months_back: int) -> List[Dict[str, Any]]:
    """Extract query logs from the audit log table"""
    
    # Since your debug shows 6.6M matches, let's get them directly
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months_back * 30)
    
    self.logger.info(f"🔍 Direct extraction: {start_date.strftime('%Y%m%d')} to {end_date.strftime('%Y%m%d')}")
    
    # SIMPLIFIED DIRECT QUERY - since debug shows data exists
    direct_query = f"""
    SELECT 
        dwh_business_date,
        client_id,
        query_text,
        status
    FROM gfolydal_managed.jdbc_centralized_audit_log
    WHERE 
        dwh_business_date >= {start_date.strftime('%Y%m%d')}
        AND dwh_business_date <= {end_date.strftime('%Y%m%d')}
        AND (
            UPPER(query_text) LIKE '%OM_TRADE_FACT%'
            OR UPPER(query_text) LIKE '%TRADE_FACT%'
            OR UPPER(query_text) LIKE '%GFOLYNSD%'
        )
        AND exception_code IS NULL
        AND UPPER(query_text) LIKE '%SELECT%'
    ORDER BY dwh_business_date DESC
    LIMIT 2000
    """
    
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.direct', delete=False) as temp_file:
            temp_output = temp_file.name
        
        self.logger.info("🔍 Executing direct extraction query...")
        self.logger.debug(f"Query: {direct_query}")
        
        success = self.db_client.execute_query(direct_query, temp_output, "TXT")
        
        if success and os.path.exists(temp_output):
            file_size = os.path.getsize(temp_output)
            self.logger.info(f"📄 Direct query output file size: {file_size} bytes")
            
            if file_size > 100:
                # LOG FIRST FEW LINES TO DEBUG
                with open(temp_output, 'r') as f:
                    lines = f.readlines()
                    self.logger.info(f"📄 Direct query returned {len(lines)} lines")
                    self.logger.info("📄 First 5 lines:")
                    for i, line in enumerate(lines[:5]):
                        self.logger.info(f"   {i}: {line.strip()}")
                
                log_data = self._parse_log_file(temp_output)
                if log_data:
                    self.logger.info(f"✅ Successfully parsed {len(log_data)} log entries")
                    return log_data
                else:
                    self.logger.error("❌ Parsing returned empty data")
            else:
                self.logger.error(f"❌ Output file too small: {file_size} bytes")
        else:
            self.logger.error("❌ Direct query failed or no output file")
            
    except Exception as e:
        self.logger.error(f"❌ Direct extraction error: {e}")
    
    return []
2. Fix the _parse_log_file Method
REPLACE the current _parse_log_file method completely:
pythondef _parse_log_file(self, log_file: str) -> List[Dict[str, Any]]:
    """Parse the log file and return structured data"""
    log_data = []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into lines
        lines = content.strip().split('\n')
        
        self.logger.info(f"📄 Parsing file with {len(lines)} total lines")
        
        # Find header and data
        header_found = False
        data_start_index = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Look for header patterns
            if ('dwh_business_date' in line.lower() or 
                'client_id' in line.lower() or
                i == 0):  # First non-empty line is likely header
                self.logger.info(f"📄 Found header at line {i}: {line}")
                header_found = True
                data_start_index = i + 1
                break
        
        # Process data lines
        processed_count = 0
        for i in range(data_start_index, len(lines)):
            line = lines[i].strip()
            if not line:
                continue
            
            try:
                # MORE FLEXIBLE PARSING
                # Handle different possible separators
                if ',' in line:
                    parts = self._split_csv_line(line)
                elif '\t' in line:
                    parts = line.split('\t')
                else:
                    parts = line.split()
                
                # Ensure we have at least 4 parts
                while len(parts) < 4:
                    parts.append('')
                
                # Create log entry
                log_entry = {
                    'dwh_business_date': parts[0].strip(),
                    'client_id': parts[1].strip() if len(parts) > 1 else '',
                    'query_text': parts[2].strip() if len(parts) > 2 else '',
                    'status': parts[3].strip() if len(parts) > 3 else ''
                }
                
                # Only include if query_text is not empty
                if log_entry['query_text']:
                    log_data.append(log_entry)
                    processed_count += 1
                    
                    # Debug first few entries
                    if processed_count <= 3:
                        self.logger.debug(f"📄 Parsed entry {processed_count}: {log_entry}")
                
            except Exception as parse_error:
                self.logger.debug(f"📄 Failed to parse line {i}: {line[:100]} - Error: {parse_error}")
                continue
        
        self.logger.info(f"✅ Successfully parsed {len(log_data)} valid log entries from {len(lines)} total lines")
        
        # Clean up the temporary file
        try:
            if os.path.exists(log_file):
                os.unlink(log_file)
        except Exception as cleanup_error:
            self.logger.warning(f"Failed to cleanup temp file {log_file}: {cleanup_error}")
        
        return log_data
        
    except Exception as e:
        self.logger.error(f"Error parsing log file: {e}")
        return []

def _split_csv_line(self, line: str) -> List[str]:
    """Split CSV line handling quoted fields"""
    parts = []
    current_part = ""
    in_quotes = False
    
    i = 0
    while i < len(line):
        char = line[i]
        
        if char == '"':
            in_quotes = not in_quotes
        elif char == ',' and not in_quotes:
            parts.append(current_part.strip())
            current_part = ""
        else:
            current_part += char
        i += 1
    
    # Add the last part
    if current_part:
        parts.append(current_part.strip())
    
    return parts
3. Add Better Debug Logging
ADD this debug method to test direct data access:
pythondef test_direct_table_queries(self, table_name: str) -> Dict[str, Any]:
    """Test direct queries to see actual data structure"""
    
    # Test 1: Get recent table-related queries
    test_query = f"""
    SELECT 
        dwh_business_date,
        client_id,
        SUBSTR(query_text, 1, 200) as query_text_sample,
        status
    FROM gfolydal_managed.jdbc_centralized_audit_log
    WHERE 
        dwh_business_date >= 20240701
        AND (
            UPPER(query_text) LIKE '%{table_name}%'
            OR UPPER(query_text) LIKE '%TRADE_FACT%'
        )
        AND exception_code IS NULL
    ORDER BY dwh_business_date DESC
    LIMIT 50
    """
    
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.test_direct', delete=False) as temp_file:
            temp_output = temp_file.name
        
        self.logger.info(f"🧪 Testing direct table queries for: {table_name}")
        success = self.db_client.execute_query(test_query, temp_output, "TXT")
        
        if success and os.path.exists(temp_output):
            with open(temp_output, 'r') as f:
                lines = f.readlines()
                
            self.logger.info(f"🧪 Direct test returned {len(lines)} lines")
            result = {
                'total_lines': len(lines),
                'sample_lines': [line.strip() for line in lines[:10]]
            }
            
            os.unlink(temp_output)
            return result
        else:
            return {'error': 'Query failed'}
            
    except Exception as e:
        return {'error': str(e)}
4. Update Your Main Process to Use Debug Test
ADD this to your process method right after the audit log debug:
python# Step 2.5: Enhanced query log analysis
if self._should_analyze_query_logs(config_data):
    self.logger.info("🚀 Starting enhanced query log analysis...")
    
    # Existing debug code...
    debug_results = self.query_analyzer.debug_audit_log_content()
    # ... save debug results ...
    
    # ADD THIS NEW TEST:
    for table_name in tables_list:
        self.logger.info(f"🧪 Testing direct data access for: {table_name}")
        direct_test = self.query_analyzer.test_direct_table_queries(table_name)
        
        direct_test_file = os.path.join(self.system_config.output_directory, f"direct_test_{table_name.replace('.', '_')}.json")
        with open(direct_test_file, 'w') as f:
            json.dump(direct_test, f, indent=4)
        self.logger.info(f"📊 Direct test results: {direct_test_file}")
    
    # Continue with existing analysis...
🎯 Expected Results