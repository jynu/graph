Update 1: Simplify Query in query_log_analyzer.py
Replace the extract_query_logs method with this much simpler version:
pythondef extract_query_logs(self, table_name: str, months_back: int) -> List[Dict[str, Any]]:
    end_date = datetime.now()
    # Just use yesterday's data for proof of concept
    start_date = end_date - timedelta(days=1)
    
    self.logger.info(f"🔍 Proof-of-concept extraction: {start_date.strftime('%Y%m%d')} to {end_date.strftime('%Y%m%d')}")
    
    # VERY simple query for proof of concept
    simple_query = f"""
    SELECT 
        dwh_business_date,
        client_id,
        SUBSTR(query_text, 1, 300) as query_text,
        status
    FROM gfolydal_managed.jdbc_centralized_audit_log
    WHERE 
        dwh_business_date = {start_date.strftime('%Y%m%d')}  -- Just 1 day
        AND UPPER(query_text) LIKE '%SELECT%'
        AND UPPER(query_text) LIKE '%TRADE_FACT%'
        AND LENGTH(query_text) > 50
        AND exception_code IS NULL
    LIMIT 200  -- Very small sample
    """
    
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.simple', delete=False) as temp_file:
            temp_output = temp_file.name
        
        self.logger.info("🔍 Executing simple proof-of-concept query...")
        success = self.db_client.execute_query(simple_query, temp_output, "TXT")
        
        if success and os.path.exists(temp_output):
            file_size = os.path.getsize(temp_output)
            self.logger.info(f"📄 Simple query output: {file_size} bytes")
            
            if file_size > 100:
                log_data = self._parse_log_file(temp_output)
                if log_data:
                    self.logger.info(f"✅ Simple extraction: {len(log_data)} entries")
                    return log_data
            else:
                self.logger.error(f"❌ Simple output too small: {file_size} bytes")
        else:
            self.logger.error("❌ Simple query execution failed")
                
    except Exception as e:
        self.logger.error(f"❌ Simple extraction error: {e}")
    
    return []
Update 2: Fix Encoding in _parse_log_file
Add encoding handling to your existing _parse_log_file method:
pythondef _parse_log_file(self, log_file: str) -> List[Dict[str, Any]]:
    """Parse the log file and return structured data"""
    log_data = []
    
    try:
        # Try different encodings to handle binary data
        content = None
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(log_file, 'r', encoding=encoding) as f:
                    content = f.read()
                self.logger.info(f"✅ File read successfully with {encoding} encoding")
                break
            except UnicodeDecodeError:
                self.logger.debug(f"Failed to read with {encoding}, trying next...")
                continue
        
        if not content:
            self.logger.error("❌ Could not decode file with any encoding")
            return []
        
        # Clean any problematic characters
        import re
        content = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', ' ', content)
        
        # Continue with your existing parsing logic
        lines = content.strip().split('\n')
        self.logger.info(f"📄 Parsing file with {len(lines)} total lines")
        
        # Rest of your existing parsing code...
        header_found = False
        data_start_index = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Look for header patterns
            if ('dwh_business_date' in line.lower() or 
                'client_id' in line.lower() or
                i == 0):
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
                # Handle different separators
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
                        self.logger.info(f"📄 Parsed entry {processed_count}: {log_entry['query_text'][:100]}...")
                
            except Exception as parse_error:
                self.logger.debug(f"📄 Failed to parse line {i}: {line[:50]}... - Error: {parse_error}")
                continue
        
        self.logger.info(f"✅ Successfully parsed {len(log_data)} valid log entries")
        
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
Update 3: Reduce Analysis Scope in Config
In your main processing, temporarily reduce the analysis scope:
python# In the _analyze_and_update_frequent_columns_enhanced method
# Change this line:
months_back = int(config_data.get('LOG_ANALYSIS_MONTHS', '6'))

# To this for proof of concept:
months_back = 1  # Just 1 month for now, but we'll only use 1 day anyway
Update 4: Add Debug Output
Add this to see what columns are being extracted:
pythonasync def parse_column_usage_enhanced(self, log_data: List[Dict[str, Any]], table_name: str) -> Dict[str, int]:
    """Enhanced column usage parsing with debug output"""
    
    # First, filter and clean the queries
    clean_queries = self.filter_and_clean_queries(log_data)
    
    self.logger.info(f"🔍 Processing {len(clean_queries)} clean queries for {table_name}")
    
    column_usage = Counter()
    
    # Add debug output for first few queries
    for i, entry in enumerate(clean_queries[:5]):  # Debug first 5 queries
        query_text = entry['query_text']
        self.logger.info(f"🔍 Debug Query {i+1}: {query_text}")
        
        # Use regex extraction for now (simpler than LLM)
        columns = self.extract_columns_regex(query_text, table_name)
        self.logger.info(f"🔍 Extracted columns: {list(columns)}")
        
        for column in columns:
            if column and len(column) > 1:
                column_usage[column] += 1
    
    # Process remaining queries without debug
    for entry in clean_queries[5:]:
        query_text = entry['query_text']
        columns = self.extract_columns_regex(query_text, table_name)
        for column in columns:
            if column and len(column) > 1:
                column_usage[column] += 1
    
    self.logger.info(f"📊 Final column usage: {dict(column_usage)}")
    return dict(column_usage)
These changes will: