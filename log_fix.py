Specific Code Updates Needed
1. Fix the extract_query_logs method in query_log_analyzer.py
UPDATE the extract_query_logs method around line 280-320:
pythondef extract_query_logs(self, table_name: str, months_back: int) -> List[Dict[str, Any]]:
    """Extract query logs from the audit log table"""
    
    # Parse schema and table name
    if '.' in table_name:
        schema_name, table_only = table_name.split('.', 1)
    else:
        schema_name = ''
        table_only = table_name
    
    # FIX 1: Use broader date range and more recent dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months_back * 30)
    
    # FIX 2: Add debug logging for date range
    self.logger.info(f"🔍 Date range: {start_date.strftime('%Y%m%d')} to {end_date.strftime('%Y%m%d')}")
    
    # FIX 3: Improved search patterns - try simpler patterns first
    search_patterns = [
        "OM_TRADE_FACT_ALL",          # Exact table name
        "OM_TRADE_FACT",              # Partial name
        "TRADE_FACT_ALL",             # Without OM prefix
        "TRADE_FACT",                 # Even shorter
        f"{schema_name}%{table_only}", # Schema with wildcard
    ]
    
    for pattern in search_patterns:
        self.logger.info(f"🔍 Trying pattern: '{pattern}'")
        
        # FIX 4: Simplified and corrected query
        log_query = f"""
        SELECT 
            dwh_business_date,
            client_id,
            query_text,
            status
        FROM gfolydal_managed.jdbc_centralized_audit_log
        WHERE 
            dwh_business_date >= {start_date.strftime('%Y%m%d')}
            AND dwh_business_date <= {end_date.strftime('%Y%m%d')}
            AND UPPER(query_text) LIKE UPPER('%{pattern}%')
            AND client_id NOT LIKE '%fid%'
            AND exception_code IS NULL
        ORDER BY dwh_business_date DESC
        LIMIT 500
        """
        
        # FIX 5: Better error handling and debugging
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
                temp_output = temp_file.name
            
            self.logger.info(f"🔍 Executing query for pattern: {pattern}")
            self.logger.debug(f"Query: {log_query}")
            
            success = self.db_client.execute_query(log_query, temp_output, "TXT")
            
            if success and os.path.exists(temp_output):
                # FIX 6: Check file size before parsing
                file_size = os.path.getsize(temp_output)
                self.logger.info(f"📄 Output file size: {file_size} bytes")
                
                if file_size > 100:  # File has content beyond headers
                    log_data = self._parse_log_file(temp_output)
                    if log_data:
                        self.logger.info(f"✅ Found {len(log_data)} queries with pattern: {pattern}")
                        return log_data
                    else:
                        self.logger.info(f"❌ File parsed but no valid data for pattern: {pattern}")
                else:
                    self.logger.info(f"❌ Empty or header-only file for pattern: {pattern}")
            else:
                self.logger.error(f"❌ Query failed or no output file for pattern: {pattern}")
                
        except Exception as e:
            self.logger.error(f"❌ Error with pattern '{pattern}': {e}")
    
    # FIX 7: Final debug attempt - try very broad search
    self.logger.warning("🔍 No specific patterns worked, trying broad search...")
    return self._try_broad_search(start_date, end_date)
2. Add this new method to QueryLogAnalyzer class:
pythondef _try_broad_search(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
    """Try a very broad search to see what's in the audit log"""
    
    broad_query = f"""
    SELECT 
        dwh_business_date,
        client_id,
        SUBSTR(query_text, 1, 100) as query_text_sample,
        status,
        COUNT(*) as query_count
    FROM gfolydal_managed.jdbc_centralized_audit_log
    WHERE 
        dwh_business_date >= {start_date.strftime('%Y%m%d')}
        AND dwh_business_date <= {end_date.strftime('%Y%m%d')}
        AND exception_code IS NULL
    GROUP BY dwh_business_date, client_id, SUBSTR(query_text, 1, 100), status
    ORDER BY query_count DESC, dwh_business_date DESC
    LIMIT 50
    """
    
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.debug', delete=False) as temp_file:
            temp_output = temp_file.name
        
        self.logger.info("🔍 Running broad search to understand data structure...")
        success = self.db_client.execute_query(broad_query, temp_output, "TXT")
        
        if success and os.path.exists(temp_output):
            with open(temp_output, 'r') as f:
                lines = f.readlines()
                self.logger.info(f"📊 Broad search results ({len(lines)} lines):")
                for i, line in enumerate(lines[:10]):  # Show first 10 lines
                    self.logger.info(f"   {i}: {line.strip()}")
            
            # Clean up
            os.unlink(temp_output)
        
    except Exception as e:
        self.logger.error(f"Broad search failed: {e}")
    
    return []  # Return empty as this is just for debugging
3. Fix the _parse_log_file method around line 240:
UPDATE this method:
pythondef _parse_log_file(self, log_file: str) -> List[Dict[str, Any]]:
    """Parse the log file and return structured data"""
    log_data = []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        self.logger.info(f"📄 Parsing file with {len(lines)} lines")
        
        for i, line in enumerate(lines):
            if i == 0:  # Skip header
                self.logger.debug(f"Header: {line.strip()}")
                continue
            
            # FIX: Better parsing logic
            line = line.strip()
            if not line:
                continue
                
            # Handle comma-separated values more carefully
            parts = []
            current_part = ""
            in_quotes = False
            
            for char in line:
                if char == '"':
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
                    parts.append(current_part.strip())
                    current_part = ""
                else:
                    current_part += char
            
            # Add the last part
            if current_part:
                parts.append(current_part.strip())
            
            # FIX: Flexible parsing based on actual columns
            if len(parts) >= 4:
                log_data.append({
                    'dwh_business_date': parts[0],
                    'client_id': parts[1] if len(parts) > 1 else '',
                    'query_text': parts[2] if len(parts) > 2 else '',
                    'status': parts[3] if len(parts) > 3 else ''
                })
                
                # Debug first few entries
                if i <= 5:
                    self.logger.debug(f"Parsed entry {i}: {log_data[-1]}")
        
        self.logger.info(f"✅ Successfully parsed {len(log_data)} log entries")
        
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
4. Add debug method to test specific queries:
ADD this method to QueryLogAnalyzer class:
pythondef debug_audit_log_content(self) -> Dict[str, Any]:
    """Debug method to understand what's actually in the audit log"""
    
    debug_queries = {
        "total_recent_records": """
            SELECT COUNT(*) as total_count
            FROM gfolydal_managed.jdbc_centralized_audit_log
            WHERE dwh_business_date >= 20240701
        """,
        
        "sample_query_patterns": """
            SELECT 
                SUBSTR(query_text, 1, 50) as query_sample,
                COUNT(*) as occurrence_count
            FROM gfolydal_managed.jdbc_centralized_audit_log
            WHERE dwh_business_date >= 20240701
            GROUP BY SUBSTR(query_text, 1, 50)
            ORDER BY occurrence_count DESC
            LIMIT 20
        """,
        
        "table_name_search": """
            SELECT COUNT(*) as count
            FROM gfolydal_managed.jdbc_centralized_audit_log
            WHERE dwh_business_date >= 20240701
            AND (
                UPPER(query_text) LIKE '%OM_TRADE_FACT%'
                OR UPPER(query_text) LIKE '%TRADE_FACT%'
                OR UPPER(query_text) LIKE '%GFOLYNSD%'
            )
        """
    }
    
    results = {}
    
    for query_name, query in debug_queries.items():
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{query_name}', delete=False) as temp_file:
                temp_output = temp_file.name
            
            self.logger.info(f"🔍 Running debug query: {query_name}")
            success = self.db_client.execute_query(query, temp_output, "TXT")
            
            if success and os.path.exists(temp_output):
                with open(temp_output, 'r') as f:
                    lines = f.readlines()
                    results[query_name] = [line.strip() for line in lines]
                    self.logger.info(f"✅ {query_name}: {len(lines)} results")
                    
                os.unlink(temp_output)
            else:
                results[query_name] = ["Query failed"]
                
        except Exception as e:
            results[query_name] = [f"Error: {e}"]
            self.logger.error(f"Debug query {query_name} failed: {e}")
    
    return results
5. Update the main processor to use debug mode:
ADD this to your process method in TrainingDataProcessor, right after the audit log access test:
python# Step 2.5: Enhanced query log analysis
if self._should_analyze_query_logs(config_data):
    self.logger.info("🚀 Starting enhanced query log analysis...")
    
    # ADD THIS DEBUG SECTION:
    self.logger.info("🔍 Running audit log debug analysis...")
    debug_results = self.query_analyzer.debug_audit_log_content()
    
    debug_file = os.path.join(self.system_config.output_directory, "audit_log_debug.json")
    with open(debug_file, 'w') as f:
        json.dump(debug_results, f, indent=4)
    self.logger.info(f"📊 Audit log debug results saved to: {debug_file}")
    
    # Continue with existing analysis...
🎯 Quick Testing Steps

Update only the methods I mentioned above