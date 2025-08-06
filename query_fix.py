Update 1: Fix the Query in query_log_analyzer.py
Replace the extract_query_logs method with this working version:
pythondef extract_query_logs(self, table_name: str, months_back: int) -> List[Dict[str, Any]]:
    """Extract query logs using the working query pattern from your tests"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)  # Use 7 days like your working test
    
    self.logger.info(f"🔍 Working query extraction: {start_date.strftime('%Y%m%d')} to {end_date.strftime('%Y%m%d')}")
    
    # Use the WORKING query pattern from your test (Test 5)
    working_query = f"""
    SELECT 
        dwh_business_date,
        client_id,
        SUBSTR(query_text, 1, 300) as query_text
    FROM gfolydal_managed.jdbc_centralized_audit_log
    WHERE 
        dwh_business_date >= {start_date.strftime('%Y%m%d')}
        AND dwh_business_date <= {end_date.strftime('%Y%m%d')}
        AND query_text LIKE '%SELECT%'
    LIMIT 200
    """
    
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.working', delete=False) as temp_file:
            temp_output = temp_file.name
        
        self.logger.info("🔍 Executing working query pattern...")
        success = self.db_client.execute_query(working_query, temp_output, "TXT")
        
        if success and os.path.exists(temp_output):
            file_size = os.path.getsize(temp_output)
            self.logger.info(f"📄 Working query output: {file_size} bytes")
            
            if file_size > 100:
                log_data = self._parse_log_file(temp_output)
                if log_data:
                    self.logger.info(f"✅ Working extraction: {len(log_data)} entries")
                    return log_data
                else:
                    self.logger.warning("⚠️  Parsing failed, trying broader query...")
                    return self._try_broader_working_query(start_date, end_date)
            else:
                self.logger.warning(f"⚠️  Output too small, trying broader query...")
                return self._try_broader_working_query(start_date, end_date)
        else:
            self.logger.error("❌ Working query execution failed")
                
    except Exception as e:
        self.logger.error(f"❌ Working extraction error: {e}")
    
    return []

def _try_broader_working_query(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
    """Try broader query without SELECT filter if needed"""
    
    broader_query = f"""
    SELECT 
        dwh_business_date,
        client_id,
        SUBSTR(query_text, 1, 200) as query_text
    FROM gfolydal_managed.jdbc_centralized_audit_log
    WHERE 
        dwh_business_date >= {start_date.strftime('%Y%m%d')}
        AND dwh_business_date <= {end_date.strftime('%Y%m%d')}
        AND LENGTH(query_text) > 20
    LIMIT 100
    """
    
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.broader', delete=False) as temp_file:
            temp_output = temp_file.name
        
        self.logger.info("🔍 Trying broader working query...")
        success = self.db_client.execute_query(broader_query, temp_output, "TXT")
        
        if success and os.path.exists(temp_output):
            log_data = self._parse_log_file(temp_output)
            if log_data:
                self.logger.info(f"✅ Broader extraction: {len(log_data)} entries")
                return log_data
    
    except Exception as e:
        self.logger.error(f"❌ Broader extraction error: {e}")
    
    return []
Update 2: Fix Case-Insensitive SQL Detection
Update the is_valid_sql method to handle case variations:
pythondef is_valid_sql(self, query_text: str) -> Tuple[bool, float]:
    """
    Determine if query text is a valid SQL statement - case insensitive
    Returns: (is_valid, confidence_score)
    """
    if not query_text or len(query_text.strip()) < 10:
        return False, 0.0
    
    query_upper = query_text.upper().strip()
    confidence = 0.0
    
    # Check for noise patterns (negative indicators)
    for pattern in self.noise_patterns:
        if re.search(pattern, query_text, re.IGNORECASE):
            return False, 0.0
    
    # Check for SQL keywords (positive indicators) - case insensitive
    keyword_count = 0
    for keyword in self.sql_keywords:
        if keyword in query_upper:
            keyword_count += 1
            confidence += 0.1
    
    # Must have SELECT and FROM - check case insensitive
    has_select = bool(re.search(r'\bSELECT\b', query_upper))
    has_from = bool(re.search(r'\bFROM\b', query_upper))
    
    if not (has_select and has_from):
        return False, confidence * 0.2
    
    # Check for proper SQL structure
    structure_score = 0.0
    
    # SELECT ... FROM pattern - case insensitive
    if re.search(r'\bSELECT\b\s+.+\s+\bFROM\b\s+\w+', query_upper):
        structure_score += 0.3
    
    # Proper identifier patterns
    if re.search(r'\b[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\b', query_text):
        structure_score += 0.2
    
    # Semicolon or complete statement
    if query_text.strip().endswith(';') or 'LIMIT' in query_upper:
        structure_score += 0.1
    
    total_confidence = min(confidence + structure_score, 1.0)
    
    # Consider valid if confidence > 0.3 (lowered threshold) and has basic structure
    is_valid = total_confidence > 0.3 and has_select and has_from
    
    return is_valid, total_confidence
Update 3: Enhanced Table-Specific Column Analysis
Update the parse_column_usage_enhanced method:
pythonasync def parse_column_usage_enhanced(self, log_data: List[Dict[str, Any]], table_name: str) -> Dict[str, int]:
    """Enhanced column usage parsing - case insensitive table matching"""
    
    # Filter for queries that mention our table (case insensitive)
    table_mentions = []
    table_simple_name = table_name.split('.')[-1].upper()  # Get just "OM_TRADE_FACT_ALL"
    table_schema = table_name.split('.')[0].upper() if '.' in table_name else ""
    
    self.logger.info(f"🔍 Looking for table: {table_simple_name} in schema: {table_schema}")
    
    for entry in log_data:
        query_text = entry.get('query_text', '').upper()
        
        # Look for any mention of our table name or similar patterns
        table_found = False
        
        # Check for exact table name
        if table_simple_name in query_text:
            table_found = True
        # Check for schema.table pattern
        elif f"{table_schema}.{table_simple_name}" in query_text:
            table_found = True
        # Check for partial matches
        elif 'TRADE_FACT' in query_text and 'OM_' in query_text:
            table_found = True
        # Check for schema name
        elif table_schema and table_schema in query_text:
            table_found = True
        
        if table_found:
            table_mentions.append(entry)
            self.logger.debug(f"🎯 Found table mention: {query_text[:100]}...")
    
    self.logger.info(f"🔍 Found {len(table_mentions)} queries mentioning {table_name} out of {len(log_data)} total")
    
    # If no specific table mentions, analyze general SQL patterns
    if not table_mentions:
        self.logger.info("🔍 No table-specific queries found, analyzing all SQL queries for patterns")
        # Filter to only SQL queries
        sql_queries = []
        for entry in log_data:
            if self.is_valid_sql(entry.get('query_text', ''))[0]:
                sql_queries.append(entry)
        queries_to_analyze = sql_queries[:50]  # Analyze first 50 SQL queries
        self.logger.info(f"🔍 Analyzing {len(queries_to_analyze)} general SQL queries")
    else:
        queries_to_analyze = table_mentions
    
    column_usage = Counter()
    
    # Debug: Show what we're analyzing
    for i, entry in enumerate(queries_to_analyze[:5]):  # Debug first 5
        query_text = entry['query_text']
        self.logger.info(f"🔍 Debug Query {i+1}: {query_text}")
        
        # Use regex extraction
        columns = self.extract_columns_regex(query_text, table_name)
        self.logger.info(f"🔍 Extracted columns: {list(columns)}")
        
        for column in columns:
            if column and len(column) > 1:
                column_usage[column] += 1
    
    # Process remaining queries
    for entry in queries_to_analyze[5:]:
        query_text = entry['query_text']
        columns = self.extract_columns_regex(query_text, table_name)
        for column in columns:
            if column and len(column) > 1:
                column_usage[column] += 1
    
    # If no columns found from table-specific queries, extract common patterns
    if not column_usage and queries_to_analyze:
        self.logger.info("🔍 No table-specific columns found, extracting common column patterns")
        column_usage = self.extract_common_column_patterns(queries_to_analyze)
    
    self.logger.info(f"📊 Final column usage from {len(queries_to_analyze)} queries: {dict(column_usage)}")
    return dict(column_usage)

def extract_common_column_patterns(self, queries: List[Dict[str, Any]]) -> Counter:
    """Extract commonly used column patterns from any SQL queries"""
    
    common_columns = Counter()
    
    for entry in queries:
        query_text = entry.get('query_text', '').upper()
        
        # Look for common trading/financial column patterns
        trading_patterns = [
            r'\b(TRADE_\w+)\b', r'\b(SETTLEMENT_\w+)\b', r'\b(COUNTER_\w+)\b',
            r'\b(PRICE_\w+)\b', r'\b(AMOUNT_\w+)\b', r'\b(\w+_DATE)\b',
            r'\b(\w+_ID)\b', r'\b(\w+_SK)\b', r'\b(\w+_STATUS)\b',
            r'\b(QUANTITY\w*)\b', r'\b(CURRENCY\w*)\b', r'\b(\w*PARTY\w*)\b'
        ]
        
        for pattern in trading_patterns:
            matches = re.findall(pattern, query_text)
            for match in matches:
                common_columns[match.lower()] += 1
        
        # Also extract any column-like identifiers from SELECT clauses
        select_matches = re.findall(r'SELECT\s+(.+?)\s+FROM', query_text, re.DOTALL)
        for select_clause in select_matches:
            # Extract column names from SELECT
            columns = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', select_clause)
            for col in columns:
                if (col.lower() not in ['select', 'from', 'where', 'and', 'or', 'as', 'distinct'] and
                    len(col) > 2):
                    common_columns[col.lower()] += 1
    
    return common_columns
Update 4: Test the Updated Version
Add this test method to your QueryLogAnalyzer class:
pythondef test_case_insensitive_extraction(self) -> Dict[str, Any]:
    """Test case insensitive SQL detection"""
    
    test_cases = [
        "SELECT trade_id FROM trades",
        "Select amount, price From transactions", 
        "select * from payments",
        "SeLeCt count(*) FrOm orders",
    ]
    
    results = {}
    for i, query in enumerate(test_cases):
        is_valid, confidence = self.is_valid_sql(query)
        results[f"test_{i+1}"] = {
            "query": query,
            "is_valid": is_valid,
            "confidence": confidence
        }
        self.logger.info(f"Test {i+1}: '{query}' -> Valid: {is_valid}, Confidence: {confidence:.2f}")
    
    return results
Summary of Changes