Fix for Impala SQL Syntax
1. Update All Queries to Use Proper Impala Syntax
REPLACE all query methods in query_log_analyzer.py with Impala-compatible versions:
pythondef debug_audit_log_content(self) -> Dict[str, Any]:
    """Debug method using Impala SQL syntax"""
    
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
    
    # Continue with existing implementation...
2. Fix the Main Extract Query for Impala
UPDATE extract_query_logs method:
pythondef extract_query_logs(self, table_name: str, months_back: int) -> List[Dict[str, Any]]:
    """Extract query logs using Impala SQL syntax"""
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months_back * 30)
    
    self.logger.info(f"🔍 Impala extraction: {start_date.strftime('%Y%m%d')} to {end_date.strftime('%Y%m%d')}")
    
    # IMPALA-OPTIMIZED QUERY with memory management
    impala_query = f"""
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
        AND LENGTH(query_text) > 20
        AND LENGTH(query_text) < 10000
    ORDER BY dwh_business_date DESC
    LIMIT 1000
    """
    
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.impala', delete=False) as temp_file:
            temp_output = temp_file.name
        
        self.logger.info("🔍 Executing Impala-optimized query...")
        success = self.db_client.execute_query(impala_query, temp_output, "TXT")
        
        if success and os.path.exists(temp_output):
            file_size = os.path.getsize(temp_output)
            self.logger.info(f"📄 Impala query output: {file_size} bytes")
            
            if file_size > 100:
                log_data = self._parse_log_file(temp_output)
                if log_data:
                    self.logger.info(f"✅ Impala extraction: {len(log_data)} entries")
                    return log_data
                else:
                    self.logger.error("❌ Impala parsing failed")
            else:
                self.logger.error(f"❌ Impala output too small: {file_size} bytes")
        else:
            self.logger.error("❌ Impala query execution failed")
            
    except Exception as e:
        self.logger.error(f"❌ Impala extraction error: {e}")
    
    # FALLBACK: Try smaller, simpler query
    return self._try_simple_impala_query(start_date, end_date)

def _try_simple_impala_query(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
    """Fallback with very simple Impala query"""
    
    # Much simpler query to avoid memory issues
    simple_query = f"""
    SELECT 
        dwh_business_date,
        client_id,
        SUBSTR(query_text, 1, 500) as query_text,
        status
    FROM gfolydal_managed.jdbc_centralized_audit_log
    WHERE 
        dwh_business_date >= {start_date.strftime('%Y%m%d')}
        AND dwh_business_date >= {(datetime.now() - timedelta(days=7)).strftime('%Y%m%d')}
        AND UPPER(query_text) LIKE '%TRADE_FACT%'
        AND LENGTH(query_text) BETWEEN 50 AND 2000
    LIMIT 500
    """
    
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.simple', delete=False) as temp_file:
            temp_output = temp_file.name
        
        self.logger.info("🔍 Trying simple Impala fallback...")
        success = self.db_client.execute_query(simple_query, temp_output, "TXT")
        
        if success and os.path.exists(temp_output):
            log_data = self._parse_log_file(temp_output)
            if log_data:
                self.logger.info(f"✅ Simple fallback: {len(log_data)} entries")
                return log_data
    
    except Exception as e:
        self.logger.error(f"❌ Simple fallback failed: {e}")
    
    return []
3. Add Memory-Efficient Test Query
ADD this method for testing:
pythondef test_impala_memory_safe(self) -> Dict[str, Any]:
    """Test with memory-safe Impala queries"""
    
    # Very simple test queries
    test_queries = {
        "basic_count": """
            SELECT COUNT(*) as total_count
            FROM gfolydal_managed.jdbc_centralized_audit_log
            WHERE dwh_business_date = 20250806
        """,
        
        "recent_trade_queries": """
            SELECT 
                dwh_business_date,
                COUNT(*) as query_count
            FROM gfolydal_managed.jdbc_centralized_audit_log
            WHERE 
                dwh_business_date >= 20250801
                AND UPPER(query_text) LIKE '%TRADE%'
            GROUP BY dwh_business_date
            ORDER BY dwh_business_date DESC
            LIMIT 10
        """,
        
        "sample_queries": """
            SELECT 
                SUBSTR(query_text, 1, 100) as sample_query
            FROM gfolydal_managed.jdbc_centralized_audit_log
            WHERE 
                dwh_business_date = 20250806
                AND UPPER(query_text) LIKE '%SELECT%'
                AND LENGTH(query_text) > 20
            LIMIT 20
        """
    }
    
    results = {}
    
    for query_name, query in test_queries.items():
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{query_name}', delete=False) as temp_file:
                temp_output = temp_file.name
            
            self.logger.info(f"🧪 Testing Impala query: {query_name}")
            success = self.db_client.execute_query(query, temp_output, "TXT")
            
            if success and os.path.exists(temp_output):
                with open(temp_output, 'r') as f:
                    lines = f.readlines()
                    results[query_name] = {
                        'status': 'success',
                        'lines_count': len(lines),
                        'sample_data': [line.strip() for line in lines[:5]]
                    }
                os.unlink(temp_output)
            else:
                results[query_name] = {'status': 'failed', 'error': 'Query execution failed'}
                
        except Exception as e:
            results[query_name] = {'status': 'error', 'error': str(e)}
    
    return results
4. Update Your Main Process Method
REPLACE the debug section in your main process:
python# Step 2.5: Enhanced query log analysis
if self._should_analyze_query_logs(config_data):
    self.logger.info("🚀 Starting Impala-compatible query log analysis...")
    
    # Test Impala compatibility first
    self.logger.info("🧪 Testing Impala memory-safe queries...")
    impala_test = self.query_analyzer.test_impala_memory_safe()
    
    impala_test_file = os.path.join(self.system_config.output_directory, "impala_test_results.json")
    with open(impala_test_file, 'w') as f:
        json.dump(impala_test, f, indent=4)
    self.logger.info(f"📊 Impala test results: {impala_test_file}")
    
    # Continue with existing debug (but with smaller queries)...
🎯 Key Impala Optimizations