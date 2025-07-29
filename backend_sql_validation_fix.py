# Replace the SQLGenerator class in your text_to_sql_backend.py with this updated version:

class SQLGenerator:
    """Enhanced SQL generator using latest best practices for text-to-SQL."""
    
    def __init__(self):
        self.conn = duckdb.connect(DB_PATH)
    
    async def generate_sql(self, query: str, tables: List[str], table_details: Dict) -> Tuple[str, str]:
        """Generate Impala SQL using enhanced context and prompting strategy."""
    
        # Create enhanced schema context with relationships
        schema_context = self._create_enhanced_schema_context(tables, table_details)
        
        # Generate SQL using Impala-specific prompt
        sql_prompt = self._create_impala_sql_prompt(query, schema_context)
        
        try:
            if not CLIENT_MANAGER_AVAILABLE:
                return self._generate_fallback_impala_sql(query, tables), "Fallback Impala SQL generation used (no GPT available)"
            
            # Use enhanced prompting with more context
            response = await client_manager.ask_gpt(sql_prompt)
            
            # Extract and validate SQL
            sql_code = self._extract_sql_from_response(response)
            
            # Validate SQL syntax for Impala compatibility (FIXED VERSION)
            validation_result = self._validate_impala_sql_fixed(sql_code, tables)
            print("sql validation result valid: ", validation_result['is_valid'])
            
            if validation_result['is_valid']:
                return sql_code, response
            else:
                # Attempt to fix SQL for Impala compatibility
                fixed_sql = await self._fix_sql(sql_code, validation_result['error'], query)
                return fixed_sql, f"Original response: {response}\n\nFixed for Impala compatibility: {validation_result['error']}"
                
        except Exception as e:
            logger.error(f"Enhanced SQL generation failed: {e}")
            fallback_sql = self._generate_fallback_impala_sql(query, tables)
            return fallback_sql, f"Error generating SQL, using Impala fallback: {str(e)}"
    
    def _validate_impala_sql_fixed(self, sql: str, available_tables: List[str] = None) -> Dict:
        """Fixed validation that properly handles table references."""
        try:
            # Extract table names from SQL
            referenced_tables = self._extract_table_names_from_sql(sql)
            
            # Check if referenced tables are in our available tables list
            if available_tables:
                missing_tables = []
                for ref_table in referenced_tables:
                    # Clean table name (remove schema prefix)
                    clean_ref = ref_table.split('.')[-1] if '.' in ref_table else ref_table
                    
                    # Check if it matches any available table (case-insensitive)
                    found = False
                    for avail_table in available_tables:
                        clean_avail = avail_table.split('.')[-1] if '.' in avail_table else avail_table
                        if clean_ref.upper() == clean_avail.upper() or ref_table.upper() == avail_table.upper():
                            found = True
                            break
                    
                    if not found:
                        # Also check if the table exists in knowledge graph
                        try:
                            check_sql = "SELECT COUNT(*) FROM tables WHERE UPPER(name) = UPPER(?)"
                            result = self.conn.execute(check_sql, [clean_ref]).fetchone()
                            if not result or result[0] == 0:
                                # Try with full name
                                result = self.conn.execute(check_sql, [ref_table]).fetchone()
                                if not result or result[0] == 0:
                                    missing_tables.append(ref_table)
                        except Exception as e:
                            logger.warning(f"Could not verify table {ref_table}: {e}")
                
                if missing_tables:
                    return {
                        'is_valid': False, 
                        'error': f"Table(s) not found: {', '.join(missing_tables)}. Available tables: {', '.join(available_tables[:5])}{'...' if len(available_tables) > 5 else ''}"
                    }
            
            # Impala-specific validation checks
            impala_issues = []
            sql_upper = sql.upper()
            
            if 'FULL OUTER JOIN' in sql_upper:
                impala_issues.append("FULL OUTER JOIN not supported in Impala, use LEFT/RIGHT JOIN")
            
            if 'VARCHAR(' in sql_upper:
                impala_issues.append("Use STRING instead of VARCHAR in Impala")
            
            if ' UNION ' in sql_upper and ' UNION ALL ' not in sql_upper:
                impala_issues.append("Consider using UNION ALL instead of UNION for better performance")
            
            # Basic syntax checks
            syntax_issues = self._check_basic_sql_syntax(sql)
            if syntax_issues:
                impala_issues.extend(syntax_issues)
            
            if impala_issues:
                return {'is_valid': False, 'error': '; '.join(impala_issues)}
            
            return {'is_valid': True, 'error': None}
            
        except Exception as e:
            # Don't fail validation due to validation errors
            logger.warning(f"SQL validation error: {e}")
            return {'is_valid': True, 'error': f"Validation warning: {str(e)}"}
    
    def _extract_table_names_from_sql(self, sql: str) -> List[str]:
        """Extract table names from SQL query."""
        import re
        
        tables = set()
        sql_upper = sql.upper()
        
        # Patterns to find table references
        patterns = [
            r'FROM\s+([^\s,\)\(;]+)',
            r'JOIN\s+([^\s,\)\(;]+)',
            r'UPDATE\s+([^\s,\)\(;]+)',
            r'INTO\s+([^\s,\)\(;]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, sql_upper, re.IGNORECASE)
            for match in matches:
                # Clean up the table name
                table_name = match.strip()
                # Remove alias if present
                if ' ' in table_name:
                    table_name = table_name.split()[0]
                # Remove quotes
                table_name = table_name.strip('"`\'')
                # Skip SQL keywords
                if table_name and table_name.upper() not in ['SELECT', 'WHERE', 'GROUP', 'ORDER', 'HAVING', 'ON', 'AS']:
                    tables.add(table_name)
        
        return list(tables)
    
    def _check_basic_sql_syntax(self, sql: str) -> List[str]:
        """Check basic SQL syntax issues."""
        issues = []
        sql_upper = sql.upper().strip()
        
        # Check if SQL starts with valid statement
        valid_starts = ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE', 'CREATE']
        if not any(sql_upper.startswith(start) for start in valid_starts):
            issues.append("SQL should start with a valid statement")
        
        # Check balanced parentheses
        if sql.count('(') != sql.count(')'):
            issues.append("Unbalanced parentheses")
        
        # Check for FROM clause in SELECT
        if sql_upper.startswith('SELECT') and 'FROM' not in sql_upper:
            issues.append("SELECT statement missing FROM clause")
        
        return issues
    
    # Keep all your other existing methods unchanged:
    # _generate_fallback_impala_sql, _create_schema_context, _create_enhanced_schema_context,
    # _get_table_relationships_context, _get_business_rules_context, _create_impala_sql_prompt,
    # _extract_sql_from_response, _validate_sql, _fix_sql, _generate_fallback_sql