1. Enhanced Schema Context with Joins
Update _create_schema_context() in SQLGenerator class:
pythondef _create_enhanced_schema_context(self, tables: List[str], table_details: Dict) -> str:
    """Create comprehensive schema context with relationships and sample data."""
    schema_parts = []
    
    # Add database/schema information header
    schema_parts.append("-- Impala SQL Database Schema")
    schema_parts.append("-- Target: Impala/Hadoop Data Warehouse")
    schema_parts.append("")
    
    # 1. Create table definitions with enhanced information
    for table_name in tables:
        if table_name in table_details:
            details = table_details[table_name]
            
            # Table header with business context
            table_type = details.get('table_type', 'table')
            description = details.get('description', '')
            
            schema_parts.append(f"-- Table: {table_name} ({table_type})")
            if description:
                schema_parts.append(f"-- Business Purpose: {description}")
            
            # Add confidence and similarity info if available
            if 'similarity_score' in details:
                schema_parts.append(f"-- Relevance Score: {details['similarity_score']:.3f}")
            
            # Column definitions with sample data
            columns = details.get('columns', [])
            if columns:
                schema_parts.append(f"CREATE TABLE {table_name} (")
                col_definitions = []
                
                for col in columns:
                    col_def = f"  {col['name']} {col['data_type']}"
                    
                    # Add description and sample data as comments
                    comments = []
                    if col.get('description'):
                        comments.append(f"Description: {col['description']}")
                    
                    # Add sample values if available from distinct_values
                    if col.get('distinct_values'):
                        sample_values = col['distinct_values'][:3]  # First 3 values
                        comments.append(f"Sample values: {sample_values}")
                    
                    if comments:
                        col_def += f" -- {' | '.join(comments)}"
                    
                    col_definitions.append(col_def)
                
                schema_parts.append(",\n".join(col_definitions))
                schema_parts.append(");")
            
            schema_parts.append("")
    
    # 2. Add table relationships from knowledge graph
    relationships_context = self._get_table_relationships_context(tables)
    if relationships_context:
        schema_parts.append("-- TABLE RELATIONSHIPS AND JOIN PATTERNS:")
        schema_parts.append(relationships_context)
        schema_parts.append("")
    
    # 3. Add business rules and constraints
    business_rules = self._get_business_rules_context(tables, table_details)
    if business_rules:
        schema_parts.append("-- BUSINESS RULES AND CONSTRAINTS:")
        schema_parts.append(business_rules)
        schema_parts.append("")
    
    return "\n".join(schema_parts)

def _get_table_relationships_context(self, tables: List[str]) -> str:
    """Get relationship context between selected tables."""
    try:
        relationships_info = []
        
        # Get relationships from knowledge graph
        relationships_sql = """
        SELECT from_table, to_table, from_column, to_column, relationship_type, confidence
        FROM relationships 
        WHERE (from_table IN ({}) AND to_table IN ({}))
           OR (from_table IN ({}) AND to_table IN ({}))
        """.format(
            ','.join(['?' for _ in tables]), ','.join(['?' for _ in tables]),
            ','.join(['?' for _ in tables]), ','.join(['?' for _ in tables])
        )
        
        params = tables + tables + tables + tables
        relationships = self.conn.execute(relationships_sql, params).fetchall()
        
        for from_table, to_table, from_col, to_col, rel_type, confidence in relationships:
            if from_table in tables and to_table in tables:
                join_example = f"-- {from_table} JOIN {to_table} ON {from_table}.{from_col} = {to_table}.{to_col}"
                relationship_info = f"{join_example} -- {rel_type} (confidence: {confidence:.2f})"
                relationships_info.append(relationship_info)
        
        return "\n".join(relationships_info) if relationships_info else ""
        
    except Exception as e:
        logger.warning(f"Failed to get relationships context: {e}")
        return ""

def _get_business_rules_context(self, tables: List[str], table_details: Dict) -> str:
    """Get business rules and constraints context."""
    rules = []
    
    for table_name in tables:
        if table_name in table_details:
            details = table_details[table_name]
            
            # Add table-specific rules if available
            if 'rules' in details and details['rules']:
                rules.append(f"-- {table_name}: {details['rules']}")
            
            # Add column-level constraints
            columns = details.get('columns', [])
            for col in columns:
                col_name = col['name']
                # Identify key columns
                if any(keyword in col_name.lower() for keyword in ['_sk', '_id', '_key']):
                    rules.append(f"-- {table_name}.{col_name} is a key column used for joins")
                
                # Identify date columns
                if any(keyword in col_name.lower() for keyword in ['date', 'time', 'timestamp']):
                    rules.append(f"-- {table_name}.{col_name} is a date/time column")
    
    return "\n".join(rules) if rules else ""
2. Impala-Specific SQL Prompt
Update _create_enhanced_sql_prompt() for Impala compatibility:
pythondef _create_impala_sql_prompt(self, query: str, schema_context: str) -> str:
    """Create enhanced SQL prompt optimized for Impala SQL dialect."""
    
    prompt = f"""You are an expert Impala SQL developer specializing in big data analytics on Hadoop. Your task is to write precise, efficient Impala SQL queries based on natural language requests.

**TASK**: Convert the user's natural language question into syntactically correct Impala SQL that follows best practices for big data analytics.

**TARGET DATABASE**: Impala/Hadoop Data Warehouse
**SQL DIALECT**: Impala SQL (follows SQL-92 standard with Hadoop extensions)

**DATABASE SCHEMA:**
{schema_context}

**USER QUESTION:**
{query}

**IMPALA SQL GUIDELINES:**

1. **Impala-Specific Syntax**:
   - Use explicit JOIN syntax (INNER JOIN, LEFT JOIN, RIGHT JOIN)
   - No FULL OUTER JOIN (not supported in Impala)
   - Use UNION ALL instead of UNION when possible (better performance)
   - Prefer WHERE clauses over HAVING when filtering non-aggregated data
   - Use LIMIT for large result sets to avoid overwhelming queries

2. **Big Data Best Practices**:
   - Always use table aliases for readability: `FROM table1 t1 JOIN table2 t2`
   - Use partitioned columns in WHERE clauses when available
   - Prefer columnar operations and avoid row-by-row processing
   - Use appropriate data types (STRING, INT, BIGINT, DOUBLE, TIMESTAMP)
   - Consider using STRAIGHT_JOIN hint for join optimization if needed

3. **Impala Data Types**:
   - Use STRING instead of VARCHAR
   - Use BIGINT for large integers
   - Use DOUBLE for decimal numbers
   - Use TIMESTAMP for date/time (stored in UTC)
   - Use BOOLEAN for true/false values

4. **Query Structure Optimization**:
   - SELECT: Choose only necessary columns, use column aliases
   - FROM: Start with the largest/most filtered table
   - JOIN: Use explicit join conditions with key columns
   - WHERE: Apply most selective filters first, use partition columns
   - GROUP BY: Use column positions (1, 2, 3) or exact column names
   - ORDER BY: Only when necessary, consider using LIMIT
   - Use COMPUTE STATS tables regularly for query optimization

5. **Impala-Specific Functions**:
   - Use regexp_extract() for pattern matching
   - Use from_unixtime() for timestamp conversion
   - Use cast() for explicit type conversions
   - Use nvl() or coalesce() for null handling
   - Use substr() instead of substring()

6. **Performance Optimization**:
   - Use partition pruning when tables are partitioned by date/region
   - Avoid SELECT * in production queries
   - Use appropriate aggregation functions (count(), sum(), avg(), max(), min())
   - Consider using analytical window functions for ranking/running totals
   - Use EXISTS instead of IN for subqueries when possible

7. **Common Patterns for Analytics**:
   - Time-based analysis: Filter by date partitions first
   - Aggregation queries: Use appropriate GROUP BY with aggregate functions
   - Ranking queries: Use ROW_NUMBER() or RANK() window functions
   - Fact-dimension joins: Join fact tables to dimension tables on surrogate keys
   - Top-N queries: Use ORDER BY with LIMIT

8. **Error Prevention**:
   - Verify all column names exist in specified tables
   - Ensure JOIN conditions use compatible data types
   - Use CAST() for type conversions between string and numeric
   - Handle potential NULL values with COALESCE() or NVL()
   - Use consistent case for keywords (uppercase recommended)

**RESPONSE FORMAT:**
Return ONLY the Impala SQL query without explanations, comments, or markdown formatting. The query should be production-ready and optimized for Impala/Hadoop execution.

**IMPALA SQL QUERY:**"""

    return prompt
3. Enhanced SQL Generation Method
Update the main generate_sql() method:
pythonasync def generate_sql(self, query: str, tables: List[str], table_details: Dict) -> Tuple[str, str]:
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
        
        # Validate SQL syntax for Impala compatibility
        validation_result = self._validate_impala_sql(sql_code)
        
        if validation_result['is_valid']:
            return sql_code, response
        else:
            # Attempt to fix SQL for Impala compatibility
            fixed_sql = await self._fix_impala_sql(sql_code, validation_result['error'], query)
            return fixed_sql, f"Original response: {response}\n\nFixed for Impala compatibility: {validation_result['error']}"
            
    except Exception as e:
        logger.error(f"Enhanced SQL generation failed: {e}")
        fallback_sql = self._generate_fallback_impala_sql(query, tables)
        return fallback_sql, f"Error generating SQL, using Impala fallback: {str(e)}"

def _validate_impala_sql(self, sql: str) -> Dict:
    """Validate SQL for Impala compatibility."""
    try:
        # Basic syntax validation using DuckDB (as proxy)
        self.conn.execute(f"EXPLAIN {sql}")
        
        # Additional Impala-specific checks
        impala_issues = []
        
        # Check for unsupported features
        sql_upper = sql.upper()
        
        if 'FULL OUTER JOIN' in sql_upper:
            impala_issues.append("FULL OUTER JOIN not supported in Impala, use LEFT/RIGHT JOIN")
        
        if 'VARCHAR(' in sql_upper:
            impala_issues.append("Use STRING instead of VARCHAR in Impala")
        
        if ' UNION ' in sql_upper and ' UNION ALL ' not in sql_upper:
            impala_issues.append("Consider using UNION ALL instead of UNION for better performance")
        
        if impala_issues:
            return {'is_valid': False, 'error': '; '.join(impala_issues)}
        
        return {'is_valid': True, 'error': None}
        
    except Exception as e:
        return {'is_valid': False, 'error': str(e)}

def _generate_fallback_impala_sql(self, query: str, tables: List[str]) -> str:
    """Generate basic Impala SQL fallback."""
    if tables:
        main_table = tables[0]
        # Create a simple Impala-compatible query
        return f"""SELECT *
FROM {main_table}
LIMIT 100;

-- Fallback query for: {query}
-- Consider adding WHERE clauses, JOINs, and specific column selections"""
    else:
        return "-- No tables found for query"
4. Add Relationship Helper Method
Add this method to your AdvancedGraphTraversalRetriever class:
pythondef get_table_relationships(self, tables: List[str]) -> List[Dict]:
    """Get relationships between the specified tables."""
    try:
        if len(tables) < 2:
            return []
        
        relationships_sql = """
        SELECT from_table, to_table, from_column, to_column, relationship_type, confidence
        FROM relationships 
        WHERE (from_table IN ({}) AND to_table IN ({}))
           OR (to_table IN ({}) AND from_table IN ({}))
        """.format(
            ','.join(['?' for _ in tables]), ','.join(['?' for _ in tables]),
            ','.join(['?' for _ in tables]), ','.join(['?' for _ in tables])
        )
        
        params = tables + tables + tables + tables
        relationships = self.conn.execute(relationships_sql, params).fetchall()
        
        return [
            {
                'from_table': row[0],
                'to_table': row[1], 
                'from_column': row[2],
                'to_column': row[3],
                'relationship_type': row[4],
                'confidence': row[5]
            }
            for row in relationships
        ]
        
    except Exception as e:
        logger.warning(f"Failed to get table relationships: {e}")
        return []
Summary of Key Improvements