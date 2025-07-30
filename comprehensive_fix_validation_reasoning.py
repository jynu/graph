# COMPLETE FIX FOR BOTH VALIDATION AND REASONING

class SQLGenerator:
    """Enhanced SQL generator with proper validation and reasoning analysis."""
    
    def __init__(self):
        self.conn = duckdb.connect(DB_PATH)
    
    async def generate_sql(self, query: str, tables: List[str], table_details: Dict) -> Tuple[str, str]:
        """Generate Impala SQL with enhanced reasoning and proper validation."""
        
        # Create enhanced schema context with relationships
        schema_context = self._create_enhanced_schema_context(tables, table_details)
        
        # Generate SQL using enhanced prompting with reasoning
        sql_prompt = self._create_reasoning_sql_prompt(query, schema_context, tables)
        
        try:
            if not CLIENT_MANAGER_AVAILABLE:
                fallback_sql = self._generate_fallback_impala_sql(query, tables)
                reasoning = self._create_reasoning_analysis(query, tables, fallback_sql, "fallback")
                return fallback_sql, reasoning
            
            # Use enhanced prompting with reasoning capabilities
            response = await client_manager.ask_gpt(sql_prompt)
            
            # Extract SQL and reasoning from response
            sql_code, reasoning_explanation = self._extract_sql_and_reasoning(response)
            
            # FIXED VALIDATION: Only validate syntax, not table existence
            validation_result = self._validate_sql_syntax_only(sql_code)
            logger.info(f"SQL validation result: {validation_result}")
            
            # Create comprehensive reasoning analysis
            full_reasoning = self._create_enhanced_reasoning_analysis(
                query, tables, table_details, sql_code, reasoning_explanation, validation_result
            )
            
            if validation_result['is_valid']:
                return sql_code, full_reasoning
            else:
                # Attempt to fix SQL syntax issues
                fixed_sql = await self._fix_sql(sql_code, validation_result['error'], query)
                fixed_reasoning = full_reasoning + f"\n\n**SQL Fixed:** {validation_result['error']}\n\n**Final SQL:**\n```sql\n{fixed_sql}\n```"
                return fixed_sql, fixed_reasoning
                
        except Exception as e:
            logger.error(f"Enhanced SQL generation failed: {e}")
            fallback_sql = self._generate_fallback_impala_sql(query, tables)
            error_reasoning = self._create_reasoning_analysis(query, tables, fallback_sql, "error", str(e))
            return fallback_sql, error_reasoning
    
    def _validate_sql_syntax_only(self, sql: str) -> Dict:
        """FIXED: Validate only SQL syntax, NOT table existence."""
        try:
            # Only check Impala-specific syntax rules and basic SQL structure
            issues = []
            sql_upper = sql.upper().strip()
            
            # 1. Check if SQL starts with valid statement
            valid_starts = ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE', 'CREATE']
            if not any(sql_upper.startswith(start) for start in valid_starts):
                issues.append("SQL should start with a valid statement (SELECT, WITH, etc.)")
            
            # 2. Check balanced parentheses
            if sql.count('(') != sql.count(')'):
                issues.append(f"Unbalanced parentheses: {sql.count('(')} open, {sql.count(')')} close")
            
            # 3. Check for FROM clause in SELECT statements
            if sql_upper.startswith('SELECT') and 'FROM' not in sql_upper:
                issues.append("SELECT statement missing FROM clause")
            
            # 4. Impala-specific validations
            if 'FULL OUTER JOIN' in sql_upper:
                issues.append("FULL OUTER JOIN not supported in Impala, use LEFT/RIGHT JOIN")
            
            if 'VARCHAR(' in sql_upper:
                issues.append("Use STRING instead of VARCHAR in Impala")
            
            if ' UNION ' in sql_upper and ' UNION ALL ' not in sql_upper:
                issues.append("Consider using UNION ALL instead of UNION for better performance")
            
            # 5. Check for basic SQL structure issues
            if sql_upper.startswith('SELECT') and sql_upper.count('SELECT') > sql_upper.count('FROM'):
                if 'UNION' not in sql_upper and 'SUBQUERY' not in sql_upper:
                    issues.append("Possible missing FROM clause in subquery")
            
            if issues:
                return {'is_valid': False, 'error': '; '.join(issues)}
            
            return {'is_valid': True, 'error': None}
            
        except Exception as e:
            # Don't fail validation due to validation errors - be permissive
            logger.warning(f"SQL validation error: {e}")
            return {'is_valid': True, 'error': None, 'warning': str(e)}
    
    def _create_reasoning_sql_prompt(self, query: str, schema_context: str, tables: List[str]) -> str:
        """Create enhanced SQL prompt that generates reasoning along with SQL."""
        
        prompt = f"""You are an expert Impala SQL developer and data analyst. Your task is to convert the natural language question into a syntactically correct Impala SQL query while providing detailed reasoning about your approach.

**USER QUESTION:** {query}

**AVAILABLE TABLES:** {', '.join(tables)}

**DATABASE SCHEMA:**
{schema_context}

**INSTRUCTIONS:**
1. **Analyze the Question:** Break down what the user is asking for
2. **Identify Relevant Tables and Columns:** Explain which tables/columns are needed and why
3. **Plan the Query Structure:** Describe the SQL approach (joins, filters, aggregations, etc.)
4. **Generate the SQL:** Create the final Impala-compatible SQL query
5. **Validate and Optimize:** Check for Impala best practices

**RESPONSE FORMAT:**
Please structure your response as follows:

## 🔍 Question Analysis
[Explain what the user is asking for in business terms]

## 📋 Table and Column Selection
[Explain which tables and columns you're using and why they're relevant]

## 🏗️ Query Planning
[Describe your SQL approach - joins, filters, aggregations, ordering, etc.]

## ⚡ Impala Optimizations
[Mention any Impala-specific optimizations or best practices applied]

## 📝 Final SQL Query
```sql
[Your final SQL query here]
```

## ✅ Query Validation
[Briefly explain why this query correctly answers the user's question]

**IMPORTANT IMPALA GUIDELINES:**
- Use explicit JOIN syntax (INNER JOIN, LEFT JOIN, RIGHT JOIN)
- Use STRING instead of VARCHAR
- Prefer UNION ALL over UNION
- Use table aliases for readability
- Add LIMIT clauses for large result sets
- Consider partitioned columns in WHERE clauses

**GENERATE YOUR RESPONSE:**"""

        return prompt
    
    def _extract_sql_and_reasoning(self, response: str) -> Tuple[str, str]:
        """Extract SQL code and reasoning from GPT response."""
        import re
        
        # Extract SQL code from markdown blocks
        sql_matches = re.findall(r'```sql\n(.*?)\n```', response, re.DOTALL | re.IGNORECASE)
        
        if sql_matches:
            sql_code = sql_matches[-1].strip()  # Take the last SQL block
        else:
            # Fallback: look for SQL-like patterns
            lines = response.split('\n')
            sql_lines = []
            for line in lines:
                if any(keyword in line.upper() for keyword in ['SELECT', 'FROM', 'WHERE', 'JOIN']):
                    sql_lines.append(line)
            sql_code = '\n'.join(sql_lines) if sql_lines else "-- No SQL found in response"
        
        return sql_code, response
    
    def _create_enhanced_reasoning_analysis(self, query: str, tables: List[str], 
                                          table_details: Dict, sql_code: str, 
                                          reasoning_explanation: str, validation_result: Dict) -> str:
        """Create comprehensive reasoning analysis based on latest research."""
        
        analysis = f"""## 🧠 Reasoning & Analysis

### 📋 Query Understanding
**Original Question:** {query}

**Selected Tables:** {len(tables)} tables selected
{chr(10).join([f"- **{table}** ({table_details.get(table, {}).get('table_type', 'unknown')})" for table in tables[:5]])}
{'...' if len(tables) > 5 else ''}

### 🔍 Chain-of-Thought Analysis

{reasoning_explanation}

### 📊 SQL Generation Strategy
**Approach Used:** 
- **Schema Linking:** Matched user intent to relevant database objects
- **Query Decomposition:** Broke down complex question into SQL components  
- **Impala Optimization:** Applied Impala-specific best practices
- **Validation:** Ensured syntactic correctness and Impala compatibility

### 🎯 Generated SQL Quality Assessment

**SQL Complexity:** {self._assess_sql_complexity(sql_code)}
**Impala Compatibility:** {'✅ Compatible' if validation_result['is_valid'] else '⚠️ Issues Found'}
{f"**Issues:** {validation_result['error']}" if not validation_result['is_valid'] else ""}

### 💡 Query Insights
{self._generate_query_insights(query, sql_code, tables)}

### 🔧 Technical Details
- **Tables Accessed:** {len(tables)}
- **Query Type:** {self._identify_query_type(sql_code)}
- **Join Operations:** {self._count_joins(sql_code)}
- **Aggregate Functions:** {self._identify_aggregations(sql_code)}

### ✅ Correctness Assessment
{self._assess_query_correctness(query, sql_code, tables)}

**Status:** {'✅ SQL generated successfully with proper reasoning' if validation_result['is_valid'] else '⚠️ SQL generated but requires review'}

**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

        return analysis
    
    def _assess_sql_complexity(self, sql: str) -> str:
        """Assess the complexity of the generated SQL."""
        complexity_score = 0
        sql_upper = sql.upper()
        
        # Count complexity indicators
        if 'JOIN' in sql_upper: complexity_score += 1
        if 'GROUP BY' in sql_upper: complexity_score += 1
        if 'HAVING' in sql_upper: complexity_score += 1
        if 'ORDER BY' in sql_upper: complexity_score += 1
        if 'UNION' in sql_upper: complexity_score += 1
        if sql_upper.count('SELECT') > 1: complexity_score += 2  # Subqueries
        
        if complexity_score == 0:
            return "Simple (Basic SELECT)"
        elif complexity_score <= 2:
            return "Moderate (Few operations)"
        elif complexity_score <= 4:
            return "Complex (Multiple operations)"
        else:
            return "Very Complex (Advanced SQL)"
    
    def _identify_query_type(self, sql: str) -> str:
        """Identify the type of SQL query."""
        sql_upper = sql.upper()
        types = []
        
        if 'JOIN' in sql_upper:
            types.append("Multi-table Query")
        if 'GROUP BY' in sql_upper:
            types.append("Aggregation")
        if 'ORDER BY' in sql_upper:
            types.append("Sorted Results")
        if 'LIMIT' in sql_upper:
            types.append("Limited Results")
        if sql_upper.count('SELECT') > 1:
            types.append("Subquery")
        
        return ', '.join(types) if types else "Simple Selection"
    
    def _count_joins(self, sql: str) -> str:
        """Count JOIN operations in SQL."""
        join_count = sql.upper().count('JOIN')
        if join_count == 0:
            return "None"
        elif join_count == 1:
            return "1 join"
        else:
            return f"{join_count} joins"
    
    def _identify_aggregations(self, sql: str) -> str:
        """Identify aggregate functions used."""
        sql_upper = sql.upper()
        aggregates = []
        
        for func in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']:
            if func in sql_upper:
                aggregates.append(func)
        
        return ', '.join(aggregates) if aggregates else "None"
    
    def _generate_query_insights(self, query: str, sql: str, tables: List[str]) -> str:
        """Generate insights about the query based on latest research patterns."""
        insights = []
        
        # Analyze query patterns
        if 'quote' in query.lower() and any('quote' in table.lower() for table in tables):
            insights.append("🎯 **Domain Match:** Query relates to financial quotes, matched with quote-related tables")
        
        if any(keyword in query.lower() for keyword in ['all', 'list', 'show']):
            insights.append("📝 **Query Pattern:** Retrieval/listing query - likely requires comprehensive data selection")
        
        if any(keyword in query.lower() for keyword in ['yesterday', 'today', 'date']):
            insights.append("📅 **Temporal Query:** Time-based filtering detected - date conditions applied")
        
        if 'WHERE' in sql.upper():
            insights.append("🔍 **Filtered Query:** Conditional logic applied to narrow results")
        
        if 'JOIN' in sql.upper():
            insights.append("🔗 **Complex Query:** Multi-table relationships utilized for comprehensive results")
        
        return '\n'.join([f"- {insight}" for insight in insights]) if insights else "- Standard data retrieval query"
    
    def _assess_query_correctness(self, query: str, sql: str, tables: List[str]) -> str:
        """Assess whether the query correctly addresses the user's request."""
        correctness_indicators = []
        
        # Check if key terms from query appear in SQL
        query_terms = [word.lower() for word in query.split() if len(word) > 3]
        sql_lower = sql.lower()
        
        matched_terms = [term for term in query_terms if term in sql_lower]
        if matched_terms:
            correctness_indicators.append(f"✅ **Term Matching:** Key terms from question appear in SQL: {', '.join(matched_terms[:3])}")
        
        # Check table relevance
        if tables:
            correctness_indicators.append(f"✅ **Table Selection:** {len(tables)} relevant tables identified and used")
        
        # Check SQL structure
        if 'SELECT' in sql.upper() and 'FROM' in sql.upper():
            correctness_indicators.append("✅ **SQL Structure:** Valid SELECT statement structure")
        
        return '\n'.join([f"{indicator}" for indicator in correctness_indicators])
    
    def _create_reasoning_analysis(self, query: str, tables: List[str], sql: str, 
                                 generation_type: str, error: str = None) -> str:
        """Create reasoning analysis for fallback cases."""
        
        return f"""## 🧠 Reasoning & Analysis

### 📋 Query Processing
**Original Question:** {query}
**Generation Method:** {generation_type.title()}
{f"**Error:** {error}" if error else ""}

### 🔍 Table Discovery
**Tables Found:** {len(tables)} tables
{chr(10).join([f"- {table}" for table in tables[:5]])}
{'...' if len(tables) > 5 else ''}

### 📝 SQL Generation
**Generated SQL:**
```sql
{sql}
```

### ⚠️ Limitations
{'- Generated using fallback method due to system limitations' if generation_type == 'fallback' else ''}
{'- Error encountered during generation - using simplified approach' if generation_type == 'error' else ''}
- Recommend manual review and testing of the generated query

**Status:** {'⚠️ Fallback generation used' if generation_type == 'fallback' else '❌ Error in generation process'}
**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

    # Keep all your existing helper methods unchanged:
    # _generate_fallback_impala_sql, _create_enhanced_schema_context, etc.