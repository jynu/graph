# enhanced_query_log_analyzer.py
import re
import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Set
import logging
from collections import Counter

# Import your existing client manager for GPT-4 calls
# from app.utils.client_manager import client_manager

class EnhancedQueryLogAnalyzer:
    """Enhanced analyzer with intelligent SQL parsing using hybrid rule-based + LLM approach"""
    
    def __init__(self, db_client):
        self.db_client = db_client
        self.logger = logging.getLogger(__name__)
        
        # SQL validation patterns
        self.sql_keywords = {
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER',
            'GROUP', 'ORDER', 'BY', 'HAVING', 'LIMIT', 'INSERT', 'UPDATE', 'DELETE',
            'CREATE', 'DROP', 'ALTER', 'INDEX', 'VIEW', 'TABLE'
        }
        
        # Patterns for noise detection
        self.noise_patterns = [
            r'The batch size',
            r'cumulative row',
            r'Finalizing Resources',
            r'Connection Details',
            r'Count of total',
            r'Next set of ResultSet',
            r'in seconds',
            r'^\s*\d+\s*$',  # Just numbers
            r'^\s*null\s*$',  # Just null
            r'^\s*$',  # Empty
        ]
        
        # Column extraction patterns
        self.column_patterns = [
            # Standard table.column format
            r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*([a-zA-Z_][a-zA-Z0-9_]*)\b',
            # Direct column references in SELECT
            r'SELECT\s+(?:DISTINCT\s+)?([^,\s]+(?:\s*,\s*[^,\s]+)*)\s+FROM',
            # WHERE clause columns
            r'WHERE\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[=<>!]',
            # GROUP BY columns  
            r'GROUP\s+BY\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*)',
            # ORDER BY columns
            r'ORDER\s+BY\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*)',
        ]
    
    def is_valid_sql(self, query_text: str) -> Tuple[bool, float]:
        """
        Determine if query text is a valid SQL statement
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
        
        # Check for SQL keywords (positive indicators)
        keyword_count = 0
        for keyword in self.sql_keywords:
            if keyword in query_upper:
                keyword_count += 1
                confidence += 0.1
        
        # Must have at least SELECT and FROM for basic SQL
        has_select = 'SELECT' in query_upper
        has_from = 'FROM' in query_upper
        
        if not (has_select and has_from):
            return False, confidence * 0.2  # Low confidence
        
        # Check for proper SQL structure
        structure_score = 0.0
        
        # SELECT ... FROM pattern
        if re.search(r'SELECT\s+.+\s+FROM\s+\w+', query_upper):
            structure_score += 0.3
        
        # Proper identifier patterns
        if re.search(r'\b[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\b', query_text):
            structure_score += 0.2
        
        # Semicolon or complete statement
        if query_text.strip().endswith(';') or 'LIMIT' in query_upper:
            structure_score += 0.1
        
        total_confidence = min(confidence + structure_score, 1.0)
        
        # Consider valid if confidence > 0.4 and has basic structure
        is_valid = total_confidence > 0.4 and has_select and has_from
        
        return is_valid, total_confidence
    
    def extract_columns_regex(self, query_text: str, table_name: str) -> Set[str]:
        """Extract column names using rule-based regex patterns"""
        columns = set()
        query_upper = query_text.upper()
        
        try:
            # 1. Extract table.column references
            table_column_pattern = rf'\b(?:{re.escape(table_name.upper())}|[A-Z_]+)\s*\.\s*([a-zA-Z_][a-zA-Z0-9_]*)\b'
            matches = re.findall(table_column_pattern, query_text, re.IGNORECASE)
            columns.update([col.lower() for col in matches])
            
            # 2. Extract from SELECT clause
            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', query_upper, re.DOTALL)
            if select_match:
                select_clause = select_match.group(1)
                if '*' not in select_clause:  # Skip SELECT *
                    # Parse individual columns
                    for item in select_clause.split(','):
                        item = item.strip()
                        # Remove aliases (AS keyword)
                        item = re.sub(r'\s+AS\s+\w+', '', item, flags=re.IGNORECASE)
                        # Extract column name
                        if '.' in item:
                            col_name = item.split('.')[-1].strip()
                        else:
                            col_name = item.strip()
                        
                        # Filter out functions and literals
                        if (col_name and len(col_name) > 1 and 
                            not any(func in col_name.upper() for func in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'CASE']) and
                            not col_name.isdigit() and not col_name.startswith("'")):
                            columns.add(col_name.lower())
            
            # 3. Extract from WHERE clause
            where_matches = re.findall(r'WHERE\s+.*?([a-zA-Z_][a-zA-Z0-9_]*)\s*[=<>!]', query_text, re.IGNORECASE)
            columns.update([col.lower() for col in where_matches])
            
            # 4. Extract from GROUP BY and ORDER BY
            for clause in ['GROUP BY', 'ORDER BY']:
                pattern = rf'{clause}\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*)'
                matches = re.findall(pattern, query_text, re.IGNORECASE)
                for match in matches:
                    for col in match.split(','):
                        col = col.strip().split()[0]  # Remove ASC/DESC
                        if '.' in col:
                            col = col.split('.')[-1]
                        columns.add(col.lower())
            
        except Exception as e:
            self.logger.debug(f"Regex extraction error: {e}")
        
        return columns
    
    async def extract_columns_llm(self, query_text: str, table_name: str) -> Set[str]:
        """Extract column names using LLM for complex cases"""
        
        # Create prompt for GPT-4
        extraction_prompt = f"""You are a SQL expert. Extract all column names referenced in this SQL query for table '{table_name}'.

**SQL Query:**
{query_text}

**Target Table:** {table_name}

**Instructions:**
1. Extract ONLY column names that belong to the target table
2. Ignore functions (COUNT, SUM, etc.), literals, and aliases
3. Return column names in lowercase
4. If the query is invalid/incomplete, return empty list

**Required Output Format (JSON):**
{{
    "is_valid_sql": true/false,
    "extracted_columns": ["column1", "column2", "column3"],
    "confidence": 0.0-1.0
}}

**Examples:**
- "SELECT t.trade_id, t.amount FROM trades t" → ["trade_id", "amount"]
- "WHERE settlement_date > '2024-01-01'" → ["settlement_date"]
- "GROUP BY currency, status" → ["currency", "status"]

**Respond with only the JSON:**"""

        try:
            # Uncomment and use your actual client manager
            # response = await client_manager.ask_gpt(extraction_prompt)
            # result = json.loads(response)
            
            # Mock response for testing - replace with actual LLM call
            result = await self._mock_llm_response(query_text, table_name)
            
            if result.get('is_valid_sql', False):
                return set(result.get('extracted_columns', []))
            else:
                return set()
                
        except Exception as e:
            self.logger.error(f"LLM extraction failed: {e}")
            return set()
    
    async def _mock_llm_response(self, query_text: str, table_name: str) -> Dict[str, Any]:
        """Mock LLM response for testing - replace with actual LLM call"""
        # This is a placeholder - replace with actual client_manager.ask_gpt() call
        
        is_valid, confidence = self.is_valid_sql(query_text)
        
        if is_valid:
            # Use regex extraction as fallback
            columns = self.extract_columns_regex(query_text, table_name)
            return {
                "is_valid_sql": True,
                "extracted_columns": list(columns),
                "confidence": confidence
            }
        else:
            return {
                "is_valid_sql": False,
                "extracted_columns": [],
                "confidence": 0.0
            }
    
    def filter_and_clean_queries(self, log_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and clean query log data, removing noise and invalid entries"""
        
        clean_queries = []
        stats = {
            'total_queries': len(log_data),
            'valid_sql_queries': 0,
            'noisy_queries': 0,
            'empty_queries': 0
        }
        
        for entry in log_data:
            query_text = entry.get('query_text', '').strip()
            
            # Remove pipe delimiters that seem to be artifacts
            query_text = query_text.replace('|', ' ').strip()
            
            if not query_text:
                stats['empty_queries'] += 1
                continue
            
            # Check if it's a valid SQL statement
            is_valid, confidence = self.is_valid_sql(query_text)
            
            if is_valid:
                # Clean up the query
                cleaned_query = self._clean_query_text(query_text)
                entry_copy = entry.copy()
                entry_copy['query_text'] = cleaned_query
                entry_copy['sql_confidence'] = confidence
                clean_queries.append(entry_copy)
                stats['valid_sql_queries'] += 1
            else:
                stats['noisy_queries'] += 1
                self.logger.debug(f"Filtered noisy query: {query_text[:100]}...")
        
        self.logger.info(f"Query filtering stats: {stats}")
        return clean_queries
    
    def _clean_query_text(self, query_text: str) -> str:
        """Clean and normalize query text"""
        # Remove extra whitespace
        query_text = re.sub(r'\s+', ' ', query_text)
        
        # Remove common artifacts
        query_text = re.sub(r'\|+', ' ', query_text)
        
        # Normalize case for keywords while preserving identifiers
        # This is a simplified approach - more sophisticated parsing could be added
        
        return query_text.strip()
    
    async def parse_column_usage_enhanced(self, log_data: List[Dict[str, Any]], table_name: str) -> Dict[str, int]:
        """Enhanced column usage parsing with hybrid approach"""
        
        # First, filter and clean the queries
        clean_queries = self.filter_and_clean_queries(log_data)
        
        self.logger.info(f"Processing {len(clean_queries)} clean queries for {table_name}")
        
        column_usage = Counter()
        llm_usage_count = 0
        regex_usage_count = 0
        
        for entry in clean_queries:
            query_text = entry['query_text']
            confidence = entry.get('sql_confidence', 0.0)
            
            # Decide whether to use regex or LLM based on complexity and confidence
            use_llm = False
            
            # Use LLM for:
            # 1. Complex queries with lower confidence scores
            # 2. Queries with subqueries or complex joins
            # 3. Queries where regex extraction might fail
            if (confidence < 0.7 or 
                'SUBQUERY' in query_text.upper() or 
                'CASE WHEN' in query_text.upper() or
                query_text.count('JOIN') > 1 or
                query_text.count('(') > 2):
                use_llm = True
            
            if use_llm:
                # Use LLM for complex cases
                try:
                    columns = await self.extract_columns_llm(query_text, table_name)
                    llm_usage_count += 1
                except Exception as e:
                    self.logger.warning(f"LLM extraction failed, falling back to regex: {e}")
                    columns = self.extract_columns_regex(query_text, table_name)
                    regex_usage_count += 1
            else:
                # Use regex for simpler cases
                columns = self.extract_columns_regex(query_text, table_name)
                regex_usage_count += 1
            
            # Update usage counts
            for column in columns:
                if column and len(column) > 1:  # Filter out single characters
                    column_usage[column] += 1
        
        self.logger.info(f"Column extraction stats - Regex: {regex_usage_count}, LLM: {llm_usage_count}")
        
        return dict(column_usage)
    
    async def analyze_table_usage_enhanced(self, table_name: str, months_back: int = 6) -> Dict[str, Any]:
        """Enhanced table usage analysis with intelligent query parsing"""
        
        try:
            self.logger.info(f"Starting enhanced analysis for table: {table_name}")
            
            # Step 1: Extract raw query logs
            log_data = self.extract_query_logs(table_name, months_back)
            
            if not log_data:
                self.logger.warning(f"No query logs found for table: {table_name}")
                return self._empty_result(table_name, months_back)
            
            self.logger.info(f"Extracted {len(log_data)} raw log entries")
            
            # Step 2: Enhanced parsing with hybrid approach
            column_usage = await self.parse_column_usage_enhanced(log_data, table_name)
            
            # Step 3: Generate enhanced statistics
            usage_stats = self.generate_enhanced_stats(column_usage, log_data)
            
            self.logger.info(f"Final analysis: {len(column_usage)} columns from {len(log_data)} queries")
            
            return {
                'table_name': table_name,
                'analysis_period_months': months_back,
                'total_queries': len(log_data),
                'valid_sql_queries': usage_stats.get('valid_queries', 0),
                'column_usage_frequency': column_usage,
                'top_columns': self.get_top_columns(column_usage, limit=20),
                'usage_statistics': usage_stats,
                'analysis_method': 'hybrid_rule_llm'
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced analysis failed for {table_name}: {e}")
            return self._empty_result(table_name, months_back)
    
    def generate_enhanced_stats(self, column_usage: Dict[str, int], log_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate enhanced usage statistics"""
        
        total_columns = len(column_usage)
        total_usage = sum(column_usage.values())
        valid_queries = len([entry for entry in log_data if self.is_valid_sql(entry.get('query_text', ''))[0]])
        
        # Calculate column frequency distribution
        usage_values = list(column_usage.values())
        if usage_values:
            usage_distribution = {
                'min_usage': min(usage_values),
                'max_usage': max(usage_values),
                'median_usage': sorted(usage_values)[len(usage_values)//2],
                'top_10_percent_threshold': sorted(usage_values, reverse=True)[min(len(usage_values)//10, len(usage_values)-1)]
            }
        else:
            usage_distribution = {'min_usage': 0, 'max_usage': 0, 'median_usage': 0, 'top_10_percent_threshold': 0}
        
        return {
            'total_unique_columns': total_columns,
            'total_column_references': total_usage,
            'valid_queries': valid_queries,
            'query_success_rate': valid_queries / len(log_data) if log_data else 0,
            'average_usage_per_column': total_usage / total_columns if total_columns > 0 else 0,
            'most_used_column': max(column_usage.items(), key=lambda x: x[1]) if column_usage else None,
            'unique_clients': len(set(entry.get('client_id', '') for entry in log_data)),
            'usage_distribution': usage_distribution
        }
    
    # Keep the existing methods from the original class
    def extract_query_logs(self, table_name: str, months_back: int) -> List[Dict[str, Any]]:
        """Extract query logs from the audit log table (keeping original implementation)"""
        # ... (keep the existing implementation from your original code)
        pass
    
    def test_audit_log_access(self) -> bool:
        """Test audit log access (keeping original implementation)"""
        # ... (keep the existing implementation)
        pass
    
    def get_top_columns(self, column_usage: Dict[str, int], limit: int = 20) -> List[Tuple[str, int]]:
        """Get top N most frequently used columns"""
        return sorted(column_usage.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def _empty_result(self, table_name: str, months_back: int) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            'table_name': table_name,
            'analysis_period_months': months_back,
            'total_queries': 0,
            'valid_sql_queries': 0,
            'column_usage_frequency': {},
            'top_columns': [],
            'usage_statistics': {
                'total_unique_columns': 0,
                'total_column_references': 0,
                'valid_queries': 0,
                'query_success_rate': 0,
                'average_usage_per_column': 0,
                'most_used_column': None,
                'unique_clients': 0
            },
            'analysis_method': 'hybrid_rule_llm'
        }


# Utility functions for integration
async def run_enhanced_analysis_async(analyzer: EnhancedQueryLogAnalyzer, table_name: str, months_back: int = 6):
    """Async wrapper for running enhanced analysis"""
    return await analyzer.analyze_table_usage_enhanced(table_name, months_back)

def integrate_with_existing_processor(processor_instance):
    """Integration function to upgrade existing processor with enhanced analyzer"""
    
    # Replace the existing query analyzer with enhanced version
    processor_instance.query_analyzer = EnhancedQueryLogAnalyzer(processor_instance.db_client)
    
    # Update the analysis method to use async version
    original_analyze_method = processor_instance._analyze_and_update_frequent_columns
    
    def enhanced_analyze_method(tables_list, config_data):
        """Enhanced analysis method with async support"""
        
        async def async_analysis():
            for table_name in tables_list:
                try:
                    # Use enhanced async analysis
                    usage_data = await processor_instance.query_analyzer.analyze_table_usage_enhanced(
                        table_name, months_back=int(config_data.get('LOG_ANALYSIS_MONTHS', '6'))
                    )
                    
                    # Process results same as original
                    if usage_data['total_queries'] > 0:
                        min_usage = int(config_data.get('LOG_ANALYSIS_MIN_USAGE', '5'))
                        discovered_columns = [
                            col for col, count in usage_data['top_columns'] 
                            if count >= min_usage
                        ]
                        
                        # Handle different column modes (keep existing logic)
                        columns_key = f"{table_name}_Columns"
                        columns_value = config_data.get(columns_key, "")
                        
                        if columns_value == '[logs]':
                            config_data[columns_key] = ",".join(discovered_columns[:20])
                            config_data[f"{table_name}_AnalysisMethod"] = "enhanced_logs"
                        
                        # Add enhanced metadata
                        config_data[f"{table_name}_ValidQueries"] = str(usage_data['valid_sql_queries'])
                        config_data[f"{table_name}_QuerySuccessRate"] = f"{usage_data['usage_statistics']['query_success_rate']:.2f}"
                        
                        processor_instance.logger.info(f"Enhanced analysis for {table_name}: {len(discovered_columns)} columns from {usage_data['valid_sql_queries']} valid queries")
                    
                except Exception as e:
                    processor_instance.logger.error(f"Enhanced analysis failed for {table_name}: {e}")
        
        # Run async analysis
        import asyncio
        if asyncio.iscoroutinefunction(async_analysis):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(async_analysis())
            finally:
                loop.close()
        else:
            asyncio.run(async_analysis())
    
    # Replace the method
    processor_instance._analyze_and_update_frequent_columns = enhanced_analyze_method
    
    return processor_instance