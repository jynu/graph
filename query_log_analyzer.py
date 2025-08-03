# query_log_analyzer.py
import re
import configparser
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import logging
from collections import Counter

class QueryLogAnalyzer:
    """Analyzes database query logs to identify most frequently used columns"""
    
    def __init__(self, db_client):
        self.db_client = db_client
        self.logger = logging.getLogger(__name__)
    
    def analyze_table_usage(self, table_name: str, months_back: int = 6) -> Dict[str, Any]:
        """
        Analyze query logs for a specific table and return usage statistics
        """
        try:
            self.logger.info(f"Analyzing query logs for table: {table_name}")
            
            # Step 1: Extract query logs
            log_data = self.extract_query_logs(table_name, months_back)
            
            if not log_data:
                self.logger.warning(f"No query logs found for table: {table_name}")
                return self._empty_result(table_name, months_back)
            
            # Step 2: Parse and analyze column usage
            column_usage = self.parse_column_usage(log_data, table_name)
            
            # Step 3: Generate statistics
            usage_stats = self.generate_usage_stats(column_usage, log_data)
            
            self.logger.info(f"Found {len(column_usage)} columns in {len(log_data)} queries for {table_name}")
            
            return {
                'table_name': table_name,
                'analysis_period_months': months_back,
                'total_queries': len(log_data),
                'column_usage_frequency': column_usage,
                'top_columns': self.get_top_columns(column_usage, limit=20),
                'usage_statistics': usage_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing table usage for {table_name}: {e}")
            return self._empty_result(table_name, months_back)
    
    def extract_query_logs(self, table_name: str, months_back: int) -> List[Dict[str, Any]]:
        """Extract query logs from the audit log table"""
        
        # Parse schema and table name
        if '.' in table_name:
            schema_name, table_only = table_name.split('.', 1)
        else:
            schema_name = ''
            table_only = table_name
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months_back * 30)
        
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
            AND dc_log.query_text LIKE '%{table_name}%'
            AND dc_log.client_id NOT LIKE '%fid%'
            AND dc_log.exception_code IS NULL
        LIMIT 10000
        """
        
        # Execute query to get logs
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
                temp_output = temp_file.name
            
            success = self.db_client.execute_query(log_query, temp_output, "TXT")
            
            if success:
                return self._parse_log_file(temp_output)
            else:
                self.logger.error("Failed to extract query logs")
                return []
                
        except Exception as e:
            self.logger.error(f"Error extracting query logs: {e}")
            return []
    
    def _parse_log_file(self, log_file: str) -> List[Dict[str, Any]]:
        """Parse the log file and return structured data"""
        log_data = []
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines):
                if i == 0:  # Skip header
                    continue
                
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    log_data.append({
                        'dwh_business_date': parts[0],
                        'schema_name': parts[1],
                        'table_name': parts[2],
                        'client_id': parts[3],
                        'query_text': parts[4],
                        'status': parts[5]
                    })
            
            return log_data
            
        except Exception as e:
            self.logger.error(f"Error parsing log file: {e}")
            return []
    
    def parse_column_usage(self, log_data: List[Dict[str, Any]], table_name: str) -> Dict[str, int]:
        """Parse SQL queries to count column usage frequency"""
        
        column_usage = Counter()
        
        for log_entry in log_data:
            query_text = log_entry['query_text'].upper()
            
            # Clean up the query text
            query_text = query_text.replace('|', ' ')
            
            # Extract columns from different SQL clauses
            columns_found = []
            
            # 1. Extract from SELECT clause
            select_columns = self.extract_select_columns(query_text, table_name)
            columns_found.extend(select_columns)
            
            # 2. Extract from WHERE clause
            where_columns = self.extract_where_columns(query_text, table_name)
            columns_found.extend(where_columns)
            
            # 3. Extract from GROUP BY clause
            groupby_columns = self.extract_groupby_columns(query_text, table_name)
            columns_found.extend(groupby_columns)
            
            # 4. Extract from ORDER BY clause
            orderby_columns = self.extract_orderby_columns(query_text, table_name)
            columns_found.extend(orderby_columns)
            
            # Count usage
            for column in columns_found:
                if column and len(column) > 1:  # Filter out single characters
                    column_usage[column.lower()] += 1
        
        return dict(column_usage)
    
    def extract_select_columns(self, query_text: str, table_name: str) -> List[str]:
        """Extract column names from SELECT clause"""
        columns = []
        
        try:
            # Pattern to match SELECT ... FROM
            select_pattern = r'SELECT\s+(.*?)\s+FROM'
            match = re.search(select_pattern, query_text, re.IGNORECASE | re.DOTALL)
            
            if match:
                select_clause = match.group(1)
                
                # Skip if SELECT *
                if '*' in select_clause:
                    return columns
                
                # Parse individual columns
                for item in select_clause.split(','):
                    item = item.strip()
                    
                    # Handle table.column references
                    if '.' in item:
                        parts = item.split('.')
                        if len(parts) >= 2:
                            column_part = parts[-1].strip()
                            # Remove aliases (AS keyword)
                            column_part = re.sub(r'\s+AS\s+\w+', '', column_part, flags=re.IGNORECASE)
                            columns.append(column_part)
                    else:
                        # Direct column reference
                        # Remove aliases and functions
                        item = re.sub(r'\s+AS\s+\w+', '', item, flags=re.IGNORECASE)
                        if not any(func in item.upper() for func in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'CASE', 'WHEN']):
                            columns.append(item.strip())
            
        except Exception as e:
            self.logger.debug(f"Error extracting SELECT columns: {e}")
        
        return columns
    
    def extract_where_columns(self, query_text: str, table_name: str) -> List[str]:
        """Extract column names from WHERE clause"""
        columns = []
        
        try:
            # Pattern to match WHERE clause
            where_pattern = r'WHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|$)'
            match = re.search(where_pattern, query_text, re.IGNORECASE | re.DOTALL)
            
            if match:
                where_clause = match.group(1)
                
                # Extract column references (simple approach)
                # Look for patterns like column_name = value, column_name IN, etc.
                column_patterns = [
                    r'(\w+)\s*[=<>!]+',  # column = value
                    r'(\w+)\s+IN\s*\(',  # column IN (...)
                    r'(\w+)\s+LIKE',     # column LIKE
                    r'(\w+)\s+BETWEEN',  # column BETWEEN
                    r'(\w+)\s+IS\s+',    # column IS NULL
                ]
                
                for pattern in column_patterns:
                    matches = re.findall(pattern, where_clause, re.IGNORECASE)
                    columns.extend(matches)
            
        except Exception as e:
            self.logger.debug(f"Error extracting WHERE columns: {e}")
        
        return columns
    
    def extract_groupby_columns(self, query_text: str, table_name: str) -> List[str]:
        """Extract column names from GROUP BY clause"""
        columns = []
        
        try:
            group_pattern = r'GROUP\s+BY\s+(.*?)(?:\s+ORDER\s+BY|\s+HAVING|\s+LIMIT|$)'
            match = re.search(group_pattern, query_text, re.IGNORECASE | re.DOTALL)
            
            if match:
                group_clause = match.group(1)
                for item in group_clause.split(','):
                    item = item.strip()
                    if '.' in item:
                        column_part = item.split('.')[-1].strip()
                        columns.append(column_part)
                    else:
                        columns.append(item)
            
        except Exception as e:
            self.logger.debug(f"Error extracting GROUP BY columns: {e}")
        
        return columns
    
    def extract_orderby_columns(self, query_text: str, table_name: str) -> List[str]:
        """Extract column names from ORDER BY clause"""
        columns = []
        
        try:
            order_pattern = r'ORDER\s+BY\s+(.*?)(?:\s+LIMIT|$)'
            match = re.search(order_pattern, query_text, re.IGNORECASE | re.DOTALL)
            
            if match:
                order_clause = match.group(1)
                for item in order_clause.split(','):
                    item = item.strip()
                    # Remove ASC/DESC
                    item = re.sub(r'\s+(ASC|DESC)', '', item, flags=re.IGNORECASE)
                    if '.' in item:
                        column_part = item.split('.')[-1].strip()
                        columns.append(column_part)
                    else:
                        columns.append(item)
            
        except Exception as e:
            self.logger.debug(f"Error extracting ORDER BY columns: {e}")
        
        return columns
    
    def generate_usage_stats(self, column_usage: Dict[str, int], log_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate usage statistics"""
        
        total_columns = len(column_usage)
        total_usage = sum(column_usage.values())
        
        return {
            'total_unique_columns': total_columns,
            'total_column_references': total_usage,
            'average_usage_per_column': total_usage / total_columns if total_columns > 0 else 0,
            'most_used_column': max(column_usage.items(), key=lambda x: x[1]) if column_usage else None,
            'unique_clients': len(set(entry['client_id'] for entry in log_data))
        }
    
    def get_top_columns(self, column_usage: Dict[str, int], limit: int = 20) -> List[Tuple[str, int]]:
        """Get top N most frequently used columns"""
        return sorted(column_usage.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def _empty_result(self, table_name: str, months_back: int) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            'table_name': table_name,
            'analysis_period_months': months_back,
            'total_queries': 0,
            'column_usage_frequency': {},
            'top_columns': [],
            'usage_statistics': {
                'total_unique_columns': 0,
                'total_column_references': 0,
                'average_usage_per_column': 0,
                'most_used_column': None,
                'unique_clients': 0
            }
        }