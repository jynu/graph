# getFrequentColumn.py
import subprocess
import os
import logging
from typing import List, Optional

def getFrequentColumn(table_name: str, host: str, user_id: str, password: str, 
                     jdk_path: Optional[str] = None, jar_path: Optional[str] = None) -> List[str]:
    """
    Get frequently used columns for a table by analyzing column usage patterns.
    This is a placeholder implementation that would typically analyze:
    - Column selectivity
    - Query frequency
    - Data completeness
    - Business importance
    """
    
    logger = logging.getLogger(__name__)
    
    # Set default paths if not provided
    if jdk_path is None:
        if os.name == 'nt':  # Windows
            jdk_path = r"C:/Users/AK06306/AppData/Local/CitiSoftware/CTC2174129_JDK_17.0_15W64/bin/java.exe"
        else:  # Linux
            jdk_path = "/opt/jdk/17.0_9l64/bin/java"
    
    if jar_path is None:
        if os.name == 'nt':  # Windows
            jar_path = r"C:/Users/AK06306/Downloads/mktdata-report-hvapi-commandline-client-1.0.30-20250519.150013-4.jar"
        else:  # Linux
            jar_path = "/home/bj33244/mktdata-report-hvapi-commandline-client-1.0.27-SNAPSHOT.jar"
    
    try:
        # Step 1: Get table schema to identify all columns
        columns = _get_all_columns(table_name, host, user_id, password, jdk_path, jar_path)
        
        if not columns:
            logger.warning(f"No columns found for table {table_name}")
            return []
        
        # Step 2: Analyze column frequency/importance
        frequent_columns = _analyze_column_importance(
            table_name, columns, host, user_id, password, jdk_path, jar_path
        )
        
        logger.info(f"Generated {len(frequent_columns)} frequent columns for table {table_name}")
        return frequent_columns
        
    except Exception as e:
        logger.error(f"Error getting frequent columns for {table_name}: {e}")
        # Return a default set of commonly important columns
        return _get_default_columns(table_name)


def _get_all_columns(table_name: str, host: str, user_id: str, password: str, 
                    jdk_path: str, jar_path: str) -> List[str]:
    """Get all columns from table schema"""
    
    import tempfile
    
    # Create temporary file for schema output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.schema', delete=False) as temp_file:
        schema_file = temp_file.name
    
    try:
        # Query to describe table
        query = f"--query=describe {table_name}"
        
        command = [
            jdk_path,
            "--add-opens=java.base/java.nio=ALL-UNNAMED",
            "-jar", jar_path,
            host,
            query,
            f"--user={user_id}",
            f"--pass={password}",
            "--env=prod",
            "--format=TXT",
            f"--destination={schema_file}"
        ]
        
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            raise Exception(f"Schema query failed: {result.stderr}")
        
        # Parse schema file to extract column names
        columns = []
        if os.path.exists(schema_file):
            with open(schema_file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if i == 0:  # Skip header
                        continue
                    parts = line.split(',', 1)
                    if len(parts) >= 1:
                        column_name = parts[0].strip().lower()
                        if column_name:
                            columns.append(column_name)
        
        return columns
        
    finally:
        # Clean up temporary file
        if os.path.exists(schema_file):
            os.unlink(schema_file)


def _analyze_column_importance(table_name: str, columns: List[str], host: str, 
                             user_id: str, password: str, jdk_path: str, jar_path: str) -> List[str]:
    """
    Analyze column importance based on various criteria.
    This is a simplified implementation that could be enhanced with:
    - Statistical analysis of data distribution
    - Query log analysis
    - Business rule evaluation
    - Data quality metrics
    """
    
    logger = logging.getLogger(__name__)
    important_columns = []
    
    # Priority 1: Always include common important columns
    priority_keywords = [
        'id', 'key', 'code', 'type', 'status', 'date', 'time', 'amount', 
        'price', 'quantity', 'name', 'description', 'currency', 'account'
    ]
    
    for column in columns:
        # Include columns that match priority keywords
        if any(keyword in column.lower() for keyword in priority_keywords):
            important_columns.append(column)
            continue
        
        # Analyze column data characteristics
        try:
            column_score = _calculate_column_score(
                table_name, column, host, user_id, password, jdk_path, jar_path
            )
            
            # Include columns with high importance scores
            if column_score > 0.6:  # Threshold for importance
                important_columns.append(column)
                
        except Exception as e:
            logger.warning(f"Could not analyze column {column}: {e}")
    
    # Ensure we have at least some columns
    if not important_columns:
        # Take first 10 columns as fallback
        important_columns = columns[:10]
    
    # Limit to reasonable number of columns (max 20)
    return important_columns[:20]


def _calculate_column_score(table_name: str, column_name: str, host: str, 
                          user_id: str, password: str, jdk_path: str, jar_path: str) -> float:
    """
    Calculate importance score for a column based on:
    - Data completeness (non-null percentage)
    - Data uniqueness (distinct value ratio)
    - Data type importance
    """
    
    import tempfile
    
    try:
        # Query to get column statistics
        query = f"""--query=SELECT 
            COUNT(*) as total_rows,
            COUNT({column_name}) as non_null_rows,
            COUNT(DISTINCT {column_name}) as distinct_values
        FROM {table_name} 
        LIMIT 100000"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.stats', delete=False) as temp_file:
            stats_file = temp_file.name
        
        command = [
            jdk_path,
            "--add-opens=java.base/java.nio=ALL-UNNAMED",
            "-jar", jar_path,
            host,
            query,
            f"--user={user_id}",
            f"--pass={password}",
            "--env=prod",
            "--format=TXT",
            f"--destination={stats_file}"
        ]
        
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0 or not os.path.exists(stats_file):
            return 0.5  # Default score if analysis fails
        
        # Parse statistics
        with open(stats_file, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 2:
                stats_line = lines[1].strip().split(',')
                if len(stats_line) >= 3:
                    total_rows = int(stats_line[0])
                    non_null_rows = int(stats_line[1])
                    distinct_values = int(stats_line[2])
                    
                    # Calculate scores
                    completeness_score = non_null_rows / total_rows if total_rows > 0 else 0
                    uniqueness_score = min(distinct_values / total_rows, 1.0) if total_rows > 0 else 0
                    
                    # Weighted importance score
                    final_score = (completeness_score * 0.4) + (uniqueness_score * 0.6)
                    return final_score
        
        os.unlink(stats_file)
        return 0.5  # Default score
        
    except Exception:
        return 0.5  # Default score if any error occurs


def _get_default_columns(table_name: str) -> List[str]:
    """
    Return default column set based on table name patterns.
    This is a fallback when automatic analysis fails.
    """
    
    table_lower = table_name.lower()
    
    # Common patterns for different table types
    if 'trade' in table_lower or 'transaction' in table_lower:
        return [
            'trade_id', 'transaction_id', 'trade_date', 'settlement_date',
            'instrument_id', 'quantity', 'price', 'amount', 'currency',
            'counterparty', 'trader_id', 'book_id', 'status'
        ]
    
    elif 'position' in table_lower:
        return [
            'position_id', 'account_id', 'instrument_id', 'quantity',
            'market_value', 'book_value', 'currency', 'as_of_date',
            'portfolio_id', 'strategy', 'risk_factor'
        ]
    
    elif 'risk' in table_lower:
        return [
            'risk_id', 'portfolio_id', 'risk_type', 'risk_amount',
            'var_amount', 'stress_test_result', 'as_of_date',
            'currency', 'confidence_level', 'time_horizon'
        ]
    
    elif 'market' in table_lower or 'price' in table_lower:
        return [
            'instrument_id', 'price_date', 'price_type', 'price_value',
            'currency', 'source', 'bid_price', 'ask_price',
            'volume', 'last_updated'
        ]
    
    elif 'reference' in table_lower or 'ref' in table_lower:
        return [
            'id', 'code', 'name', 'description', 'type',
            'status', 'effective_date', 'expiry_date',
            'created_date', 'last_updated'
        ]
    
    else:
        # Generic default columns
        return [
            'id', 'name', 'type', 'status', 'date',
            'amount', 'currency', 'description', 'created_date',
            'last_updated'
        ]