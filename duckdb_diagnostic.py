#!/usr/bin/env python3
"""
Enhanced DuckDB Diagnostic Script
This script provides comprehensive diagnostics for DuckDB connection issues.
"""

import sys
import logging
import os
import stat
import platform
import traceback
from pathlib import Path

# Enhanced logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def system_diagnostics():
    """Comprehensive system diagnostics."""
    logger.info("=== SYSTEM DIAGNOSTICS ===")
    
    # Python version
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Architecture: {platform.architecture()}")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"Available memory: {memory.available / (1024**3):.2f} GB")
        logger.info(f"Total memory: {memory.total / (1024**3):.2f} GB")
    except ImportError:
        logger.warning("psutil not available - cannot check memory")
    
    # Check disk space
    try:
        disk_usage = os.statvfs('.')
        free_space = disk_usage.f_bavail * disk_usage.f_frsize
        logger.info(f"Free disk space: {free_space / (1024**3):.2f} GB")
    except Exception as e:
        logger.warning(f"Cannot check disk space: {e}")

def test_imports_enhanced():
    """Enhanced import testing with version info."""
    logger.info("=== IMPORT DIAGNOSTICS ===")
    
    # Test DuckDB
    try:
        import duckdb
        logger.info(f"✓ DuckDB imported successfully")
        logger.info(f"  Version: {duckdb.__version__}")
        logger.info(f"  File location: {duckdb.__file__}")
        
        # Test basic DuckDB functionality
        try:
            test_conn = duckdb.connect(':memory:')
            test_conn.execute("CREATE TABLE test (id INTEGER)")
            test_conn.execute("INSERT INTO test VALUES (1)")
            result = test_conn.execute("SELECT * FROM test").fetchone()
            test_conn.close()
            logger.info("✓ Basic DuckDB functionality works")
        except Exception as e:
            logger.error(f"✗ Basic DuckDB functionality failed: {e}")
            return False
            
    except ImportError as e:
        logger.error(f"✗ DuckDB import failed: {e}")
        return False
    
    # Test other imports
    try:
        import pandas as pd
        logger.info(f"✓ Pandas: {pd.__version__}")
    except ImportError as e:
        logger.error(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import json
        logger.info("✓ JSON module available")
    except ImportError as e:
        logger.error(f"✗ JSON import failed: {e}")
        return False
    
    return True

def diagnose_database_file(db_path="knowledge_graph.duckdb"):
    """Comprehensive database file diagnostics."""
    logger.info("=== DATABASE FILE DIAGNOSTICS ===")
    
    # Check if file exists
    if not os.path.exists(db_path):
        logger.error(f"✗ Database file does not exist: {db_path}")
        return False
    
    logger.info(f"✓ Database file exists: {db_path}")
    
    # Get file info
    try:
        file_stats = os.stat(db_path)
        logger.info(f"  File size: {file_stats.st_size / (1024**2):.2f} MB")
        logger.info(f"  Last modified: {file_stats.st_mtime}")
        logger.info(f"  File permissions: {oct(file_stats.st_mode)}")
        
        # Check if file is readable
        if os.access(db_path, os.R_OK):
            logger.info("✓ File is readable")
        else:
            logger.error("✗ File is not readable")
            return False
            
        # Check if file is writable (needed for some DuckDB operations)
        if os.access(db_path, os.W_OK):
            logger.info("✓ File is writable")
        else:
            logger.warning("⚠ File is not writable (may cause issues)")
            
    except Exception as e:
        logger.error(f"✗ Cannot get file stats: {e}")
        return False
    
    # Check if file is actually a DuckDB file
    try:
        with open(db_path, 'rb') as f:
            header = f.read(16)
            if header.startswith(b'DUCK'):
                logger.info("✓ File appears to be a valid DuckDB file")
            else:
                logger.warning("⚠ File may not be a valid DuckDB file")
                logger.info(f"  Header: {header}")
    except Exception as e:
        logger.error(f"✗ Cannot read file header: {e}")
        return False
    
    return True

def test_database_connection_detailed(db_path="knowledge_graph.duckdb"):
    """Detailed database connection testing."""
    logger.info("=== DATABASE CONNECTION DIAGNOSTICS ===")
    
    try:
        import duckdb
        
        # Test 1: Try read-only connection
        logger.info("Testing read-only connection...")
        try:
            conn = duckdb.connect(db_path, read_only=True)
            logger.info("✓ Read-only connection successful")
            conn.close()
        except Exception as e:
            logger.error(f"✗ Read-only connection failed: {e}")
            logger.error(f"  Error type: {type(e).__name__}")
            logger.error(f"  Traceback: {traceback.format_exc()}")
            return False
        
        # Test 2: Try regular connection
        logger.info("Testing regular connection...")
        try:
            conn = duckdb.connect(db_path)
            logger.info("✓ Regular connection successful")
            
            # Test basic query
            try:
                result = conn.execute("SELECT 1").fetchone()
                logger.info("✓ Basic query works")
            except Exception as e:
                logger.error(f"✗ Basic query failed: {e}")
                return False
            
            # Test SHOW TABLES
            try:
                tables = conn.execute("SHOW TABLES").fetchall()
                logger.info(f"✓ SHOW TABLES works - found {len(tables)} tables")
                for table in tables[:5]:  # Show first 5 tables
                    logger.info(f"  - {table[0]}")
                if len(tables) > 5:
                    logger.info(f"  ... and {len(tables) - 5} more tables")
            except Exception as e:
                logger.error(f"✗ SHOW TABLES failed: {e}")
                logger.error(f"  This suggests the database may be corrupted")
                return False
            
            conn.close()
            logger.info("✓ Connection closed successfully")
            
        except Exception as e:
            logger.error(f"✗ Regular connection failed: {e}")
            logger.error(f"  Error type: {type(e).__name__}")
            logger.error(f"  Traceback: {traceback.format_exc()}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Connection test failed: {e}")
        return False

def test_specific_table_structure(db_path="knowledge_graph.duckdb"):
    """Test the specific table structure expected by the script."""
    logger.info("=== TABLE STRUCTURE DIAGNOSTICS ===")
    
    try:
        import duckdb
        conn = duckdb.connect(db_path, read_only=True)
        
        # Check for expected tables
        expected_tables = ['tables', 'columns']
        
        all_tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [table[0] for table in all_tables]
        
        logger.info(f"All tables in database: {table_names}")
        
        for expected_table in expected_tables:
            if expected_table in table_names:
                logger.info(f"✓ Expected table '{expected_table}' exists")
                
                # Check table structure
                try:
                    columns = conn.execute(f"PRAGMA table_info('{expected_table}')").fetchall()
                    logger.info(f"  Columns: {[col[1] for col in columns]}")
                    
                    # Check row count
                    count = conn.execute(f"SELECT COUNT(*) FROM {expected_table}").fetchone()[0]
                    logger.info(f"  Row count: {count}")
                    
                except Exception as e:
                    logger.error(f"✗ Cannot analyze table '{expected_table}': {e}")
            else:
                logger.warning(f"⚠ Expected table '{expected_table}' not found")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"✗ Table structure test failed: {e}")
        return False

def suggest_solutions():
    """Suggest potential solutions based on common issues."""
    logger.info("=== POTENTIAL SOLUTIONS ===")
    
    solutions = [
        "1. PERMISSIONS: Ensure the database file has proper read/write permissions",
        "2. DISK SPACE: Check if there's sufficient disk space (DuckDB may need temp space)",
        "3. MEMORY: Ensure sufficient RAM is available",
        "4. FILE CORRUPTION: Try copying the database file again from your colleague",
        "5. DUCKDB VERSION: Ensure you're using a compatible DuckDB version",
        "6. ANTIVIRUS: Check if antivirus software is blocking database access",
        "7. NETWORK DRIVE: If the file is on a network drive, copy it locally",
        "8. CONCURRENT ACCESS: Ensure no other process is accessing the database",
        "9. CORPORATE RESTRICTIONS: Check if corporate security policies block database access"
    ]
    
    for solution in solutions:
        logger.info(solution)

def main():
    """Main diagnostic function."""
    logger.info("=== COMPREHENSIVE DUCKDB DIAGNOSTICS ===")
    
    # Run all diagnostics
    system_diagnostics()
    
    if not test_imports_enhanced():
        logger.error("Import tests failed. Cannot proceed.")
        return
    
    db_path = "knowledge_graph.duckdb"
    
    if not diagnose_database_file(db_path):
        logger.error("Database file diagnostics failed.")
        suggest_solutions()
        return
    
    if not test_database_connection_detailed(db_path):
        logger.error("Database connection failed.")
        suggest_solutions()
        return
    
    if not test_specific_table_structure(db_path):
        logger.error("Table structure validation failed.")
        suggest_solutions()
        return
    
    logger.info("=== ALL DIAGNOSTICS PASSED ===")
    logger.info("The database appears to be working correctly!")
    logger.info("Try running your original script again.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Diagnostic script failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)