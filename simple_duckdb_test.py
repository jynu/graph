#!/usr/bin/env python3
"""
Simple DuckDB Connection Test
Quick test to isolate the exact connection issue
"""

import sys
import os
import traceback

def test_file_exists():
    """Test if the database file exists and is accessible."""
    db_path = "knowledge_graph.duckdb"
    
    print(f"1. Checking if file exists: {db_path}")
    if os.path.exists(db_path):
        print("   ✓ File exists")
        
        # Get file size
        size = os.path.getsize(db_path)
        print(f"   ✓ File size: {size / (1024*1024):.2f} MB")
        
        # Check permissions
        readable = os.access(db_path, os.R_OK)
        writable = os.access(db_path, os.W_OK)
        print(f"   ✓ Readable: {readable}")
        print(f"   ✓ Writable: {writable}")
        
        return True
    else:
        print("   ✗ File does not exist")
        return False

def test_duckdb_basic():
    """Test basic DuckDB functionality."""
    print("\n2. Testing basic DuckDB functionality...")
    
    try:
        import duckdb
        print(f"   ✓ DuckDB imported: {duckdb.__version__}")
        
        # Test in-memory database
        print("   Testing in-memory database...")
        conn = duckdb.connect(":memory:")
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.execute("INSERT INTO test VALUES (1)")
        result = conn.execute("SELECT * FROM test").fetchone()
        conn.close()
        print(f"   ✓ In-memory test successful: {result}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Basic DuckDB test failed: {e}")
        traceback.print_exc()
        return False

def test_file_connection():
    """Test connection to the actual database file."""
    print("\n3. Testing connection to knowledge_graph.duckdb...")
    
    db_path = "knowledge_graph.duckdb"
    
    try:
        import duckdb
        
        # Try read-only first
        print("   Attempting read-only connection...")
        conn = duckdb.connect(db_path, read_only=True)
        print("   ✓ Read-only connection successful")
        
        # Test simple query
        print("   Testing simple query...")
        result = conn.execute("SELECT 1 as test").fetchone()
        print(f"   ✓ Simple query works: {result}")
        
        conn.close()
        print("   ✓ Read-only connection closed")
        
        return True
        
    except Exception as e:
        print(f"   ✗ File connection failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        traceback.print_exc()
        return False

def test_show_tables():
    """Test SHOW TABLES command."""
    print("\n4. Testing SHOW TABLES...")
    
    db_path = "knowledge_graph.duckdb"
    
    try:
        import duckdb
        
        conn = duckdb.connect(db_path, read_only=True)
        
        print("   Executing SHOW TABLES...")
        tables = conn.execute("SHOW TABLES").fetchall()
        print(f"   ✓ Found {len(tables)} tables")
        
        # Show first few tables
        for i, table in enumerate(tables[:5]):
            print(f"   - {table[0]}")
        
        if len(tables) > 5:
            print(f"   ... and {len(tables) - 5} more tables")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"   ✗ SHOW TABLES failed: {e}")
        traceback.print_exc()
        return False

def test_expected_tables():
    """Test for expected tables (tables, columns)."""
    print("\n5. Testing expected table structure...")
    
    db_path = "knowledge_graph.duckdb"
    
    try:
        import duckdb
        
        conn = duckdb.connect(db_path, read_only=True)
        
        # Get all table names
        all_tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [table[0] for table in all_tables]
        
        print(f"   All tables: {table_names}")
        
        # Check for expected tables
        expected = ['tables', 'columns']
        for expected_table in expected:
            if expected_table in table_names:
                print(f"   ✓ Table '{expected_table}' exists")
                
                # Try to count rows
                try:
                    count = conn.execute(f"SELECT COUNT(*) FROM {expected_table}").fetchone()[0]
                    print(f"     Rows: {count}")
                except Exception as e:
                    print(f"     ✗ Cannot count rows: {e}")
            else:
                print(f"   ✗ Table '{expected_table}' missing")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"   ✗ Table structure test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=== SIMPLE DUCKDB CONNECTION TEST ===")
    
    # Test 1: File exists
    if not test_file_exists():
        print("\n❌ Cannot proceed - database file not found")
        return
    
    # Test 2: Basic DuckDB
    if not test_duckdb_basic():
        print("\n❌ Cannot proceed - DuckDB not working")
        return
    
    # Test 3: File connection
    if not test_file_connection():
        print("\n❌ Cannot connect to database file")
        return
    
    # Test 4: Show tables
    if not test_show_tables():
        print("\n❌ Cannot list tables")
        return
    
    # Test 5: Expected tables
    if not test_expected_tables():
        print("\n⚠️  Database structure may not match expectations")
        return
    
    print("\n✅ ALL TESTS PASSED!")
    print("The database appears to be working correctly.")
    print("Try running your original script again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Test script failed: {e}")
        traceback.print_exc()
        sys.exit(1)