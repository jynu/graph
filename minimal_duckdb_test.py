#!/usr/bin/env python3
"""
Minimal DuckDB Test - Step by step isolation
"""

import sys
import signal
import time

def timeout_handler(signum, frame):
    print("\n❌ TIMEOUT: Operation took too long!")
    raise TimeoutError("Operation timed out")

def test_with_timeout(test_func, timeout_seconds=10):
    """Run a test with a timeout."""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        result = test_func()
        signal.alarm(0)  # Cancel the alarm
        return result
    except TimeoutError:
        print(f"   ✗ Test timed out after {timeout_seconds} seconds")
        return False
    except Exception as e:
        signal.alarm(0)  # Cancel the alarm
        print(f"   ✗ Test failed: {e}")
        return False

def step1_import_duckdb():
    """Step 1: Just import DuckDB"""
    print("Step 1: Importing DuckDB...")
    import duckdb
    print(f"   ✓ Import successful: {duckdb.__version__}")
    return True

def step2_create_memory_connection():
    """Step 2: Create in-memory connection"""
    print("Step 2: Creating in-memory connection...")
    import duckdb
    conn = duckdb.connect(":memory:")
    print("   ✓ Memory connection created")
    conn.close()
    print("   ✓ Memory connection closed")
    return True

def step3_simple_query():
    """Step 3: Execute simple query"""
    print("Step 3: Executing simple query...")
    import duckdb
    conn = duckdb.connect(":memory:")
    result = conn.execute("SELECT 1 as test").fetchone()
    print(f"   ✓ Query result: {result}")
    conn.close()
    return True

def step4_create_table():
    """Step 4: Create and insert into table"""
    print("Step 4: Creating table and inserting data...")
    import duckdb
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE test (id INTEGER)")
    print("   ✓ Table created")
    conn.execute("INSERT INTO test VALUES (1)")
    print("   ✓ Data inserted")
    result = conn.execute("SELECT * FROM test").fetchone()
    print(f"   ✓ Data retrieved: {result}")
    conn.close()
    return True

def step5_file_connection():
    """Step 5: Connect to actual file"""
    print("Step 5: Connecting to knowledge_graph.duckdb...")
    import duckdb
    conn = duckdb.connect("knowledge_graph.duckdb", read_only=True)
    print("   ✓ File connection successful")
    conn.close()
    print("   ✓ File connection closed")
    return True

def main():
    """Run tests step by step."""
    print("=== MINIMAL DUCKDB TEST ===")
    print("Testing each step with 10-second timeout...\n")
    
    steps = [
        ("Import DuckDB", step1_import_duckdb),
        ("Memory Connection", step2_create_memory_connection),
        ("Simple Query", step3_simple_query),
        ("Create Table", step4_create_table),
        ("File Connection", step5_file_connection)
    ]
    
    for step_name, step_func in steps:
        print(f"\n--- {step_name} ---")
        
        if sys.platform == "win32":
            # Windows doesn't have signal.SIGALRM, so we'll run without timeout
            try:
                success = step_func()
                if success:
                    print(f"✅ {step_name} PASSED")
                else:
                    print(f"❌ {step_name} FAILED")
                    break
            except Exception as e:
                print(f"❌ {step_name} FAILED: {e}")
                import traceback
                traceback.print_exc()
                break
        else:
            # Unix-like systems can use timeout
            success = test_with_timeout(step_func, 10)
            if success:
                print(f"✅ {step_name} PASSED")
            else:
                print(f"❌ {step_name} FAILED")
                break
    
    print("\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user (Ctrl+C)")
        print("This suggests DuckDB might be hanging.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()