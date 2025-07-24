#!/usr/bin/env python3
"""
Simple Test Script for Advanced Graph Text-to-SQL API

This script tests the three main API endpoints with realistic examples.

Usage:
    python test_api.py

Requirements:
    pip install requests
"""

import requests
import json
import time
from datetime import datetime

class SimpleAPITester:
    """Simple tester for the Advanced Graph Text-to-SQL API."""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        print(f"🚀 Testing API at: {self.base_url}")
        print(f"{'='*60}")
    
    def test_health(self):
        """Test the health endpoint."""
        print(f"\n🏥 Testing Health Check")
        print(f"-" * 30)
        
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Service Status: {data.get('status')}")
                print(f"📊 Database Tables: {data.get('database', {}).get('table_count', 'Unknown')}")
                print(f"🔧 Components Ready: {data.get('services', {})}")
            else:
                print(f"❌ Health check failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Health check error: {e}")
    
    def test_text_to_table(self):
        """Test the text-to-table endpoint."""
        print(f"\n🔍 Testing Text-to-Table Endpoint")
        print(f"-" * 40)
        
        test_queries = [
            "show me all trades by government entities",
            "get me the CUSIP that was traded highest last week", 
            "find trades where currencies are different",
            "show me ETD source systems for cash trades"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Test {i}: {query}")
            
            payload = {
                "query": query,
                "user_id": "test_user",
                "max_tables": 5
            }
            
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/text-to-table",
                    json=payload,
                    timeout=30
                )
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    tables = data.get('tables', [])
                    processing_time = data.get('processing_time', 0)
                    
                    print(f"✅ Found {len(tables)} tables in {duration:.2f}s")
                    print(f"📋 Tables: {', '.join(tables[:3])}")
                    if len(tables) > 3:
                        print(f"    ... and {len(tables)-3} more")
                    
                    # Show table details for first table
                    table_details = data.get('table_details', {})
                    if table_details:
                        first_table = list(table_details.keys())[0]
                        details = table_details[first_table]
                        columns = len(details.get('columns', []))
                        print(f"📊 Sample table '{first_table}': {columns} columns, type: {details.get('table_type', 'unknown')}")
                        
                else:
                    print(f"❌ Failed: {response.status_code} - {response.text[:100]}")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def test_table_to_sql(self):
        """Test the table-to-sql endpoint."""
        print(f"\n🔧 Testing Table-to-SQL Endpoint")
        print(f"-" * 40)
        
        test_cases = [
            {
                "selected_tables": ["trades", "counterparty"],
                "original_query": "show me all trades by government entities"
            },
            {
                "selected_tables": ["trades", "product", "cusip_master"],
                "original_query": "get me the CUSIP that was traded with highest volume"
            },
            {
                "selected_tables": ["trades", "currency_ref"],
                "original_query": "find trades where trade currency differs from issue currency"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}: {test_case['original_query']}")
            print(f"📋 Tables: {', '.join(test_case['selected_tables'])}")
            
            payload = {
                "selected_tables": test_case['selected_tables'],
                "original_query": test_case['original_query'],
                "user_id": "test_user"
            }
            
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/table-to-sql",
                    json=payload,
                    timeout=30
                )
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    sql = data.get('sql', '')
                    validation_status = data.get('validation_status', '')
                    
                    print(f"✅ SQL generated in {duration:.2f}s")
                    print(f"🔍 Validation: {validation_status}")
                    print(f"📄 SQL Preview: {sql[:100]}...")
                    
                    # Check if it looks like valid SQL
                    if 'SELECT' in sql.upper():
                        print(f"✅ SQL appears valid")
                    else:
                        print(f"⚠️  SQL may need review")
                        
                else:
                    print(f"❌ Failed: {response.status_code} - {response.text[:100]}")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def test_text_to_sql(self):
        """Test the complete text-to-sql endpoint."""
        print(f"\n🚀 Testing Complete Text-to-SQL Endpoint")
        print(f"-" * 45)
        
        test_queries = [
            {
                "query": "show me all trades by government entities with amounts over 1 million",
                "description": "Government trades with filtering"
            },
            {
                "query": "get me the top 5 CUSIPs by trading volume for last week",
                "description": "Top CUSIP analysis"
            },
            {
                "query": "find all trades where trade price currency is different from issue currency",
                "description": "Currency mismatch analysis"
            }
        ]
        
        for i, test_case in enumerate(test_queries, 1):
            print(f"\n📝 Test {i}: {test_case['description']}")
            print(f"❓ Query: {test_case['query']}")
            
            payload = {
                "query": test_case['query'],
                "user_id": "test_user",
                "max_tables": 5,
                "include_reasoning": True
            }
            
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/text-to-sql",
                    json=payload,
                    timeout=45  # Longer timeout for complete pipeline
                )
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    sql = data.get('sql', '')
                    tables_found = data.get('tables_found', [])
                    reasoning = data.get('reasoning', '')
                    
                    print(f"✅ Complete pipeline finished in {duration:.2f}s")
                    print(f"📊 Tables discovered: {len(tables_found)}")
                    print(f"📋 Tables used: {', '.join(tables_found[:3])}")
                    
                    if sql:
                        print(f"📄 SQL length: {len(sql)} characters")
                        print(f"📄 SQL preview: {sql[:150]}...")
                        
                        # Check SQL quality
                        sql_upper = sql.upper()
                        has_select = 'SELECT' in sql_upper
                        has_from = 'FROM' in sql_upper
                        has_join = 'JOIN' in sql_upper
                        
                        print(f"🔍 SQL Quality Check:")
                        print(f"   SELECT: {'✅' if has_select else '❌'}")
                        print(f"   FROM: {'✅' if has_from else '❌'}")
                        print(f"   JOIN: {'✅' if has_join else '⚠️'}")
                    
                    if reasoning and len(reasoning) > 50:
                        print(f"🧠 Reasoning preview: {reasoning[:100]}...")
                        
                else:
                    print(f"❌ Failed: {response.status_code}")
                    try:
                        error_data = response.json()
                        print(f"❌ Error: {error_data.get('detail', 'Unknown error')}")
                    except:
                        print(f"❌ Error: {response.text[:200]}")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def test_utility_endpoints(self):
        """Test utility endpoints."""
        print(f"\n🛠️ Testing Utility Endpoints")
        print(f"-" * 35)
        
        # Test database info
        print(f"\n📊 Database Info:")
        try:
            response = requests.get(f"{self.base_url}/database/info")
            if response.status_code == 200:
                data = response.json()
                tables_by_type = data.get('tables_by_type', {})
                total_columns = data.get('total_columns', 0)
                
                print(f"✅ Database info retrieved")
                for table_type, count in tables_by_type.items():
                    print(f"   {table_type}: {count} tables")
                print(f"   Total columns: {total_columns}")
            else:
                print(f"❌ Database info failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Database info error: {e}")
        
        # Test table search
        print(f"\n🔍 Table Search:")
        try:
            response = requests.get(f"{self.base_url}/tables/search?query=trade&limit=3")
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                print(f"✅ Found {len(results)} tables matching 'trade':")
                for table in results:
                    print(f"   • {table.get('name')} ({table.get('table_type', 'unknown')})")
            else:
                print(f"❌ Table search failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Table search error: {e}")
        
        # Test SQL validation
        print(f"\n✅ SQL Validation:")
        test_sql = "SELECT * FROM trades WHERE trade_date > '2024-01-01'"
        try:
            response = requests.post(f"{self.base_url}/sql/validate?sql_query={test_sql}")
            if response.status_code == 200:
                data = response.json()
                is_valid = data.get('is_valid', False)
                error = data.get('error')
                print(f"✅ SQL validation: {'Valid' if is_valid else 'Invalid'}")
                if error:
                    print(f"   Error: {error}")
            else:
                print(f"❌ SQL validation failed: {response.status_code}")
        except Exception as e:
            print(f"❌ SQL validation error: {e}")
    
    def run_all_tests(self):
        """Run all tests."""
        print(f"🧪 Starting API Tests")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        # Run tests
        self.test_health()
        self.test_text_to_table()
        self.test_table_to_sql()
        self.test_text_to_sql()
        self.test_utility_endpoints()
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"🏁 All tests completed!")
        print(f"⏱️ Total time: {total_time:.2f} seconds")
        print(f"📝 Summary:")
        print(f"   ✅ If you see mostly green checkmarks, your API is working well!")
        print(f"   ⚠️  Yellow warnings are usually okay")
        print(f"   ❌ Red errors need attention")
        print(f"\n💡 Tip: Check the logs in your API service for more details")

def main():
    """Main function to run tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple API Test Script")
    parser.add_argument(
        "--url", 
        default="http://localhost:8000", 
        help="Base URL of the API (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--test", 
        choices=["health", "text-to-table", "table-to-sql", "text-to-sql", "utilities", "all"],
        default="all",
        help="Specific test to run (default: all)"
    )
    
    args = parser.parse_args()
    
    # Create tester
    tester = SimpleAPITester(args.url)
    
    # Run specific test
    if args.test == "health":
        tester.test_health()
    elif args.test == "text-to-table":
        tester.test_text_to_table()
    elif args.test == "table-to-sql":
        tester.test_table_to_sql()
    elif args.test == "text-to-sql":
        tester.test_text_to_sql()
    elif args.test == "utilities":
        tester.test_utility_endpoints()
    else:
        tester.run_all_tests()

if __name__ == "__main__":
    main()

# === Example Individual Test Functions ===

def quick_test():
    """Quick test for immediate feedback."""
    print("🚀 Quick API Test")
    print("-" * 20)
    
    base_url = "http://localhost:8000"
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running")
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return
    
    # Test 2: Simple text-to-table
    try:
        payload = {
            "query": "show me trades",
            "user_id": "quick_test"
        }
        response = requests.post(f"{base_url}/text-to-table", json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            tables = data.get('tables', [])
            print(f"✅ Found {len(tables)} tables")
        else:
            print(f"❌ Text-to-table failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Text-to-table error: {e}")
    
    print("✅ Quick test completed!")

def test_single_endpoint():
    """Test a single endpoint interactively."""
    print("🎯 Single Endpoint Test")
    print("-" * 25)
    
    base_url = "http://localhost:8000"
    
    # Get user input
    query = input("Enter your query: ")
    if not query:
        query = "show me all trades"
    
    payload = {
        "query": query,
        "user_id": "interactive_test",
        "include_reasoning": True
    }
    
    print(f"\n🔍 Testing query: '{query}'")
    
    try:
        start_time = time.time()
        response = requests.post(f"{base_url}/text-to-sql", json=payload, timeout=30)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Success! ({duration:.2f}s)")
            print(f"📊 Tables found: {data.get('tables_found', [])}")
            print(f"📄 Generated SQL:")
            print(f"   {data.get('sql', 'No SQL generated')}")
            
            reasoning = data.get('reasoning', '')
            if reasoning and len(reasoning) > 10:
                print(f"🧠 Reasoning: {reasoning[:200]}...")
        else:
            print(f"❌ Failed: {response.status_code}")
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

# Allow running individual tests
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            quick_test()
        elif sys.argv[1] == "interactive":
            test_single_endpoint()
        else:
            main()
    else:
        main()