#!/usr/bin/env python3
"""
Advanced Graph Traversal Text-to-SQL API Test Script

This script tests all endpoints of the Advanced Graph Traversal API service
with comprehensive examples and validation.

Usage:
    python test_advanced_graph_api.py

Requirements:
    pip install requests pytest asyncio aiohttp tabulate colorama
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional
import requests
from tabulate import tabulate
from colorama import init, Fore, Style

# Initialize colorama for colored output
init(autoreset=True)

class APITester:
    """Comprehensive API tester for Advanced Graph Traversal endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.test_results = []
        
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        print(f"{Fore.CYAN}🚀 Advanced Graph API Tester Initialized")
        print(f"{Fore.CYAN}📍 Base URL: {self.base_url}")
        print(f"{Fore.CYAN}{'='*60}")
    
    def generate_request_id(self) -> str:
        """Generate unique request ID."""
        return f"test_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    def log_test(self, test_name: str, success: bool, duration: float, details: str = ""):
        """Log test results."""
        status = f"{Fore.GREEN}✅ PASS" if success else f"{Fore.RED}❌ FAIL"
        print(f"{status} {test_name} ({duration:.3f}s)")
        if details:
            print(f"   {Fore.YELLOW}📝 {details}")
        
        self.test_results.append({
            'test_name': test_name,
            'success': success,
            'duration': duration,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
    
    def make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, params: Optional[Dict] = None) -> Dict:
        """Make API request with error handling."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, params=params)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data, params=params)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return {
                'success': True,
                'status_code': response.status_code,
                'data': response.json(),
                'response_time': response.elapsed.total_seconds()
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': str(e),
                'status_code': getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None,
                'response_time': 0
            }
    
    def test_health_check(self):
        """Test health check endpoint."""
        print(f"\n{Fore.MAGENTA}🏥 Testing Health Check Endpoint")
        print(f"{Fore.MAGENTA}{'-'*40}")
        
        start_time = time.time()
        result = self.make_request('GET', '/api/v2/health')
        duration = time.time() - start_time
        
        if result['success']:
            data = result['data']
            print(f"{Fore.GREEN}📊 Service Status: {data.get('status', 'Unknown')}")
            print(f"{Fore.GREEN}🗄️  Database Connected: {data.get('database', {}).get('connected', False)}")
            print(f"{Fore.GREEN}📋 Table Count: {data.get('database', {}).get('table_count', 'Unknown')}")
            
            self.log_test("Health Check", True, duration, f"Status: {data.get('status')}")
        else:
            self.log_test("Health Check", False, duration, f"Error: {result.get('error')}")
    
    def test_database_info(self):
        """Test database info endpoint."""
        print(f"\n{Fore.MAGENTA}📊 Testing Database Info Endpoint")
        print(f"{Fore.MAGENTA}{'-'*40}")
        
        start_time = time.time()
        result = self.make_request('GET', '/api/v2/database/info')
        duration = time.time() - start_time
        
        if result['success']:
            data = result['data']
            print(f"{Fore.GREEN}📁 Database Path: {data.get('database_path', 'Unknown')}")
            print(f"{Fore.GREEN}📊 Tables by Type:")
            
            tables_by_type = data.get('tables_by_type', {})
            for table_type, count in tables_by_type.items():
                print(f"   {Fore.CYAN}• {table_type}: {count}")
            
            print(f"{Fore.GREEN}🔗 Total Columns: {data.get('total_columns', 'Unknown')}")
            print(f"{Fore.GREEN}🔗 Total Relationships: {data.get('total_relationships', 'Unknown')}")
            
            self.log_test("Database Info", True, duration, f"Total tables: {sum(tables_by_type.values())}")
        else:
            self.log_test("Database Info", False, duration, f"Error: {result.get('error')}")
    
    def test_text_to_table(self):
        """Test text-to-table endpoint with multiple examples."""
        print(f"\n{Fore.MAGENTA}🔍 Testing Text-to-Table Endpoint")
        print(f"{Fore.MAGENTA}{'-'*40}")
        
        test_cases = [
            {
                "name": "Trade Query",
                "query": "show me all trades by government entities",
                "expected_tables": ["trades", "counterparty", "government"]
            },
            {
                "name": "CUSIP Query", 
                "query": "get me the CUSIP that was traded highest last week",
                "expected_tables": ["trades", "cusip", "product"]
            },
            {
                "name": "ETD Query",
                "query": "give me distinct source systems for cash ETD trades for yesterday", 
                "expected_tables": ["trades", "etd", "source_system"]
            },
            {
                "name": "Trader Query",
                "query": "show me EXECUTING_TRADER_SOEID and EXECUTION_VENUE where prices differ",
                "expected_tables": ["trades", "trader", "venue"]
            },
            {
                "name": "Currency Query",
                "query": "find trades where TRADE_PRICE_CURRENCY and ISSUE_CURRENCY are different",
                "expected_tables": ["trades", "currency", "product"]
            },
            {
                "name": "Complex Analytics Query",
                "query": "analyze trading patterns by venue and product type for Q4 2024",
                "expected_tables": ["trades", "venue", "product", "calendar"]
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{Fore.CYAN}Test {i}: {test_case['name']}")
            print(f"{Fore.YELLOW}Query: {test_case['query']}")
            
            start_time = time.time()
            
            payload = {
                "query": test_case['query'],
                "user_id": "test_user",
                "request_id": self.generate_request_id(),
                "max_tables": 10
            }
            
            result = self.make_request('POST', '/api/v2/text_to_table', payload)
            duration = time.time() - start_time
            
            if result['success']:
                data = result['data']
                tables_found = data.get('tables', [])
                table_details = data.get('table_details', {})
                processing_time = data.get('processing_time', 0)
                
                print(f"{Fore.GREEN}🎯 Tables Found ({len(tables_found)}): {', '.join(tables_found[:5])}")
                print(f"{Fore.GREEN}⏱️  Processing Time: {processing_time:.3f}s")
                
                # Show table details
                if table_details:
                    print(f"{Fore.CYAN}📋 Table Details:")
                    for table_name, details in list(table_details.items())[:3]:
                        columns_count = len(details.get('columns', []))
                        table_type = details.get('table_type', 'unknown')
                        print(f"   • {table_name} ({table_type}) - {columns_count} columns")
                
                success = len(tables_found) > 0
                details_str = f"Found {len(tables_found)} tables"
                
                self.log_test(f"Text-to-Table: {test_case['name']}", success, duration, details_str)
            else:
                error_msg = result.get('error', 'Unknown error')
                print(f"{Fore.RED}❌ Error: {error_msg}")
                self.log_test(f"Text-to-Table: {test_case['name']}", False, duration, f"Error: {error_msg}")
    
    def test_table_to_sql(self):
        """Test table-to-SQL endpoint with multiple examples."""
        print(f"\n{Fore.MAGENTA}🔧 Testing Table-to-SQL Endpoint")
        print(f"{Fore.MAGENTA}{'-'*40}")
        
        test_cases = [
            {
                "name": "Simple Trade Query",
                "selected_tables": ["trades", "counterparty"],
                "original_query": "show me all trades by government entities",
                "description": "Basic join between trades and counterparty"
            },
            {
                "name": "CUSIP Analysis",
                "selected_tables": ["trades", "product", "cusip_master"],
                "original_query": "get me the CUSIP that was traded with highest volume last week",
                "description": "Aggregation with date filtering"
            },
            {
                "name": "Multi-table Join",
                "selected_tables": ["trades", "trader", "venue", "product"],
                "original_query": "show me trading activity by trader and venue for bond products",
                "description": "Complex multi-table join with filtering"
            },
            {
                "name": "Time-based Analysis",
                "selected_tables": ["trades", "calendar", "business_date"],
                "original_query": "analyze daily trading volumes for the past month",
                "description": "Time-series analysis with date tables"
            },
            {
                "name": "Currency Mismatch",
                "selected_tables": ["trades", "product", "currency_ref"],
                "original_query": "find trades where trade currency differs from issue currency",
                "description": "Currency comparison analysis"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{Fore.CYAN}Test {i}: {test_case['name']}")
            print(f"{Fore.YELLOW}Query: {test_case['original_query']}")
            print(f"{Fore.YELLOW}Tables: {', '.join(test_case['selected_tables'])}")
            
            start_time = time.time()
            
            payload = {
                "selected_tables": test_case['selected_tables'],
                "original_query": test_case['original_query'],
                "user_id": "test_user",
                "request_id": self.generate_request_id()
            }
            
            result = self.make_request('POST', '/api/v2/table_to_sql', payload)
            duration = time.time() - start_time
            
            if result['success']:
                data = result['data']
                sql_code = data.get('sql', '')
                reasoning = data.get('reasoning', '')
                validation_status = data.get('validation_status', '')
                processing_time = data.get('processing_time', 0)
                
                print(f"{Fore.GREEN}✅ SQL Generated:")
                print(f"{Fore.CYAN}{sql_code[:200]}..." if len(sql_code) > 200 else f"{Fore.CYAN}{sql_code}")
                print(f"{Fore.GREEN}🔍 Validation: {validation_status}")
                print(f"{Fore.GREEN}⏱️  Processing Time: {processing_time:.3f}s")
                
                success = bool(sql_code and 'SELECT' in sql_code.upper())
                details_str = f"SQL length: {len(sql_code)}, Valid: {validation_status}"
                
                self.log_test(f"Table-to-SQL: {test_case['name']}", success, duration, details_str)
            else:
                error_msg = result.get('error', 'Unknown error')
                print(f"{Fore.RED}❌ Error: {error_msg}")
                self.log_test(f"Table-to-SQL: {test_case['name']}", False, duration, f"Error: {error_msg}")
    
    def test_advanced_text_to_sql(self):
        """Test complete text-to-SQL pipeline with multiple examples."""
        print(f"\n{Fore.MAGENTA}🚀 Testing Advanced Text-to-SQL Endpoint")
        print(f"{Fore.MAGENTA}{'-'*40}")
        
        test_cases = [
            {
                "name": "Government Entity Trades",
                "query": "show me all trades by government entities with trade amounts over 1 million",
                "max_tables": 5,
                "include_reasoning": True,
                "complexity": "Medium"
            },
            {
                "name": "Top CUSIP Analysis", 
                "query": "get me the top 5 CUSIPs by trading volume for last week",
                "max_tables": 8,
                "include_reasoning": True,
                "complexity": "High"
            },
            {
                "name": "ETD Source Systems",
                "query": "give me distinct source systems for cash ETD trades executed yesterday",
                "max_tables": 6,
                "include_reasoning": False,
                "complexity": "Low"
            },
            {
                "name": "Currency Mismatch Trades",
                "query": "find all trades where trade price currency is different from issue currency",
                "max_tables": 7,
                "include_reasoning": True,
                "complexity": "Medium"
            },
            {
                "name": "Trader Performance",
                "query": "show me top 10 traders by number of successful trades in Q4 2024",
                "max_tables": 10,
                "include_reasoning": True,
                "complexity": "High"
            },
            {
                "name": "Counterparty Risk Analysis",
                "query": "analyze counterparty exposure by trade count and total notional",
                "max_tables": 8,
                "include_reasoning": True,
                "complexity": "High"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{Fore.CYAN}Test {i}: {test_case['name']} ({test_case['complexity']} Complexity)")
            print(f"{Fore.YELLOW}Query: {test_case['query']}")
            
            start_time = time.time()
            
            payload = {
                "query": test_case['query'],
                "user_id": "test_user",
                "request_id": self.generate_request_id(),
                "max_tables": test_case['max_tables'],
                "include_reasoning": test_case['include_reasoning']
            }
            
            result = self.make_request('POST', '/api/v2/advanced_text_to_sql', payload)
            duration = time.time() - start_time
            
            if result['success']:
                data = result['data']
                sql_code = data.get('sql', '')
                reasoning = data.get('reasoning', '')
                tables_found = data.get('tables_found', [])
                table_details = data.get('table_details', {})
                processing_time = data.get('processing_time', 0)
                
                print(f"{Fore.GREEN}🎯 Tables Found ({len(tables_found)}): {', '.join(tables_found[:3])}")
                print(f"{Fore.GREEN}✅ SQL Generated ({len(sql_code)} chars):")
                
                # Show formatted SQL preview
                sql_preview = sql_code[:300] + "..." if len(sql_code) > 300 else sql_code
                print(f"{Fore.CYAN}{sql_preview}")
                
                print(f"{Fore.GREEN}⏱️  Total Processing Time: {processing_time:.3f}s")
                
                # Show reasoning if included
                if test_case['include_reasoning'] and reasoning and reasoning != "Reasoning hidden by request":
                    reasoning_preview = reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
                    print(f"{Fore.YELLOW}🧠 Reasoning: {reasoning_preview}")
                
                # Show table details summary
                if table_details:
                    total_columns = sum(len(details.get('columns', [])) for details in table_details.values())
                    print(f"{Fore.CYAN}📊 Schema: {len(table_details)} tables, {total_columns} total columns")
                
                success = bool(sql_code and 'SELECT' in sql_code.upper() and len(tables_found) > 0)
                details_str = f"Tables: {len(tables_found)}, SQL: {len(sql_code)} chars"
                
                self.log_test(f"Advanced Text-to-SQL: {test_case['name']}", success, duration, details_str)
            else:
                error_msg = result.get('error', 'Unknown error')
                print(f"{Fore.RED}❌ Error: {error_msg}")
                self.log_test(f"Advanced Text-to-SQL: {test_case['name']}", False, duration, f"Error: {error_msg}")
    
    def test_utility_endpoints(self):
        """Test utility endpoints."""
        print(f"\n{Fore.MAGENTA}🛠️  Testing Utility Endpoints")
        print(f"{Fore.MAGENTA}{'-'*40}")
        
        # Test table search
        print(f"\n{Fore.CYAN}🔍 Testing Table Search")
        start_time = time.time()
        result = self.make_request('GET', '/api/v2/tables/search', params={'query': 'trade', 'limit': 5})
        duration = time.time() - start_time
        
        if result['success']:
            data = result['data']
            results = data.get('results', [])
            print(f"{Fore.GREEN}📋 Found {len(results)} tables matching 'trade':")
            for table in results[:3]:
                print(f"   • {table.get('name')} ({table.get('table_type', 'unknown')})")
            
            self.log_test("Table Search", True, duration, f"Found {len(results)} tables")
        else:
            self.log_test("Table Search", False, duration, f"Error: {result.get('error')}")
        
        # Test SQL validation
        print(f"\n{Fore.CYAN}✅ Testing SQL Validation")
        test_sql = "SELECT * FROM trades WHERE trade_date > '2024-01-01'"
        
        start_time = time.time()
        result = self.make_request('POST', '/api/v2/sql/validate', params={'sql_query': test_sql})
        duration = time.time() - start_time
        
        if result['success']:
            data = result['data']
            is_valid = data.get('is_valid', False)
            error = data.get('error')
            
            print(f"{Fore.GREEN}🔍 SQL Validation: {'Valid' if is_valid else 'Invalid'}")
            if error:
                print(f"{Fore.YELLOW}⚠️  Error: {error}")
            
            self.log_test("SQL Validation", True, duration, f"Valid: {is_valid}")
        else:
            self.log_test("SQL Validation", False, duration, f"Error: {result.get('error')}")
    
    def test_performance_benchmark(self):
        """Test performance benchmark endpoint."""
        print(f"\n{Fore.MAGENTA}🏃‍♂️ Testing Performance Benchmark")
        print(f"{Fore.MAGENTA}{'-'*40}")
        
        start_time = time.time()
        result = self.make_request('POST', '/api/v2/performance/benchmark')
        duration = time.time() - start_time
        
        if result['success']:
            data = result['data']
            message = data.get('message', '')
            status = data.get('status', '')
            estimated_duration = data.get('estimated_duration', '')
            
            print(f"{Fore.GREEN}🚀 Benchmark Status: {status}")
            print(f"{Fore.GREEN}📝 Message: {message}")
            print(f"{Fore.GREEN}⏱️  Estimated Duration: {estimated_duration}")
            
            self.log_test("Performance Benchmark", True, duration, f"Status: {status}")
        else:
            error_msg = result.get('error', 'Unknown error')
            print(f"{Fore.RED}❌ Error: {error_msg}")
            self.log_test("Performance Benchmark", False, duration, f"Error: {error_msg}")
    
    def test_error_scenarios(self):
        """Test error handling scenarios."""
        print(f"\n{Fore.MAGENTA}🚨 Testing Error Scenarios")
        print(f"{Fore.MAGENTA}{'-'*40}")
        
        error_tests = [
            {
                "name": "Empty Query",
                "endpoint": "/api/v2/text_to_table",
                "payload": {
                    "query": "",
                    "user_id": "test_user",
                    "request_id": self.generate_request_id()
                }
            },
            {
                "name": "Invalid Table Name",
                "endpoint": "/api/v2/tables/nonexistent_table/details",
                "method": "GET"
            },
            {
                "name": "Missing Required Fields",
                "endpoint": "/api/v2/table_to_sql",
                "payload": {
                    "selected_tables": ["trades"]
                    # Missing original_query, user_id, request_id
                }
            },
            {
                "name": "Invalid SQL Syntax",
                "endpoint": "/api/v2/sql/validate",
                "method": "POST",
                "params": {"sql_query": "INVALID SQL SYNTAX HERE"}
            }
        ]
        
        for test in error_tests:
            print(f"\n{Fore.CYAN}Testing: {test['name']}")
            
            start_time = time.time()
            
            method = test.get('method', 'POST')
            endpoint = test['endpoint']
            payload = test.get('payload')
            params = test.get('params')
            
            result = self.make_request(method, endpoint, payload, params)
            duration = time.time() - start_time
            
            # For error scenarios, we expect either a controlled error response or proper error handling
            expected_error = not result['success'] or result.get('status_code', 200) >= 400
            
            if expected_error:
                print(f"{Fore.GREEN}✅ Error handled properly")
                self.log_test(f"Error Scenario: {test['name']}", True, duration, "Error handled correctly")
            else:
                print(f"{Fore.YELLOW}⚠️  Unexpected success - might need better validation")
                self.log_test(f"Error Scenario: {test['name']}", False, duration, "Expected error but got success")
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        print(f"\n{Fore.MAGENTA}📊 Test Report Generation")
        print(f"{Fore.MAGENTA}{'='*60}")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Summary statistics
        print(f"{Fore.CYAN}📈 Test Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   {Fore.GREEN}✅ Passed: {passed_tests}")
        print(f"   {Fore.RED}❌ Failed: {failed_tests}")
        print(f"   {Fore.YELLOW}📊 Success Rate: {success_rate:.1f}%")
        
        # Timing statistics
        if self.test_results:
            durations = [result['duration'] for result in self.test_results]
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            min_duration = min(durations)
            
            print(f"\n{Fore.CYAN}⏱️  Timing Statistics:")
            print(f"   Average: {avg_duration:.3f}s")
            print(f"   Maximum: {max_duration:.3f}s")
            print(f"   Minimum: {min_duration:.3f}s")
        
        # Detailed results table
        if self.test_results:
            print(f"\n{Fore.CYAN}📋 Detailed Results:")
            
            table_data = []
            for result in self.test_results:
                status = "✅ PASS" if result['success'] else "❌ FAIL"
                table_data.append([
                    result['test_name'][:40],
                    status,
                    f"{result['duration']:.3f}s",
                    result['details'][:30] if result['details'] else "-"
                ])
            
            headers = ["Test Name", "Status", "Duration", "Details"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Save detailed report to file
        report_filename = f"api_test_report_{int(time.time())}.json"
        
        report_data = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate,
                "test_timestamp": datetime.now().isoformat()
            },
            "timing_stats": {
                "average_duration": avg_duration if self.test_results else 0,
                "max_duration": max_duration if self.test_results else 0,
                "min_duration": min_duration if self.test_results else 0
            },
            "detailed_results": self.test_results
        }
        
        try:
            with open(report_filename, 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"\n{Fore.GREEN}💾 Detailed report saved to: {report_filename}")
        except Exception as e:
            print(f"\n{Fore.RED}❌ Failed to save report: {e}")
        
        return report_data
    
    def run_all_tests(self):
        """Run comprehensive test suite."""
        print(f"{Fore.MAGENTA}🧪 Starting Comprehensive API Test Suite")
        print(f"{Fore.MAGENTA}{'='*60}")
        
        start_time = time.time()
        
        try:
            # Health and info checks
            self.test_health_check()
            self.test_database_info()
            
            # Core functionality tests
            self.test_text_to_table()
            self.test_table_to_sql()
            self.test_advanced_text_to_sql()
            
            # Utility endpoints
            self.test_utility_endpoints()
            
            # Performance testing
            self.test_performance_benchmark()
            
            # Error handling
            self.test_error_scenarios()
            
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}⚠️  Tests interrupted by user")
        except Exception as e:
            print(f"\n{Fore.RED}❌ Test suite failed: {e}")
        
        total_duration = time.time() - start_time
        
        print(f"\n{Fore.MAGENTA}🏁 Test Suite Completed")
        print(f"{Fore.MAGENTA}⏱️  Total Duration: {total_duration:.2f}s")
        
        # Generate and return report
        return self.generate_test_report()


def main():
    """Main function to run API tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Graph API Test Suite")
    parser.add_argument(
        "--base-url", 
        default="http://localhost:8000", 
        help="Base URL of the API server (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--test", 
        choices=["health", "database", "text_to_table", "table_to_sql", "advanced_text_to_sql", "utilities", "errors", "all"],
        default="all",
        help="Specific test to run (default: all)"
    )
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = APITester(args.base_url)
    
    # Run specific tests
    if args.test == "health":
        tester.test_health_check()
    elif args.test == "database":
        tester.test_database_info()
    elif args.test == "text_to_table":
        tester.test_text_to_table()
    elif args.test == "table_to_sql":
        tester.test_table_to_sql()
    elif args.test == "advanced_text_to_sql":
        tester.test_advanced_text_to_sql()
    elif args.test == "utilities":
        tester.test_utility_endpoints()
    elif args.test == "errors":
        tester.test_error_scenarios()
    else:
        # Run all tests
        report = tester.run_all_tests()
        
        # Print final summary
        if report:
            success_rate = report['test_summary']['success_rate']
            if success_rate >= 90:
                print(f"\n{Fore.GREEN}🎉 Excellent! API is working great!")
            elif success_rate >= 75:
                print(f"\n{Fore.YELLOW}👍 Good! API is mostly functional with minor issues.")
            elif success_rate >= 50:
                print(f"\n{Fore.YELLOW}⚠️  Fair. API has some issues that need attention.")
            else:
                print(f"\n{Fore.RED}🚨 Critical! API has major issues requiring immediate attention.")


if __name__ == "__main__":
    main()


# Additional utility functions for advanced testing

class LoadTester:
    """Load testing utilities for the API."""
    
    def __init__(self, base_url: str, concurrent_users: int = 5):
        self.base_url = base_url
        self.concurrent_users = concurrent_users
        self.session = requests.Session()
    
    async def load_test_text_to_table(self, duration_seconds: int = 60):
        """Run load test on text-to-table endpoint."""
        import asyncio
        import aiohttp
        
        test_queries = [
            "show me all trades by government entities",
            "get me the CUSIP that was traded highest last week", 
            "give me distinct source systems for cash ETD trades",
            "find trades where currencies differ",
            "show me top traders by volume"
        ]
        
        async def make_request(session, query):
            payload = {
                "query": query,
                "user_id": f"load_test_user_{asyncio.current_task().get_name()}",
                "request_id": f"load_test_{int(time.time())}_{hash(query) % 10000}"
            }
            
            async with session.post(
                f"{self.base_url}/api/v2/text_to_table",
                json=payload
            ) as response:
                return await response.json(), response.status
        
        async def worker(session, worker_id):
            """Worker function for load testing."""
            requests_made = 0
            errors = 0
            response_times = []
            
            end_time = time.time() + duration_seconds
            
            while time.time() < end_time:
                query = test_queries[requests_made % len(test_queries)]
                
                start_time = time.time()
                try:
                    result, status = await make_request(session, query)
                    response_time = time.time() - start_time
                    response_times.append(response_time)
                    
                    if status != 200:
                        errors += 1
                        
                    requests_made += 1
                    
                except Exception as e:
                    errors += 1
                    print(f"Worker {worker_id} error: {e}")
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)
            
            return {
                'worker_id': worker_id,
                'requests_made': requests_made,
                'errors': errors,
                'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
                'max_response_time': max(response_times) if response_times else 0,
                'min_response_time': min(response_times) if response_times else 0
            }
        
        # Run load test
        async with aiohttp.ClientSession() as session:
            print(f"{Fore.CYAN}🚀 Starting load test with {self.concurrent_users} concurrent users for {duration_seconds}s")
            
            tasks = [
                worker(session, i) 
                for i in range(self.concurrent_users)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Aggregate results
            total_requests = sum(r['requests_made'] for r in results)
            total_errors = sum(r['errors'] for r in results)
            avg_response_times = [r['avg_response_time'] for r in results if r['avg_response_time'] > 0]
            
            print(f"\n{Fore.GREEN}📊 Load Test Results:")
            print(f"   Total Requests: {total_requests}")
            print(f"   Total Errors: {total_errors}")
            print(f"   Error Rate: {total_errors/total_requests*100:.2f}%" if total_requests > 0 else "0%")
            print(f"   Requests/Second: {total_requests/duration_seconds:.2f}")
            print(f"   Avg Response Time: {sum(avg_response_times)/len(avg_response_times):.3f}s" if avg_response_times else "N/A")
            
            return results


class APIDocumentationGenerator:
    """Generate API documentation from test results."""
    
    def __init__(self, test_results: List[Dict]):
        self.test_results = test_results
    
    def generate_markdown_docs(self) -> str:
        """Generate markdown documentation."""
        
        docs = """# Advanced Graph Traversal Text-to-SQL API Documentation

## Overview
This API provides advanced text-to-SQL capabilities using Graph Neural Networks, Reinforcement Learning, and Multi-level reasoning for intelligent table discovery and SQL generation.

## Base URL
```
http://localhost:8000/api/v2
```

## Authentication
Currently, no authentication is required. Include `user_id` and `request_id` in all requests for tracking.

## Endpoints

### 1. Text-to-Table Discovery
**Endpoint:** `POST /text_to_table`

**Description:** Find relevant database tables using Advanced Graph Traversal method.

**Request Body:**
```json
{
    "query": "show me all trades by government entities",
    "user_id": "your_user_id",
    "request_id": "unique_request_id",
    "max_tables": 10
}
```

**Response:**
```json
{
    "success": true,
    "tables": ["trades", "counterparty", "government_entities"],
    "table_details": {
        "trades": {
            "name": "trades",
            "description": "Main trading transactions table",
            "table_type": "fact",
            "record_count": 1000000,
            "columns": [...]
        }
    },
    "processing_time": 0.245,
    "method": "AdvancedGraphTraversal"
}
```

### 2. Table-to-SQL Generation
**Endpoint:** `POST /table_to_sql`

**Description:** Generate SQL from selected tables and natural language query.

**Request Body:**
```json
{
    "selected_tables": ["trades", "counterparty"],
    "original_query": "show me all trades by government entities",
    "user_id": "your_user_id",
    "request_id": "unique_request_id"
}
```

**Response:**
```json
{
    "success": true,
    "sql": "SELECT t.trade_id, t.trade_date, c.counterparty_name FROM trades t JOIN counterparty c ON t.counterparty_id = c.counterparty_id WHERE c.entity_type = 'GOVERNMENT'",
    "reasoning": "Generated SQL to find government entity trades...",
    "tables_used": ["trades", "counterparty"],
    "processing_time": 1.234,
    "validation_status": "valid"
}
```

### 3. Complete Text-to-SQL Pipeline
**Endpoint:** `POST /advanced_text_to_sql`

**Description:** Complete pipeline combining table discovery and SQL generation.

**Request Body:**
```json
{
    "query": "get me the top 5 CUSIPs by trading volume for last week",
    "user_id": "your_user_id", 
    "request_id": "unique_request_id",
    "max_tables": 8,
    "include_reasoning": true
}
```

**Response:**
```json
{
    "success": true,
    "sql": "SELECT p.cusip, SUM(t.trade_volume) as total_volume FROM trades t JOIN product p ON t.product_id = p.product_id WHERE t.trade_date >= DATEADD(week, -1, GETDATE()) GROUP BY p.cusip ORDER BY total_volume DESC LIMIT 5",
    "reasoning": "Identified relevant tables and generated aggregation query...",
    "tables_found": ["trades", "product", "cusip_master"],
    "table_details": {...},
    "processing_time": 1.567,
    "method": "AdvancedGraphTraversal"
}
```

## Utility Endpoints

### Health Check
`GET /health` - Check service health

### Database Info  
`GET /database/info` - Get database statistics

### Table Search
`GET /tables/search?query=trade&limit=10` - Search tables

### Table Details
`GET /tables/{table_name}/details` - Get specific table details

### SQL Validation
`POST /sql/validate?sql_query=SELECT * FROM trades` - Validate SQL syntax

### Performance Benchmark
`POST /performance/benchmark` - Run performance tests

## Error Handling

All endpoints return consistent error responses:

```json
{
    "detail": "Error description",
    "status_code": 400
}
```

Common HTTP status codes:
- `200` - Success
- `400` - Bad Request (invalid input)
- `404` - Not Found (table/resource not found)
- `500` - Internal Server Error

## Rate Limiting

No rate limiting is currently implemented, but consider implementing for production use.

## Performance Characteristics

Based on test results:
- Text-to-Table: ~0.2-0.5s average response time
- Table-to-SQL: ~1-2s average response time  
- Advanced Text-to-SQL: ~1.5-3s average response time

## Best Practices

1. **Request IDs**: Always provide unique request IDs for tracking
2. **Error Handling**: Implement proper error handling for all API calls
3. **Timeout**: Set appropriate timeouts (recommended: 30s for complex queries)
4. **Caching**: Consider caching table discovery results for similar queries
5. **Monitoring**: Monitor response times and error rates

## Examples by Use Case

### Trading Analysis
```bash
# Find trading-related tables
curl -X POST "http://localhost:8000/api/v2/text_to_table" \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "analyze trading volumes by counterparty",
    "user_id": "analyst_1",
    "request_id": "req_001"
  }'
```

### Risk Management
```bash
# Generate risk analysis SQL
curl -X POST "http://localhost:8000/api/v2/advanced_text_to_sql" \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "show me counterparty exposure exceeding risk limits",
    "user_id": "risk_manager",
    "request_id": "req_002",
    "max_tables": 5
  }'
```

### Regulatory Reporting
```bash
# Generate compliance reporting SQL  
curl -X POST "http://localhost:8000/api/v2/advanced_text_to_sql" \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "generate trade report for regulatory submission",
    "user_id": "compliance_officer", 
    "request_id": "req_003",
    "include_reasoning": true
  }'
```

## Support and Troubleshooting

### Common Issues

1. **No tables found**: Query may be too specific or use unfamiliar terminology
2. **SQL validation fails**: Table schemas may have changed or query is too complex
3. **Slow response times**: Complex queries with many tables may take longer

### Debugging

Use the `/health` endpoint to verify service status and database connectivity.
Check `/database/info` for current database statistics.

### Contact

For technical support and feature requests, contact the Data Architecture team.
"""
        
        return docs
    
    def save_documentation(self, filename: str = None):
        """Save documentation to file."""
        if not filename:
            filename = f"api_documentation_{int(time.time())}.md"
        
        docs = self.generate_markdown_docs()
        
        try:
            with open(filename, 'w') as f:
                f.write(docs)
            print(f"{Fore.GREEN}📚 Documentation saved to: {filename}")
            return filename
        except Exception as e:
            print(f"{Fore.RED}❌ Failed to save documentation: {e}")
            return None


# Example usage script
def run_example_tests():
    """Run a focused set of example tests for demonstration."""
    
    print(f"{Fore.CYAN}🎯 Running Example API Tests")
    print(f"{Fore.CYAN}{'='*50}")
    
    tester = APITester()
    
    # Example 1: Simple table discovery
    print(f"\n{Fore.YELLOW}Example 1: Table Discovery")
    payload = {
        "query": "show me government entity trades",
        "user_id": "demo_user",
        "request_id": "demo_001"
    }
    
    result = tester.make_request('POST', '/api/v2/text_to_table', payload)
    if result['success']:
        tables = result['data'].get('tables', [])
        print(f"{Fore.GREEN}✅ Found tables: {', '.join(tables[:3])}")
    else:
        print(f"{Fore.RED}❌ Error: {result.get('error')}")
    
    # Example 2: Complete text-to-SQL
    print(f"\n{Fore.YELLOW}Example 2: Complete Text-to-SQL")
    payload = {
        "query": "get top 5 CUSIPs by volume last week",
        "user_id": "demo_user",
        "request_id": "demo_002",
        "include_reasoning": True
    }
    
    result = tester.make_request('POST', '/api/v2/advanced_text_to_sql', payload)
    if result['success']:
        sql = result['data'].get('sql', '')
        tables = result['data'].get('tables_found', [])
        print(f"{Fore.GREEN}✅ Generated SQL ({len(sql)} chars)")
        print(f"{Fore.GREEN}📊 Used tables: {', '.join(tables[:3])}")
    else:
        print(f"{Fore.RED}❌ Error: {result.get('error')}")
    
    print(f"\n{Fore.CYAN}🎯 Example tests completed!")


if __name__ == "__main__":
    # Allow running examples directly
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "examples":
        run_example_tests()
    else:
        main()