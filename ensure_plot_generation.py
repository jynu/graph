# AT THE END OF YOUR run_duckdb_benchmark() FUNCTION, MAKE SURE YOU HAVE:

def run_duckdb_benchmark():
    """Run comprehensive benchmark across all DuckDB retrieval methods."""
    
    # ... all your existing benchmark code ...
    
    # At the end of the function:
    
    # Performance Summary
    print(f"\n{'='*100}")
    print("📊 METHOD PERFORMANCE SUMMARY")
    print(f"{'='*100}")
    
    for method_name, perf in method_performance.items():
        if perf['success_count'] > 0:
            avg_time = perf['total_time'] / len(queries)
            avg_tables = perf['total_tables'] / perf['success_count']
            success_rate = (perf['success_count'] / len(queries)) * 100
            
            method_type = "🤖 LLM" if any(x in method_name for x in ['GPT', 'Gemini', 'Azure']) else "⚡ Local"
            print(f"{method_type} {method_name:25}: {success_rate:5.1f}% success | {avg_time:6.3f}s avg | {avg_tables:4.1f} avg tables")
    
    # *** CRITICAL: Export results and create visualizations ***
    print(f"\n📊 Exporting results and creating visualizations...")
    excel_filename = export_results(results)
    
    print(f"\n{'='*100}")
    print("✅ Comprehensive DuckDB Benchmark completed successfully!")
    print(f"📊 Processed {len(queries)} queries with {len(retrievers)} methods")
    print(f"📁 Results exported to {excel_filename}")
    print(f"📊 Visualizations saved as PNG files")
    print(f"🎯 Key Insights:")
    print(f"   • Check Recall@5 for graph traversal methods")
    print(f"   • Check Precision@1 for ranking methods") 
    print(f"   • Coverage shows comprehensive discovery")
    print(f"{'='*100}")
    
    return results

# ALSO MAKE SURE YOUR MAIN FUNCTION CALLS THE BENCHMARK:

def main():
    """Enhanced main function for DuckDB benchmark with comprehensive accuracy evaluation."""
    print("🚀 Enhanced DuckDB Table Retrieval Benchmark with Comprehensive Accuracy")
    print("=" * 80)
    
    # ... existing setup code ...
    
    try:
        # Run the enhanced benchmark
        results = run_duckdb_benchmark()  # This should create the plots
        
        if results:
            print("\n🎯 Benchmark Summary:")
            print(f"   📊 Processed {len(results)} queries")
            print(f"   📈 Comprehensive accuracy analysis completed")
            print(f"   📊 Visualizations generated")
            print(f"   🔍 Check PNG files for detailed plots")
            
    except Exception as e:
        print(f"❌ Enhanced benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()