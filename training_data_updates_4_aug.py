Issues Found in Your Code:
1. Missing _compare_column_lists Method
Your code calls this method but it doesn't exist. Add this to the TrainingDataProcessor class:
pythondef _compare_column_lists(self, explicit_columns: List[str], discovered_columns: List[str], 
                         usage_frequency: Dict[str, int]) -> Dict[str, Any]:
    """Compare explicit column list with discovered frequent columns"""
    
    explicit_set = set(col.lower() for col in explicit_columns)
    discovered_set = set(col.lower() for col in discovered_columns)
    
    # Find overlaps and differences
    overlap = explicit_set.intersection(discovered_set)
    missing_from_explicit = discovered_set - explicit_set
    not_frequent = explicit_set - discovered_set
    
    # Get usage stats for missing columns
    missing_with_usage = [
        (col, usage_frequency.get(col, 0)) 
        for col in missing_from_explicit
    ]
    missing_with_usage.sort(key=lambda x: x[1], reverse=True)
    
    # Get usage stats for non-frequent columns in explicit list
    not_frequent_with_usage = [
        (col, usage_frequency.get(col, 0)) 
        for col in not_frequent
    ]
    not_frequent_with_usage.sort(key=lambda x: x[1], reverse=True)
    
    return {
        'explicit_count': len(explicit_columns),
        'discovered_count': len(discovered_columns),
        'overlap': list(overlap),
        'overlap_count': len(overlap),
        'missing_from_explicit': missing_with_usage,
        'not_frequent': not_frequent_with_usage,
        'coverage_percentage': (len(overlap) / len(discovered_columns)) * 100 if discovered_columns else 0
    }
2. Replace Your Current _analyze_and_update_frequent_columns Method
Replace your existing method with this enhanced version that outputs JSON files:
pythondef _analyze_and_update_frequent_columns(self, tables_list: List[str], config_data: dict):
    """Analyze query logs and update frequent columns"""
    
    self.logger.info("Initializing query log analyzer...")
    query_analyzer = QueryLogAnalyzer(self.db_client)
    
    months_back = int(config_data.get('LOG_ANALYSIS_MONTHS', '6'))
    min_usage = int(config_data.get('LOG_ANALYSIS_MIN_USAGE', '5'))
    
    # CREATE DETAILED OUTPUT FILES
    output_dir = self.system_config.output_directory
    os.makedirs(output_dir, exist_ok=True)
    detailed_output_file = os.path.join(output_dir, "column_discovery_analysis.json")
    comparison_output_file = os.path.join(output_dir, "column_comparison_analysis.json")
    all_analysis_results = {}
    comparison_results = {}
    
    # ALWAYS RUN LOG ANALYSIS WHEN ENABLED (regardless of column setting)
    for table_name in tables_list:
        columns_key = f"{table_name}_Columns"
        columns_value = config_data.get(columns_key, "")
        
        try:
            self.logger.info(f"Analyzing query logs for table: {table_name}")
            print(f"DEBUG: Starting log analysis for {table_name}")
            print(f"DEBUG: Column setting: {columns_value[:100]}...")  # Show first 100 chars
            
            # Analyze query logs
            usage_data = query_analyzer.analyze_table_usage(table_name, months_back=months_back)
            all_analysis_results[table_name] = usage_data
            
            if usage_data['total_queries'] > 0:
                # Get discovered frequent columns
                discovered_columns = [
                    col for col, count in usage_data['top_columns'] 
                    if count >= min_usage
                ]
                
                self.logger.info(f"Found {len(discovered_columns)} frequent columns from {usage_data['total_queries']} queries")
                print(f"DEBUG: Discovered {len(discovered_columns)} frequent columns")
                
                # HANDLE DIFFERENT COLUMN MODES
                if columns_value == '[logs]':
                    # Replace with discovered columns
                    config_data[columns_key] = ",".join(discovered_columns[:20])
                    config_data[f"{table_name}_AnalysisMethod"] = "logs"
                    self.logger.info(f"Updated {table_name} with discovered columns from logs")
                    
                elif columns_value == '[generated]':
                    # Use heuristic method but also save discovery
                    try:
                        heuristic_columns = getFrequentColumn(
                            table_name, self.db_config.host, 
                            self.db_config.user_id, self.db_config.password
                        )
                        config_data[columns_key] = ",".join(heuristic_columns)
                        config_data[f"{table_name}_AnalysisMethod"] = "heuristic"
                    except Exception as e:
                        self.logger.error(f"Heuristic analysis failed, using discovered columns: {e}")
                        config_data[columns_key] = ",".join(discovered_columns[:20])
                        config_data[f"{table_name}_AnalysisMethod"] = "logs_fallback"
                        
                else:
                    # EXPLICIT LIST: Keep existing, but compare and analyze
                    explicit_columns = [col.strip().lower() for col in columns_value.split(",")]
                    comparison_results[table_name] = self._compare_column_lists(
                        explicit_columns, discovered_columns, usage_data['column_usage_frequency']
                    )
                    
                    config_data[f"{table_name}_AnalysisMethod"] = "explicit_with_discovery"
                    
                    self.logger.info(f"=== COLUMN COMPARISON FOR {table_name} ===")
                    self.logger.info(f"Your explicit list: {len(explicit_columns)} columns")
                    self.logger.info(f"Discovered frequent: {len(discovered_columns)} columns")
                    self.logger.info(f"Overlap: {len(comparison_results[table_name]['overlap'])} columns")
                    self.logger.info(f"Coverage: {comparison_results[table_name]['coverage_percentage']:.1f}%")
                    
                    # Log top missing frequent columns
                    missing = comparison_results[table_name]['missing_from_explicit'][:5]
                    if missing:
                        self.logger.info("Top 5 missing frequent columns:")
                        for col, usage in missing:
                            self.logger.info(f"  - {col}: used {usage} times")
                    
                    # Log low-usage columns from your list
                    not_frequent = comparison_results[table_name]['not_frequent'][:5]
                    if not_frequent:
                        self.logger.info("Top 5 low-usage columns in your list:")
                        for col, usage in not_frequent:
                            self.logger.info(f"  - {col}: used {usage} times")
                    
                    self.logger.info(f"=== END COMPARISON FOR {table_name} ===")
                
                # ALWAYS ADD DISCOVERY METADATA
                config_data[f"{table_name}_LogAnalysis_Date"] = datetime.now().strftime("%Y%m%d")
                config_data[f"{table_name}_QueryCount"] = str(usage_data['total_queries'])
                config_data[f"{table_name}_DiscoveredColumns"] = ",".join(discovered_columns[:20])
                config_data[f"{table_name}_UniqueColumns"] = str(len(usage_data['column_usage_frequency']))
                
                # LOG TOP DISCOVERED COLUMNS
                self.logger.info(f"Top 10 most frequent columns for {table_name}:")
                for i, (col, count) in enumerate(usage_data['top_columns'][:10], 1):
                    self.logger.info(f"  {i:2d}. {col:30s} - used {count:4d} times")
                
            else:
                self.logger.warning(f"No query logs found for {table_name}")
                config_data[f"{table_name}_AnalysisMethod"] = "no_logs_found"
                
        except Exception as e:
            self.logger.error(f"Failed to analyze logs for {table_name}: {e}")
            config_data[f"{table_name}_AnalysisMethod"] = "analysis_failed"
            print(f"DEBUG: Analysis failed for {table_name}: {e}")
    
    # SAVE COMPREHENSIVE ANALYSIS RESULTS TO JSON FILES
    try:
        with open(detailed_output_file, 'w') as f:
            json.dump(all_analysis_results, f, indent=4, default=str)
        self.logger.info(f"Detailed column discovery analysis saved to: {detailed_output_file}")
        print(f"📊 DISCOVERY ANALYSIS SAVED: {detailed_output_file}")
        
        if comparison_results:
            with open(comparison_output_file, 'w') as f:
                json.dump(comparison_results, f, indent=4, default=str)
            self.logger.info(f"Column comparison analysis saved to: {comparison_output_file}")
            print(f"📊 COMPARISON ANALYSIS SAVED: {comparison_output_file}")
        
        # ALSO SAVE A SUMMARY REPORT
        summary_file = os.path.join(output_dir, "frequent_columns_summary.json")
        summary_data = self._create_summary_report(all_analysis_results, comparison_results, config_data)
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=4, default=str)
        self.logger.info(f"Summary report saved to: {summary_file}")
        print(f"📊 SUMMARY REPORT SAVED: {summary_file}")
        
    except Exception as e:
        self.logger.error(f"Failed to save detailed analysis: {e}")
3. Add Summary Report Generation Method
Add this method to create a concise summary:
pythondef _create_summary_report(self, all_analysis_results: Dict[str, Any], 
                          comparison_results: Dict[str, Any], config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a summary report of the frequent column analysis"""
    
    summary = {
        "analysis_metadata": {
            "analysis_date": datetime.now().isoformat(),
            "analysis_type": "query_log_based_frequent_columns",
            "total_tables_analyzed": len(all_analysis_results)
        },
        "tables": {}
    }
    
    for table_name, analysis_data in all_analysis_results.items():
        table_summary = {
            "total_queries_analyzed": analysis_data['total_queries'],
            "analysis_period_months": analysis_data['analysis_period_months'],
            "total_unique_columns_found": len(analysis_data['column_usage_frequency']),
            "most_frequent_columns": analysis_data['top_columns'][:10],  # Top 10
            "analysis_method": config_data.get(f"{table_name}_AnalysisMethod", "unknown"),
            "columns_selected_for_processing": config_data.get(f"{table_name}_Columns", "").split(",")[:10]  # First 10
        }
        
        # Add comparison data if available
        if table_name in comparison_results:
            comp_data = comparison_results[table_name]
            table_summary["comparison_with_explicit_list"] = {
                "explicit_list_size": comp_data['explicit_count'],
                "discovered_frequent_size": comp_data['discovered_count'],
                "overlap_count": comp_data['overlap_count'],
                "coverage_percentage": comp_data['coverage_percentage'],
                "top_missing_frequent": comp_data['missing_from_explicit'][:5],
                "low_usage_in_explicit": comp_data['not_frequent'][:5]
            }
        
        summary["tables"][table_name] = table_summary
    
    return summary
4. Fix Import Issue in query_log_analyzer.py
Your query_log_analyzer.py looks good, but ensure it can handle the tempfile cleanup properly. Add this fix to the _parse_log_file method:
pythondef _parse_log_file(self, log_file: str) -> List[Dict[str, Any]]:
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
        
        # Clean up the temporary file
        try:
            if os.path.exists(log_file):
                os.unlink(log_file)
        except Exception as cleanup_error:
            self.logger.warning(f"Failed to cleanup temp file {log_file}: {cleanup_error}")
        
        return log_data
        
    except Exception as e:
        self.logger.error(f"Error parsing log file: {e}")
        return []
JSON Output Files You'll Get: