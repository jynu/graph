# Updated sections for training_data_main_v3.py
# Add these imports at the top
import asyncio
from enhanced_query_log_analyzer import EnhancedQueryLogAnalyzer

class TrainingDataProcessor:
    """Main processor with enhanced query log analysis capabilities"""
    
    def __init__(self, config_files: List[str]):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        try:
            self.config_manager = ConfigManager(config_files)
            self.db_config = self.config_manager.get_database_config()
            self.system_config = self.config_manager.get_system_config()
            self.db_client = DatabaseClient(self.db_config, self.system_config)
            self.metadata_manager = MetadataManager(self.db_client, self.system_config)
            
            # UPDATED: Use enhanced query analyzer instead of basic one
            self.query_analyzer = EnhancedQueryLogAnalyzer(self.db_client)
            
            # Load rules engine
            rules_path = os.path.join(os.path.dirname(__file__), 'rules.json')
            self.rules_engine = RulesEngine(rules_path)
            self.output_generator = OutputGenerator(self.system_config)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize processor: {e}")
            raise
    
    async def _analyze_and_update_frequent_columns_enhanced(self, tables_list: List[str], config_data: dict):
        """Enhanced analysis method with intelligent SQL parsing and LLM integration"""
        
        self.logger.info("Starting enhanced query log analysis with intelligent SQL parsing...")
        
        months_back = int(config_data.get('LOG_ANALYSIS_MONTHS', '6'))
        min_usage = int(config_data.get('LOG_ANALYSIS_MIN_USAGE', '5'))
        
        # Create detailed output files
        output_dir = self.system_config.output_directory
        os.makedirs(output_dir, exist_ok=True)
        
        detailed_output_file = os.path.join(output_dir, "enhanced_column_discovery_analysis.json")
        comparison_output_file = os.path.join(output_dir, "enhanced_column_comparison_analysis.json")
        quality_metrics_file = os.path.join(output_dir, "query_quality_metrics.json")
        
        all_analysis_results = {}
        comparison_results = {}
        quality_metrics = {}
        
        for table_name in tables_list:
            try:
                self.logger.info(f"🔍 Enhanced analysis for table: {table_name}")
                
                # Use enhanced async analysis
                usage_data = await self.query_analyzer.analyze_table_usage_enhanced(
                    table_name, months_back=months_back
                )
                
                all_analysis_results[table_name] = usage_data
                
                # Extract quality metrics
                quality_metrics[table_name] = {
                    'total_raw_queries': usage_data['total_queries'],
                    'valid_sql_queries': usage_data['valid_sql_queries'],
                    'success_rate': usage_data['usage_statistics']['query_success_rate'],
                    'unique_columns_found': len(usage_data['column_usage_frequency']),
                    'analysis_method': usage_data['analysis_method']
                }
                
                if usage_data['total_queries'] > 0:
                    # Get discovered frequent columns
                    discovered_columns = [
                        col for col, count in usage_data['top_columns'] 
                        if count >= min_usage
                    ]
                    
                    self.logger.info(f"✅ Found {len(discovered_columns)} frequent columns from {usage_data['valid_sql_queries']} valid queries")
                    self.logger.info(f"   Query success rate: {usage_data['usage_statistics']['query_success_rate']:.1%}")
                    
                    # Handle different column modes
                    columns_key = f"{table_name}_Columns"
                    columns_value = config_data.get(columns_key, "")
                    
                    if columns_value == '[logs]':
                        # Replace with discovered columns
                        config_data[columns_key] = ",".join(discovered_columns[:20])
                        config_data[f"{table_name}_AnalysisMethod"] = "enhanced_logs"
                        self.logger.info(f"✅ Updated {table_name} with enhanced discovered columns")
                        
                    elif columns_value == '[generated]':
                        # Use heuristic method but also save discovery
                        try:
                            heuristic_columns = getFrequentColumn(
                                table_name, self.db_config.host, 
                                self.db_config.user_id, self.db_config.password
                            )
                            config_data[columns_key] = ",".join(heuristic_columns)
                            config_data[f"{table_name}_AnalysisMethod"] = "heuristic_with_enhanced_validation"
                        except Exception as e:
                            self.logger.error(f"Heuristic analysis failed, using enhanced discovered columns: {e}")
                            config_data[columns_key] = ",".join(discovered_columns[:20])
                            config_data[f"{table_name}_AnalysisMethod"] = "enhanced_logs_fallback"
                            
                    else:
                        # Explicit list: Compare and analyze
                        explicit_columns = [col.strip().lower() for col in columns_value.split(",")]
                        comparison_results[table_name] = self._compare_column_lists_enhanced(
                            explicit_columns, discovered_columns, usage_data
                        )
                        
                        config_data[f"{table_name}_AnalysisMethod"] = "explicit_with_enhanced_discovery"
                        
                        # Log enhanced comparison results
                        comp_data = comparison_results[table_name]
                        self.logger.info(f"📊 ENHANCED COMPARISON FOR {table_name}")
                        self.logger.info(f"   Your explicit list: {comp_data['explicit_count']} columns")
                        self.logger.info(f"   Enhanced discovered: {comp_data['discovered_count']} columns")
                        self.logger.info(f"   Quality score: {comp_data['discovery_quality_score']:.2f}")
                        self.logger.info(f"   Coverage: {comp_data['coverage_percentage']:.1f}%")
                        
                        # Log quality insights
                        if comp_data['high_confidence_missing']:
                            self.logger.info(f"   🎯 High-confidence missing columns:")
                            for col, usage, conf in comp_data['high_confidence_missing'][:3]:
                                self.logger.info(f"     - {col}: used {usage} times (confidence: {conf:.2f})")
                    
                    # Enhanced metadata with quality metrics
                    config_data[f"{table_name}_Enhanced_Analysis_Date"] = datetime.now().strftime("%Y%m%d_%H%M%S")
                    config_data[f"{table_name}_RawQueryCount"] = str(usage_data['total_queries'])
                    config_data[f"{table_name}_ValidQueryCount"] = str(usage_data['valid_sql_queries'])
                    config_data[f"{table_name}_QuerySuccessRate"] = f"{usage_data['usage_statistics']['query_success_rate']:.3f}"
                    config_data[f"{table_name}_DiscoveredColumns"] = ",".join(discovered_columns[:20])
                    config_data[f"{table_name}_UniqueColumnsFound"] = str(len(usage_data['column_usage_frequency']))
                    config_data[f"{table_name}_AnalysisMethod"] = usage_data['analysis_method']
                    
                    # Log top discovered columns with enhanced details
                    self.logger.info(f"🏆 Top 10 most frequent columns for {table_name}:")
                    for i, (col, count) in enumerate(usage_data['top_columns'][:10], 1):
                        percentage = (count / usage_data['valid_sql_queries']) * 100 if usage_data['valid_sql_queries'] > 0 else 0
                        self.logger.info(f"  {i:2d}. {col:30s} - used {count:4d} times ({percentage:5.1f}%)")
                    
                else:
                    self.logger.warning(f"❌ No valid query logs found for {table_name}")
                    config_data[f"{table_name}_AnalysisMethod"] = "no_valid_logs_found"
                    quality_metrics[table_name]['status'] = 'no_data'
                    
            except Exception as e:
                self.logger.error(f"❌ Enhanced analysis failed for {table_name}: {e}")
                config_data[f"{table_name}_AnalysisMethod"] = "enhanced_analysis_failed"
                quality_metrics[table_name] = {
                    'status': 'error',
                    'error_message': str(e)
                }
        
        # Save comprehensive analysis results
        try:
            # Detailed discovery analysis
            with open(detailed_output_file, 'w') as f:
                json.dump(all_analysis_results, f, indent=4, default=str)
            self.logger.info(f"📊 Enhanced discovery analysis saved: {detailed_output_file}")
            
            # Comparison analysis (if any)
            if comparison_results:
                with open(comparison_output_file, 'w') as f:
                    json.dump(comparison_results, f, indent=4, default=str)
                self.logger.info(f"📊 Enhanced comparison analysis saved: {comparison_output_file}")
            
            # Quality metrics
            with open(quality_metrics_file, 'w') as f:
                json.dump(quality_metrics, f, indent=4, default=str)
            self.logger.info(f"📊 Query quality metrics saved: {quality_metrics_file}")
            
            # Enhanced summary report
            summary_file = os.path.join(output_dir, "enhanced_analysis_summary.json")
            summary_data = self._create_enhanced_summary_report(
                all_analysis_results, comparison_results, quality_metrics, config_data
            )
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=4, default=str)
            self.logger.info(f"📊 Enhanced summary report saved: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save enhanced analysis results: {e}")
    
    def _compare_column_lists_enhanced(self, explicit_columns: List[str], discovered_columns: List[str], 
                                     usage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced comparison with quality metrics and confidence scores"""
        
        explicit_set = set(col.lower() for col in explicit_columns)
        discovered_set = set(col.lower() for col in discovered_columns)
        usage_frequency = usage_data['column_usage_frequency']
        
        # Basic comparison
        overlap = explicit_set.intersection(discovered_set)
        missing_from_explicit = discovered_set - explicit_set
        not_frequent = explicit_set - discovered_set
        
        # Enhanced analysis with confidence scoring
        high_confidence_missing = []
        medium_confidence_missing = []
        
        for col in missing_from_explicit:
            usage_count = usage_frequency.get(col, 0)
            total_queries = usage_data['valid_sql_queries']
            
            # Calculate confidence based on usage frequency and query success rate
            if total_queries > 0:
                usage_rate = usage_count / total_queries
                confidence = min(usage_rate * 2, 1.0)  # Scale confidence
                
                if confidence > 0.3:  # High confidence threshold
                    high_confidence_missing.append((col, usage_count, confidence))
                elif confidence > 0.1:  # Medium confidence threshold
                    medium_confidence_missing.append((col, usage_count, confidence))
        
        # Sort by confidence
        high_confidence_missing.sort(key=lambda x: x[2], reverse=True)
        medium_confidence_missing.sort(key=lambda x: x[2], reverse=True)
        
        # Calculate enhanced metrics
        discovery_quality_score = 0.0
        if discovered_columns:
            # Quality based on usage distribution and coverage
            total_discovered_usage = sum(usage_frequency.get(col, 0) for col in discovered_columns)
            coverage_score = len(overlap) / len(discovered_columns) if discovered_columns else 0
            usage_score = total_discovered_usage / (usage_data['valid_sql_queries'] * len(discovered_columns)) if usage_data['valid_sql_queries'] > 0 and discovered_columns else 0
            discovery_quality_score = (coverage_score * 0.6) + (usage_score * 0.4)
        
        # Analyze patterns in missing columns
        missing_patterns = self._analyze_column_patterns(list(missing_from_explicit))
        explicit_patterns = self._analyze_column_patterns(explicit_columns)
        
        return {
            'explicit_count': len(explicit_columns),
            'discovered_count': len(discovered_columns),
            'overlap': list(overlap),
            'overlap_count': len(overlap),
            'coverage_percentage': (len(overlap) / len(discovered_columns)) * 100 if discovered_columns else 0,
            'discovery_quality_score': discovery_quality_score,
            
            # Enhanced missing analysis
            'high_confidence_missing': high_confidence_missing,
            'medium_confidence_missing': medium_confidence_missing,
            'total_missing_count': len(missing_from_explicit),
            
            # Enhanced not-frequent analysis
            'not_frequent': [(col, usage_frequency.get(col, 0)) for col in not_frequent],
            'not_frequent_count': len(not_frequent),
            
            # Pattern analysis
            'missing_column_patterns': missing_patterns,
            'explicit_column_patterns': explicit_patterns,
            
            # Usage statistics
            'total_usage_in_discovered': sum(usage_frequency.get(col, 0) for col in discovered_columns),
            'total_usage_in_explicit': sum(usage_frequency.get(col, 0) for col in explicit_columns),
            'usage_efficiency_ratio': sum(usage_frequency.get(col, 0) for col in overlap) / sum(usage_frequency.get(col, 0) for col in explicit_columns) if sum(usage_frequency.get(col, 0) for col in explicit_columns) > 0 else 0
        }
    
    def _analyze_column_patterns(self, columns: List[str]) -> Dict[str, Any]:
        """Analyze patterns in column names to provide insights"""
        
        patterns = {
            'date_time_columns': [],
            'id_key_columns': [],
            'amount_price_columns': [],
            'status_flag_columns': [],
            'code_type_columns': []
        }
        
        for col in columns:
            col_lower = col.lower()
            
            if any(keyword in col_lower for keyword in ['date', 'time', 'timestamp']):
                patterns['date_time_columns'].append(col)
            elif any(keyword in col_lower for keyword in ['id', 'key', 'sk']):
                patterns['id_key_columns'].append(col)
            elif any(keyword in col_lower for keyword in ['amount', 'price', 'value', 'quantity']):
                patterns['amount_price_columns'].append(col)
            elif any(keyword in col_lower for keyword in ['status', 'flag', 'indicator']):
                patterns['status_flag_columns'].append(col)
            elif any(keyword in col_lower for keyword in ['code', 'type', 'category']):
                patterns['code_type_columns'].append(col)
        
        return {
            'patterns': patterns,
            'total_categorized': sum(len(pattern_list) for pattern_list in patterns.values()),
            'categorization_rate': sum(len(pattern_list) for pattern_list in patterns.values()) / len(columns) if columns else 0
        }
    
    def _create_enhanced_summary_report(self, all_analysis_results: Dict[str, Any], 
                                      comparison_results: Dict[str, Any], 
                                      quality_metrics: Dict[str, Any],
                                      config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create enhanced summary report with quality insights"""
        
        summary = {
            "analysis_metadata": {
                "analysis_date": datetime.now().isoformat(),
                "analysis_type": "enhanced_query_log_analysis_with_llm",
                "total_tables_analyzed": len(all_analysis_results),
                "analysis_version": "2.0"
            },
            "overall_quality_metrics": {},
            "tables": {}
        }
        
        # Calculate overall quality metrics
        total_raw_queries = sum(metrics.get('total_raw_queries', 0) for metrics in quality_metrics.values())
        total_valid_queries = sum(metrics.get('valid_sql_queries', 0) for metrics in quality_metrics.values())
        overall_success_rate = total_valid_queries / total_raw_queries if total_raw_queries > 0 else 0
        
        summary["overall_quality_metrics"] = {
            "total_raw_queries_processed": total_raw_queries,
            "total_valid_sql_queries": total_valid_queries,
            "overall_query_success_rate": overall_success_rate,
            "tables_with_data": len([m for m in quality_metrics.values() if m.get('total_raw_queries', 0) > 0]),
            "analysis_success_rate": len([m for m in quality_metrics.values() if m.get('status') != 'error']) / len(quality_metrics) if quality_metrics else 0
        }
        
        # Process each table
        for table_name, analysis_data in all_analysis_results.items():
            table_summary = {
                "data_quality": quality_metrics.get(table_name, {}),
                "column_discovery": {
                    "total_unique_columns_found": len(analysis_data.get('column_usage_frequency', {})),
                    "most_frequent_columns": analysis_data.get('top_columns', [])[:10],
                    "usage_distribution": analysis_data.get('usage_statistics', {}).get('usage_distribution', {}),
                },
                "analysis_method": analysis_data.get('analysis_method', 'unknown'),
                "final_column_selection": {
                    "method": config_data.get(f"{table_name}_AnalysisMethod", "unknown"),
                    "columns": config_data.get(f"{table_name}_Columns", "").split(",")[:10]
                }
            }
            
            # Add comparison insights if available
            if table_name in comparison_results:
                comp_data = comparison_results[table_name]
                table_summary["comparison_analysis"] = {
                    "quality_score": comp_data.get('discovery_quality_score', 0),
                    "coverage_percentage": comp_data.get('coverage_percentage', 0),
                    "high_confidence_recommendations": comp_data.get('high_confidence_missing', [])[:5],
                    "usage_efficiency": comp_data.get('usage_efficiency_ratio', 0),
                    "pattern_insights": comp_data.get('missing_column_patterns', {})
                }
            
            summary["tables"][table_name] = table_summary
        
        return summary
    
    def process(self, section: str = "transcation") -> bool:
        """Updated main processing workflow with enhanced analysis"""
        try:
            self.logger.info("Starting enhanced training data processing workflow")
            
            # Get configuration data
            config_data = self.config_manager.get_section_config(section)
            tables_list = self.config_manager.get_tables_list(section)
            
            # Test database connection
            self.logger.info("Testing database connection...")
            if hasattr(self.db_client, 'test_connection'):
                if not self.db_client.test_connection():
                    self.logger.error("Database connection test failed.")
                    return False
                else:
                    self.logger.info("✅ Database connection successful!")
            
            # Initialize master dictionary
            master_dict = {
                'APPLICATION_CONFIG': config_data,
                'META_DATA': {},
                'COLUMN_DISTINCT_VALUES': {},
                'DISTINCT_VALUE_DECISION': {},
                'DISTINCT_VALUES': {},
                'WHERE_CLAUSE': {}
            }
            
            # Step 1: Load metadata for all tables
            self.logger.info("Loading table metadata...")
            for table_name in tables_list:
                if not self.metadata_manager.load_table_metadata(table_name):
                    self.logger.warning(f"Failed to load metadata for table: {table_name}")
                    continue
            
            master_dict['META_DATA'] = self.metadata_manager.table_schemas
            
            # Step 2: Set processing day ranges
            self.logger.info("Setting processing day ranges...")
            for table_name in tables_list:
                config_data[f"{table_name}_NoOfDays"] = 90
            
            # Step 2.5: Enhanced query log analysis
            if self._should_analyze_query_logs(config_data):
                self.logger.info("🚀 Starting enhanced query log analysis...")
                
                # Test audit log access
                if hasattr(self.query_analyzer, 'test_audit_log_access'):
                    if not self.query_analyzer.test_audit_log_access():
                        self.logger.error("❌ Cannot access audit log table - skipping enhanced analysis")
                    else:
                        self.logger.info("✅ Audit log accessible! Running enhanced analysis...")
                        # Run enhanced async analysis
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(
                                self._analyze_and_update_frequent_columns_enhanced(tables_list, config_data)
                            )
                        finally:
                            loop.close()
                else:
                    self.logger.info("🔍 Running enhanced analysis without access test...")
                    # Run enhanced async analysis
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(
                            self._analyze_and_update_frequent_columns_enhanced(tables_list, config_data)
                        )
                    finally:
                        loop.close()
            
            # Continue with existing workflow...
            # Step 3: Process columns and get statistics
            self.logger.info("Processing column statistics...")
            for table_name in tables_list:
                columns_key = f"{table_name}_Columns"
                columns_value = config_data.get(columns_key, "")
                
                # Handle auto-generated columns
                if columns_value == "[generated]":
                    try:
                        column_list = getFrequentColumn(
                            table_name, 
                            self.db_config.host, 
                            self.db_config.user_id, 
                            self.db_config.password
                        )
                        config_data[columns_key] = ",".join(column_list)
                        self.logger.info(f"Generated columns for {table_name}: {column_list}")
                    except Exception as e:
                        self.logger.error(f"Failed to generate columns for {table_name}: {e}")
                        continue
                else:
                    column_list = [col.strip().lower() for col in columns_value.split(",")]
                
                # Get distinct counts for each column
                for column_name in column_list:
                    distinct_count = self.metadata_manager.get_column_distinct_count(
                        table_name, column_name, config_data
                    )
                    
                    if distinct_count is not None:
                        if table_name not in master_dict['COLUMN_DISTINCT_VALUES']:
                            master_dict['COLUMN_DISTINCT_VALUES'][table_name] = {}
                        master_dict['COLUMN_DISTINCT_VALUES'][table_name][column_name] = distinct_count
                        self.logger.info(f"Column {table_name}.{column_name}: {distinct_count} distinct values")
            
            # Continue with remaining steps (4-7) as in original implementation...
            # Step 4: Apply business rules
            self.logger.info("Applying business rules...")
            decisions = self.rules_engine.apply_rules(master_dict)
            master_dict['DISTINCT_VALUE_DECISION'] = decisions
            
            # Step 5: Get distinct values for approved columns
            self.logger.info("Fetching distinct values for approved columns...")
            for table_name, columns in decisions.items():
                for column_name, decision in columns.items():
                    if decision.enum in [Decision.COLUMNS_NEEDBA_EVAL, Decision.COLUMNS_NOTREJALLPHASE_NEEDBA_EVAL]:
                        distinct_values = self.metadata_manager.get_distinct_values(
                            table_name, column_name, config_data
                        )
                        
                        if table_name not in master_dict['DISTINCT_VALUES']:
                            master_dict['DISTINCT_VALUES'][table_name] = {}
                        master_dict['DISTINCT_VALUES'][table_name][column_name] = distinct_values
                        
                        self.logger.info(f"Retrieved {len(distinct_values)} distinct values for {table_name}.{column_name}")
            
            # Step 6: Generate final JSON output
            self.logger.info("Generating final JSON output...")
            final_data = self.output_generator.generate_final_json(master_dict, tables_list, config_data)
            
            # Step 7: Save outputs
            if self.output_generator.save_outputs(final_data):
                self.logger.info("🎉 Enhanced training data processing completed successfully!")
                return True
            else:
                self.logger.error("Failed to save outputs")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in enhanced processing workflow: {e}")
            return False