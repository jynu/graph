# Enhanced JSON Export Methods for Training Data Preparation

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

class EnhancedConfigGenerator(ConfigGenerator):
    """Enhanced ConfigGenerator with comprehensive JSON export capabilities"""
    
    def __init__(self, system_config):
        super().__init__(system_config)
        self.logger = logging.getLogger(__name__)
    
    def export_training_data_json(self, table_name: str, analysis_result: Dict[str, Any], 
                                 schema_info: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Export comprehensive JSON files for training data preparation
        
        Returns:
            Dict with paths to generated JSON files
        """
        try:
            # Ensure output directory exists
            os.makedirs(self.system_config.output_dir, exist_ok=True)
            
            # Generate sanitized table name for filenames
            safe_table_name = table_name.replace('.', '_').replace(' ', '_')
            
            # Generate all JSON export files
            json_files = {}
            
            # 1. Frequent Columns Summary (for quick reference)
            json_files['summary'] = self._export_frequent_columns_summary(
                safe_table_name, table_name, analysis_result
            )
            
            # 2. Detailed Analysis Results (complete analysis data)
            json_files['detailed_analysis'] = self._export_detailed_analysis(
                safe_table_name, table_name, analysis_result
            )
            
            # 3. Training Features (ML-ready format)
            json_files['training_features'] = self._export_training_features(
                safe_table_name, table_name, analysis_result, schema_info
            )
            
            # 4. Column Usage Statistics (for analytics)
            json_files['usage_stats'] = self._export_usage_statistics(
                safe_table_name, table_name, analysis_result
            )
            
            # 5. Query Samples (for model training)
            json_files['query_samples'] = self._export_query_samples(
                safe_table_name, table_name, analysis_result
            )
            
            self.logger.info(f"✅ Generated {len(json_files)} JSON files for table: {table_name}")
            for file_type, file_path in json_files.items():
                self.logger.info(f"   - {file_type}: {file_path}")
            
            return json_files
            
        except Exception as e:
            self.logger.error(f"Error exporting JSON files for {table_name}: {e}")
            raise
    
    def _export_frequent_columns_summary(self, safe_table_name: str, table_name: str, 
                                       analysis_result: Dict[str, Any]) -> str:
        """Export 1: Frequent Columns Summary - Quick reference for most used columns"""
        
        output_path = os.path.join(self.system_config.output_dir, f"{safe_table_name}_frequent_columns.json")
        
        # Extract frequent columns based on minimum usage threshold
        top_columns = analysis_result.get('top_columns', [])
        frequent_columns = []
        column_details = []
        
        for item in top_columns:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                col_name, usage_count = item[0], item[1]
                if isinstance(usage_count, (int, float)) and usage_count >= LOG_ANALYSIS_MIN_USAGE:
                    frequent_columns.append(str(col_name))
                    column_details.append({
                        "column_name": str(col_name),
                        "usage_count": int(usage_count),
                        "usage_rank": len(column_details) + 1
                    })
        
        summary_data = {
            "metadata": {
                "table_name": table_name,
                "generated_at": datetime.now().isoformat(),
                "min_usage_threshold": LOG_ANALYSIS_MIN_USAGE,
                "analysis_method": analysis_result.get('usage_statistics', {}).get('analysis_method', 'unknown')
            },
            "summary": {
                "total_frequent_columns": len(frequent_columns),
                "frequent_columns": frequent_columns,
                "total_queries_analyzed": analysis_result.get('total_queries', 0),
                "valid_queries_found": analysis_result.get('valid_queries', 0)
            },
            "frequent_columns_details": column_details
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        return output_path
    
    def _export_detailed_analysis(self, safe_table_name: str, table_name: str, 
                                analysis_result: Dict[str, Any]) -> str:
        """Export 2: Detailed Analysis Results - Complete analysis data"""
        
        output_path = os.path.join(self.system_config.output_dir, f"{safe_table_name}_detailed_analysis.json")
        
        # Serialize datetime objects properly
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return str(obj)
        
        detailed_data = {
            "metadata": {
                "table_name": table_name,
                "exported_at": datetime.now().isoformat(),
                "export_version": "v1.0"
            },
            "analysis_results": analysis_result,
            "processing_summary": {
                "llm_calls_made": analysis_result.get('usage_statistics', {}).get('llm_calls_made', 0),
                "analysis_method": analysis_result.get('usage_statistics', {}).get('analysis_method', 'rule_based'),
                "unique_columns_found": len(analysis_result.get('column_usage_frequency', {})),
                "most_used_column": analysis_result.get('usage_statistics', {}).get('most_used_column'),
                "average_usage_per_column": analysis_result.get('usage_statistics', {}).get('average_usage_per_column', 0)
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, ensure_ascii=False, indent=2, default=json_serializer)
        
        return output_path
    
    def _export_training_features(self, safe_table_name: str, table_name: str, 
                                analysis_result: Dict[str, Any], 
                                schema_info: Optional[Dict[str, str]] = None) -> str:
        """Export 3: Training Features - ML-ready format for model training"""
        
        output_path = os.path.join(self.system_config.output_dir, f"{safe_table_name}_training_features.json")
        
        column_usage = analysis_result.get('column_usage_frequency', {})
        total_usage = sum(column_usage.values()) if column_usage else 1
        
        # Create feature vectors for each column
        column_features = []
        for col_name, usage_count in column_usage.items():
            feature_vector = {
                "column_name": col_name,
                "features": {
                    "usage_count": usage_count,
                    "usage_frequency": round(usage_count / total_usage, 4),
                    "usage_rank": sorted(column_usage.items(), key=lambda x: x[1], reverse=True).index((col_name, usage_count)) + 1,
                    "is_frequent": usage_count >= LOG_ANALYSIS_MIN_USAGE,
                    "column_length": len(col_name),
                    "has_underscore": "_" in col_name,
                    "has_date_keyword": any(keyword in col_name.lower() for keyword in ['date', 'time', 'timestamp']),
                    "has_id_keyword": col_name.lower().endswith('_id') or col_name.lower().endswith('id'),
                    "is_business_date": col_name.lower() == 'dwh_business_date'
                }
            }
            
            # Add schema information if available
            if schema_info and col_name in schema_info:
                schema_parts = schema_info[col_name].split('|')
                feature_vector["features"]["data_type"] = schema_parts[1] if len(schema_parts) > 1 else "unknown"
                feature_vector["features"]["raw_type"] = schema_parts[0] if len(schema_parts) > 0 else "unknown"
            
            column_features.append(feature_vector)
        
        # Sort by usage count for consistency
        column_features.sort(key=lambda x: x["features"]["usage_count"], reverse=True)
        
        training_data = {
            "metadata": {
                "table_name": table_name,
                "feature_version": "v1.0",
                "generated_at": datetime.now().isoformat(),
                "total_columns": len(column_features),
                "frequent_columns_count": sum(1 for cf in column_features if cf["features"]["is_frequent"])
            },
            "table_features": {
                "total_queries": analysis_result.get('total_queries', 0),
                "valid_queries": analysis_result.get('valid_queries', 0),
                "unique_clients": analysis_result.get('usage_statistics', {}).get('unique_clients', 0),
                "analysis_method": analysis_result.get('usage_statistics', {}).get('analysis_method', 'rule_based'),
                "table_type": self._classify_table_type(table_name)
            },
            "column_features": column_features
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        return output_path
    
    def _export_usage_statistics(self, safe_table_name: str, table_name: str, 
                               analysis_result: Dict[str, Any]) -> str:
        """Export 4: Usage Statistics - For analytics and reporting"""
        
        output_path = os.path.join(self.system_config.output_dir, f"{safe_table_name}_usage_stats.json")
        
        usage_stats = analysis_result.get('usage_statistics', {})
        column_usage = analysis_result.get('column_usage_frequency', {})
        
        # Calculate additional statistics
        usage_values = list(column_usage.values()) if column_usage else [0]
        
        stats_data = {
            "metadata": {
                "table_name": table_name,
                "generated_at": datetime.now().isoformat(),
                "analysis_period": "last_60_days"  # Based on your query
            },
            "query_statistics": {
                "total_queries_found": analysis_result.get('total_queries', 0),
                "valid_sql_queries": analysis_result.get('valid_queries', 0),
                "query_validation_rate": round(
                    analysis_result.get('valid_queries', 0) / max(analysis_result.get('total_queries', 1), 1), 4
                )
            },
            "column_statistics": {
                "total_unique_columns": usage_stats.get('total_unique_columns', 0),
                "frequent_columns": len([v for v in usage_values if v >= LOG_ANALYSIS_MIN_USAGE]),
                "total_column_references": usage_stats.get('total_column_references', 0),
                "average_usage_per_column": round(usage_stats.get('average_usage_per_column', 0), 2),
                "max_usage_count": max(usage_values) if usage_values else 0,
                "min_usage_count": min(usage_values) if usage_values else 0,
                "most_used_column": usage_stats.get('most_used_column')
            },
            "analysis_statistics": {
                "analysis_method": usage_stats.get('analysis_method', 'rule_based'),
                "llm_calls_made": usage_stats.get('llm_calls_made', 0),
                "unique_clients": usage_stats.get('unique_clients', 0)
            },
            "usage_distribution": self._calculate_usage_distribution(column_usage)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2)
        
        return output_path
    
    def _export_query_samples(self, safe_table_name: str, table_name: str, 
                            analysis_result: Dict[str, Any]) -> str:
        """Export 5: Query Samples - For model training and validation"""
        
        output_path = os.path.join(self.system_config.output_dir, f"{safe_table_name}_query_samples.json")
        
        # Note: In your current implementation, actual query texts might not be stored
        # This is a placeholder structure for when you add query sample collection
        
        query_samples_data = {
            "metadata": {
                "table_name": table_name,
                "generated_at": datetime.now().isoformat(),
                "note": "Query samples for training column extraction models"
            },
            "sample_queries": [
                # This would be populated with actual query samples
                # Format: {"query": "SELECT ...", "extracted_columns": ["col1", "col2"], "method": "llm|regex"}
            ],
            "extraction_patterns": {
                "common_select_patterns": [],
                "common_where_patterns": [],
                "common_join_patterns": []
            },
            "statistics": {
                "total_samples": 0,
                "llm_extracted": 0,
                "regex_extracted": 0,
                "complex_queries": 0,
                "simple_queries": 0
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(query_samples_data, f, ensure_ascii=False, indent=2)
        
        return output_path
    
    def _classify_table_type(self, table_name: str) -> str:
        """Classify table type based on name patterns"""
        table_lower = table_name.lower()
        
        if 'trade' in table_lower:
            return "trading"
        elif 'market' in table_lower or 'quote' in table_lower:
            return "market_data"
        elif 'position' in table_lower:
            return "position"
        elif 'audit' in table_lower or 'log' in table_lower:
            return "audit_log"
        elif 'reference' in table_lower or 'ref' in table_lower:
            return "reference"
        else:
            return "general"
    
    def _calculate_usage_distribution(self, column_usage: Dict[str, int]) -> Dict[str, Any]:
        """Calculate usage distribution statistics"""
        if not column_usage:
            return {"error": "No usage data available"}
        
        usage_values = list(column_usage.values())
        total_columns = len(usage_values)
        
        # Create usage buckets
        buckets = {
            "very_high_usage": len([v for v in usage_values if v >= 20]),
            "high_usage": len([v for v in usage_values if 10 <= v < 20]),
            "medium_usage": len([v for v in usage_values if 5 <= v < 10]),
            "low_usage": len([v for v in usage_values if 1 <= v < 5]),
            "minimal_usage": len([v for v in usage_values if v == 1])
        }
        
        return {
            "buckets": buckets,
            "percentages": {k: round(v / total_columns * 100, 2) for k, v in buckets.items()},
            "total_columns": total_columns
        }

# Update the main FrequentColumnProcessor class to use enhanced JSON export
class FrequentColumnProcessor:
    """Updated processor with enhanced JSON export"""
    
    def __init__(self, user_id: str, password: str, database_spec: str = "impala:prod"):
        # ... existing initialization code ...
        
        # Replace the config_generator with enhanced version
        minimal_system_config = type('SystemConfig', (), {
            'config_dir': CONFIG_OUTPUT_DIR,
            'output_dir': BASE_OUTPUT_DIR
        })()
        
        self.config_generator = EnhancedConfigGenerator(minimal_system_config)
        # ... rest of initialization ...
    
    def process_table(self, table_name: str) -> str:
        """Enhanced process_table with comprehensive JSON export"""
        try:
            self.logger.info(f"Processing table: {table_name}")
            
            # Test database connection
            self.logger.info("Testing database connection...")
            test_query = "SELECT 1 as test"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.test', delete=False) as temp_file:
                temp_output = temp_file.name
            
            try:
                if not self.db_client.execute_query(test_query, temp_output):
                    raise Exception("Database connection test failed")
                self.logger.info("✅ Database connection successful!")
            finally:
                if os.path.exists(temp_output):
                    os.unlink(temp_output)
            
            # Get table schema (for enhanced JSON export)
            self.logger.info(f"Loading schema for table: {table_name}")
            schema = self.db_client.get_table_schema(table_name)
            
            if not schema:
                self.logger.warning(f"Could not load schema for table: {table_name}")
            else:
                self.logger.info(f"Loaded schema with {len(schema)} columns")
            
            # Analyze query logs
            self.logger.info(f"Analyzing query logs for table: {table_name}")
            analysis_result = self.query_analyzer.analyze_table_usage(table_name)
            
            # Generate configuration file
            self.logger.info(f"Generating configuration file for table: {table_name}")
            try:
                config_path = self.config_generator.generate_config_file(
                    table_name, analysis_result, self.user_id, (self.secure_password or self.password or "")
                )
                self.logger.info(f"Configuration file generated successfully: {config_path}")
            except Exception as config_error:
                self.logger.error(f"Failed to generate configuration file: {config_error}")
                raise
            
            # Generate comprehensive JSON exports
            self.logger.info(f"Generating comprehensive JSON exports for table: {table_name}")
            try:
                json_files = self.config_generator.export_training_data_json(
                    table_name, analysis_result, schema
                )
                self.logger.info(f"JSON exports generated successfully:")
                for file_type, file_path in json_files.items():
                    self.logger.info(f"   - {file_type}: {file_path}")
            except Exception as json_error:
                self.logger.error(f"Failed to generate JSON exports: {json_error}")
                # Continue processing even if JSON export fails
            
            # Log results summary
            self.logger.info(f"✅ Processing completed for table: {table_name}")
            self.logger.info(f"   - Total queries analyzed: {analysis_result['total_queries']}")
            self.logger.info(f"   - Valid SQL queries: {analysis_result['valid_queries']}")
            self.logger.info(f"   - Unique columns found: {len(analysis_result['column_usage_frequency'])}")
            self.logger.info(f"   - Top frequent columns: {len(analysis_result['top_columns'])}")
            self.logger.info(f"   - Analysis method: {analysis_result['usage_statistics'].get('analysis_method', 'rule_based')}")
            
            if 'llm_calls_made' in analysis_result['usage_statistics']:
                self.logger.info(f"   - LLM calls made: {analysis_result['usage_statistics']['llm_calls_made']}")
            
            self.logger.info(f"   - Configuration file: {config_path}")
            
            return config_path
            
        except Exception as e:
            self.logger.error(f"Error processing table {table_name}: {e}")
            raise