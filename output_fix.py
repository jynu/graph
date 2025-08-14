📝 Functions to Update:
1. Fix the variable scope issue in generate_config_file method
Find this section in the ConfigGenerator class:
python# Enhanced metadata with LLM usage info
try:
    config.set(section_name, f"{table_name}_Enhanced_Analysis_Date", str(datetime.now().strftime("%Y%m%d_%H%M%S")))
    config.set(section_name, f"{table_name}_TotalQueries", str(analysis_result.get('total_queries', 0)))
    config.set(section_name, f"{table_name}_ValidQueries", str(analysis_result.get('valid_queries', 0)))
    config.set(section_name, f"{table_name}_ColumnsFound", str(len(analysis_result.get('column_usage_frequency', {}))))
    
    # Safely get analysis method
    analysis_method = analysis_result.get('usage_statistics', {}).get('analysis_method', 'rule_based')
    config.set(section_name, f"{table_name}_AnalysisMethod", str(analysis_method))
    
    # Safely get usage statistics and convert ALL to strings
    usage_stats = analysis_result.get('usage_statistics', {})
    
    # Convert specific numeric values to strings
    config.set(section_name, f"{table_name}_LLMCallsMade", str(usage_stats.get('llm_calls_made', 0)))
    config.set(section_name, f"{table_name}_TotalUniqueColumns", str(usage_stats.get('total_unique_columns', 0)))
    config.set(section_name, f"{table_name}_TotalColumnReferences", str(usage_stats.get('total_column_references', 0)))
    config.set(section_name, f"{table_name}_AverageUsagePerColumn", str(round(usage_stats.get('average_usage_per_column', 0.0), 2)))
    config.set(section_name, f"{table_name}_UniqueClients", str(usage_stats.get('unique_clients', 0)))
    
    # Handle most_used_column (could be tuple or None)
    most_used = usage_stats.get('most_used_column', None)
    if most_used and isinstance(most_used, (list, tuple)) and len(most_used) >= 2:
        config.set(section_name, f"{table_name}_MostUsedColumn", f"{most_used[0]}:{most_used[1]}")
    else:
        config.set(section_name, f"{table_name}_MostUsedColumn", "None")
    
    # LLM configuration info (ensure boolean values are converted to strings)
    config.set(section_name, f"{table_name}_LLMEnabled", str(bool(ENABLE_LLM_ANALYSIS and LLM_CLIENT_AVAILABLE)))
    config.set(section_name, f"{table_name}_HybridAnalysis", str(bool(USE_HYBRID_APPROACH)))
    
    self.logger.info("Successfully set all configuration metadata")
    
except Exception as metadata_error:
    self.logger.error(f"Error setting configuration metadata: {metadata_error}")
    self.logger.error(f"Usage statistics causing error: {usage_stats}")  # This line causes the error!
Replace with (fix variable scope):
python# Enhanced metadata with LLM usage info (move usage_stats outside try block)
usage_stats = analysis_result.get('usage_statistics', {})
self.logger.debug(f"Usage statistics: {usage_stats}")

try:
    config.set(section_name, f"{table_name}_Enhanced_Analysis_Date", str(datetime.now().strftime("%Y%m%d_%H%M%S")))
    config.set(section_name, f"{table_name}_TotalQueries", str(analysis_result.get('total_queries', 0)))
    config.set(section_name, f"{table_name}_ValidQueries", str(analysis_result.get('valid_queries', 0)))
    config.set(section_name, f"{table_name}_ColumnsFound", str(len(analysis_result.get('column_usage_frequency', {}))))
    
    # Safely get analysis method
    analysis_method = usage_stats.get('analysis_method', 'rule_based')
    config.set(section_name, f"{table_name}_AnalysisMethod", str(analysis_method))
    
    # Convert specific numeric values to strings with explicit type checking
    llm_calls = usage_stats.get('llm_calls_made', 0)
    config.set(section_name, f"{table_name}_LLMCallsMade", str(llm_calls))
    
    total_unique = usage_stats.get('total_unique_columns', 0)
    config.set(section_name, f"{table_name}_TotalUniqueColumns", str(total_unique))
    
    total_refs = usage_stats.get('total_column_references', 0)
    config.set(section_name, f"{table_name}_TotalColumnReferences", str(total_refs))
    
    # Handle average with safe conversion
    avg_usage = usage_stats.get('average_usage_per_column', 0.0)
    if isinstance(avg_usage, (int, float)):
        config.set(section_name, f"{table_name}_AverageUsagePerColumn", str(round(float(avg_usage), 2)))
    else:
        config.set(section_name, f"{table_name}_AverageUsagePerColumn", "0.0")
    
    unique_clients = usage_stats.get('unique_clients', 0)
    config.set(section_name, f"{table_name}_UniqueClients", str(unique_clients))
    
    # Handle most_used_column (could be tuple or None)
    most_used = usage_stats.get('most_used_column', None)
    if most_used and isinstance(most_used, (list, tuple)) and len(most_used) >= 2:
        config.set(section_name, f"{table_name}_MostUsedColumn", f"{str(most_used[0])}:{str(most_used[1])}")
    else:
        config.set(section_name, f"{table_name}_MostUsedColumn", "None")
    
    # LLM configuration info (ensure boolean values are converted to strings)
    config.set(section_name, f"{table_name}_LLMEnabled", str(bool(ENABLE_LLM_ANALYSIS and LLM_CLIENT_AVAILABLE)))
    config.set(section_name, f"{table_name}_HybridAnalysis", str(bool(USE_HYBRID_APPROACH)))
    
    self.logger.info("Successfully set all configuration metadata")
    
except Exception as metadata_error:
    self.logger.error(f"Error setting configuration metadata: {metadata_error}")
    self.logger.error(f"Usage statistics that caused error: {usage_stats}")
    # Continue execution even if metadata fails
2. Simplify the validation sections
Replace the two validation try-catch blocks with one simpler version:
python# Final validation: ensure ALL values are strings (simplified)
try:
    self.logger.info("Performing final validation of configuration values...")
    
    validation_errors = []
    for section_name_check in config.sections():
        for option_name in config.options(section_name_check):
            try:
                value = config.get(section_name_check, option_name)
                if not isinstance(value, str):
                    self.logger.warning(f"Converting non-string value: {option_name} = {value} ({type(value)})")
                    config.set(section_name_check, option_name, str(value))
            except Exception as val_error:
                validation_errors.append(f"{option_name}: {val_error}")
    
    if validation_errors:
        self.logger.warning(f"Validation issues found: {validation_errors}")
    else:
        self.logger.info("Configuration validation completed successfully")
            
except Exception as validation_error:
    self.logger.error(f"Configuration validation failed: {validation_error}")
    # Don't raise - try to continue with config file creation
3. Add a safety wrapper for the config.write operation
Replace the config file saving section:
python# Save configuration file
try:
    with open(config_path, 'w') as configfile:
        config.write(configfile)
    
    self.logger.info(f"Configuration file generated: {config_path}")
    return config_path
    
except Exception as e:
    self.logger.error(f"Error saving configuration file: {e}")
    raise
With this safer version:
python# Save configuration file with additional safety checks
try:
    # First, try to validate that all values can be written
    test_validation_passed = True
    for section_name_check in config.sections():
        for option_name in config.options(section_name_check):
            try:
                value = config.get(section_name_check, option_name)
                if not isinstance(value, str):
                    config.set(section_name_check, option_name, str(value))
            except Exception as test_error:
                self.logger.error(f"Cannot convert option {option_name}: {test_error}")
                test_validation_passed = False
    
    if not test_validation_passed:
        raise Exception("Configuration validation failed - cannot write file")
    
    # Now try to write the file
    with open(config_path, 'w', encoding='utf-8') as configfile:
        config.write(configfile)
    
    # Verify file was created and has content
    if os.path.exists(config_path) and os.path.getsize(config_path) > 0:
        self.logger.info(f"Configuration file generated successfully: {config_path}")
        return config_path
    else:
        raise Exception("Configuration file was not created properly")
    
except Exception as e:
    self.logger.error(f"Error saving configuration file: {e}")
    self.logger.error(f"Config sections: {config.sections()}")
    for section in config.sections():
        self.logger.error(f"Section {section} options: {config.options(section)}")
    raise
4. Fix the debug function scope issue
In the FrequentColumnProcessor class, fix the debug function:
python@staticmethod
def debug_analysis_result(analysis_result: Dict[str, Any], logger):
    """Debug function to inspect analysis result structure"""
    logger.info("=== ANALYSIS RESULT DEBUG ===")
    for key, value in analysis_result.items():
        logger.info(f"{key}: {type(value)} = {value}")
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                logger.info(f"  {sub_key}: {type(sub_value)} = {sub_value}")
    logger.info("=== END DEBUG ===")
And uncomment the call in process_table:
python# Uncomment this line for debugging
FrequentColumnProcessor.debug_analysis_result(analysis_result, self.logger)
📋 Summary of Changes:

Move usage_stats outside the try block to fix variable scope