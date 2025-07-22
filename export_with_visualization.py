# UPDATE YOUR EXPORT_RESULTS FUNCTION TO INCLUDE VISUALIZATION CREATION:

def export_results(results: List[Dict]):
    """Export enhanced benchmark results with comprehensive accuracy analysis and create visualizations."""
    if not results:
        logger.warning("No results to export")
        return
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate comprehensive accuracy metrics
    comprehensive_metrics = calculate_all_accuracy_metrics(results)
    
    # Export to Excel with multiple sheets
    excel_filename = f"comprehensive_accuracy_benchmark_{timestamp}.xlsx"
    
    # Create comprehensive summary for visualizations
    summary_data = []
    method_columns = [col for col in df.columns if col.endswith('_Count')]
    
    for col in method_columns:
        method_name = col.replace('_Count', '')
        
        # Get method metrics
        method_data = comprehensive_metrics.get('method_metrics', {}).get(method_name, {})
        
        duration_col = f'{method_name}_Duration_sec'
        avg_duration = df[duration_col].mean() if duration_col in df.columns else 0
        
        count_col = f'{method_name}_Count'
        avg_tables = df[count_col].mean() if count_col in df.columns else 0
        success_rate = (df[count_col] > 0).mean() * 100 if count_col in df.columns else 0
        
        summary_data.append({
            'Method': method_name,
            'Precision@1_%': round(method_data.get('precision_at_1', 0) * 100, 1),
            'Precision@3_%': round(method_data.get('precision_at_3', 0) * 100, 1),
            'Precision@5_%': round(method_data.get('precision_at_5', 0) * 100, 1),
            'Recall@3_%': round(method_data.get('recall_at_3', 0) * 100, 1),
            'Recall@5_%': round(method_data.get('recall_at_5', 0) * 100, 1),
            'Recall@10_%': round(method_data.get('recall_at_10', 0) * 100, 1),
            'Avg_Coverage_%': round(method_data.get('avg_coverage', 0), 1),
            'Retrieval_Success_%': round(method_data.get('retrieval_success_rate', 0) * 100, 1),
            'Avg_Tables_Found': round(avg_tables, 1),
            'Avg_Duration_sec': round(avg_duration, 3),
            'Query_Success_Rate_%': round(success_rate, 1),
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Recall@5_%', ascending=False)  # Sort by key metric for graph methods
    
    # Export to Excel
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Raw_Results', index=False)
        summary_df.to_excel(writer, sheet_name='Comprehensive_Summary', index=False)
        
        # Add accuracy comparison sheet
        accuracy_comparison = summary_df[['Method', 'Precision@1_%', 'Precision@5_%', 'Recall@5_%', 'Recall@10_%', 'Avg_Coverage_%']].copy()
        accuracy_comparison.to_excel(writer, sheet_name='Accuracy_Comparison', index=False)
    
    # Export to CSV
    csv_filename = f"comprehensive_accuracy_benchmark_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    
    # CREATE COMPREHENSIVE VISUALIZATIONS
    print(f"\n📊 Creating comprehensive visualizations...")
    try:
        create_comprehensive_visualizations(results, summary_df, comprehensive_metrics, timestamp)
        print(f"✅ Visualizations created successfully")
    except Exception as e:
        print(f"⚠️ Visualization creation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Print comprehensive summary
    print_comprehensive_summary(comprehensive_metrics)
    
    logger.info(f"✅ Results exported to {excel_filename} and {csv_filename}")
    
    return excel_filename