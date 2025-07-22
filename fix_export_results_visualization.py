# UPDATE YOUR EXISTING export_results FUNCTION BY ADDING THESE LINES:

def export_results(results: List[Dict]):
    """Export comprehensive results with ALL accuracy types clearly labeled."""
    if not results:
        logger.warning("No results to export")
        return
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df = pd.DataFrame(results)
    
    # Calculate comprehensive metrics
    comprehensive_metrics = calculate_all_accuracy_metrics(results)
    
    excel_filename = f"comprehensive_accuracy_benchmark_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        
        # 1. Raw Results
        df.to_excel(writer, sheet_name='Raw_Results', index=False)
        
        # 2. COMPREHENSIVE SUMMARY with ALL accuracy types
        summary_data = []
        method_columns = [col for col in df.columns if col.endswith('_Count')]
        
        for col in method_columns:
            method_name = col.replace('_Count', '')
            
            # Get all metrics for this method
            method_data = comprehensive_metrics.get('method_metrics', {}).get(method_name, {})
            
            duration_col = f'{method_name}_Duration_sec'
            avg_duration = df[duration_col].mean() if duration_col in df.columns else 0
            
            count_col = f'{method_name}_Count'
            avg_tables = df[count_col].mean() if count_col in df.columns else 0
            success_rate = (df[count_col] > 0).mean() * 100 if count_col in df.columns else 0
            
            summary_data.append({
                'Method': method_name,
                
                # PRECISION@K (Ranking-based): Main table in top K positions
                'Precision@1_%': round(method_data.get('precision_at_1', 0) * 100, 1),
                'Precision@3_%': round(method_data.get('precision_at_3', 0) * 100, 1),
                'Precision@5_%': round(method_data.get('precision_at_5', 0) * 100, 1),
                
                # RECALL@K (Any match): ANY GT table in top K results
                'Recall@3_%': round(method_data.get('recall_at_3', 0) * 100, 1),
                'Recall@5_%': round(method_data.get('recall_at_5', 0) * 100, 1),
                'Recall@10_%': round(method_data.get('recall_at_10', 0) * 100, 1),
                
                # HIT RATE@K (same as Recall@K, different name)
                'Hit_Rate@5_%': round(method_data.get('hit_rate_at_5', 0) * 100, 1),
                'Hit_Rate@10_%': round(method_data.get('hit_rate_at_10', 0) * 100, 1),
                
                # COVERAGE: Percentage of all GT tables found
                'Avg_Coverage_%': round(method_data.get('avg_coverage', 0), 1),
                
                # RETRIEVAL SUCCESS: Any useful table found
                'Retrieval_Success_%': round(method_data.get('retrieval_success_rate', 0) * 100, 1),
                
                # Performance metrics
                'Avg_Tables_Found': round(avg_tables, 1),
                'Avg_Duration_sec': round(avg_duration, 3),
                'Query_Success_Rate_%': round(success_rate, 1),
                
                # Improvements
                'P@3_vs_P@1_Improvement_%': round(method_data.get('precision_improvement_3', 0), 1),
                'P@5_vs_P@1_Improvement_%': round(method_data.get('precision_improvement_5', 0), 1),
                'R@5_vs_R@3_Improvement_%': round(method_data.get('recall_improvement_5', 0), 1),
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Recall@5_%', ascending=False)  # Sort by most practical metric
        summary_df.to_excel(writer, sheet_name='COMPREHENSIVE_SUMMARY', index=False)
        
        # 3. ACCURACY TYPE COMPARISON
        accuracy_comparison = summary_df[[
            'Method', 'Precision@1_%', 'Precision@3_%', 'Precision@5_%',
            'Recall@3_%', 'Recall@5_%', 'Recall@10_%', 'Hit_Rate@5_%', 'Retrieval_Success_%'
        ]].copy()
        accuracy_comparison.to_excel(writer, sheet_name='ACCURACY_TYPES_COMPARISON', index=False)
        
        # 4. METHOD RANKINGS by different criteria
        rankings_data = []
        for _, row in summary_df.iterrows():
            rankings_data.append({
                'Method': row['Method'],
                'Best_For_Precision': f"P@1: {row['Precision@1_%']}%",
                'Best_For_Recall': f"R@5: {row['Recall@5_%']}%", 
                'Best_For_Coverage': f"Cov: {row['Avg_Coverage_%']}%",
                'Best_For_Success': f"Success: {row['Retrieval_Success_%']}%",
                'Best_For_Speed': f"Speed: {row['Avg_Duration_sec']}s",
                'Overall_Recommendation': determine_method_recommendation(row)
            })
        
        rankings_df = pd.DataFrame(rankings_data)
        rankings_df.to_excel(writer, sheet_name='METHOD_RECOMMENDATIONS', index=False)
        
        # 5. ACCURACY DEFINITIONS SHEET
        definitions_data = [
            {
                'Accuracy_Type': 'Precision@1',
                'Definition': 'Most important ground truth table appears in position 1',
                'Use_Case': 'When you need the EXACT best table first',
                'Strictness': 'Very Strict'
            },
            {
                'Accuracy_Type': 'Precision@3',
                'Definition': 'Most important ground truth table appears in top 3 positions',
                'Use_Case': 'When you can review top 3 results',
                'Strictness': 'Moderate'
            },
            {
                'Accuracy_Type': 'Precision@5',
                'Definition': 'Most important ground truth table appears in top 5 positions',
                'Use_Case': 'When you can review top 5 results',
                'Strictness': 'Relaxed'
            },
            {
                'Accuracy_Type': 'Recall@3',
                'Definition': 'ANY ground truth table appears in top 3 results',
                'Use_Case': 'When any relevant table is useful',
                'Strictness': 'Moderate'
            },
            {
                'Accuracy_Type': 'Recall@5',
                'Definition': 'ANY ground truth table appears in top 5 results',
                'Use_Case': 'Best for graph traversal methods',
                'Strictness': 'Relaxed'
            },
            {
                'Accuracy_Type': 'Recall@10',
                'Definition': 'ANY ground truth table appears in top 10 results',
                'Use_Case': 'Maximum coverage evaluation',
                'Strictness': 'Very Relaxed'
            },
            {
                'Accuracy_Type': 'Hit Rate@K',
                'Definition': 'Same as Recall@K (different terminology)',
                'Use_Case': 'Alternative name for Recall@K',
                'Strictness': 'Same as Recall'
            },
            {
                'Accuracy_Type': 'Coverage',
                'Definition': 'Percentage of ALL ground truth tables found (any position)',
                'Use_Case': 'Comprehensive table discovery',
                'Strictness': 'Position Independent'
            },
            {
                'Accuracy_Type': 'Retrieval Success',
                'Definition': 'Whether ANY useful table was found',
                'Use_Case': 'Binary success/failure evaluation',
                'Strictness': 'Minimum Viable'
            }
        ]
        
        definitions_df = pd.DataFrame(definitions_data)
        definitions_df.to_excel(writer, sheet_name='ACCURACY_DEFINITIONS', index=False)
    
    # Export to CSV
    csv_filename = f"comprehensive_accuracy_benchmark_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    
    # *** ADD THIS SECTION FOR VISUALIZATIONS ***
    print(f"\n📊 Creating comprehensive visualizations...")
    try:
        create_comprehensive_visualizations(results, summary_df, comprehensive_metrics, timestamp)
        print(f"✅ Visualizations created successfully")
    except Exception as e:
        print(f"⚠️ Visualization creation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Print comprehensive summary
    try:
        print_comprehensive_summary(comprehensive_metrics)
    except Exception as e:
        print(f"⚠️ Summary printing failed: {e}")
    
    # *** UPDATE RETURN STATEMENT ***
    logger.info(f"✅ Results exported to {excel_filename} and {csv_filename}")
    
    return excel_filename, summary_df, comprehensive_metrics  # Return tuple instead of just filename