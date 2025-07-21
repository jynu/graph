def export_results(results: List[Dict]):
    """Export enhanced benchmark results with accuracy analysis to Excel and create visualizations."""
    if not results:
        logger.warning("No results to export")
        return
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate accuracy metrics
    accuracy_metrics = calculate_accuracy_metrics(results)
    
    # Export to Excel with multiple sheets
    excel_filename = f"enhanced_duckdb_benchmark_with_accuracy_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # Main results
        df.to_excel(writer, sheet_name='Results', index=False)
        
        # Enhanced Summary statistics with accuracy
        summary_data = []
        method_columns = [col for col in df.columns if col.endswith('_Count')]
        
        for col in method_columns:
            method_name = col.replace('_Count', '')
            avg_count = df[col].mean()
            total_queries = len(df)
            successful_queries = sum(df[col] > 0)
            
            # Calculate average duration
            duration_col = f'{method_name}_Duration_sec'
            avg_duration = df[duration_col].mean() if duration_col in df.columns else 0
            max_duration = df[duration_col].max() if duration_col in df.columns else 0
            min_duration = df[duration_col].min() if duration_col in df.columns else 0
            
            # Calculate accuracy metrics
            accuracy_col = f'{method_name}_Accuracy'
            confidence_col = f'{method_name}_Confidence'
            relevant_col = f'{method_name}_Relevant_Tables'
            
            if accuracy_col in df.columns:
                accuracy_values = df[accuracy_col].dropna()
                avg_accuracy = (accuracy_values.sum() / len(accuracy_values) * 100) if len(accuracy_values) > 0 else 0
                
                confidence_values = df[confidence_col].dropna() if confidence_col in df.columns else []
                avg_confidence = confidence_values.mean() if len(confidence_values) > 0 else 0
                
                relevant_values = df[relevant_col].dropna() if relevant_col in df.columns else []
                avg_relevant = relevant_values.mean() if len(relevant_values) > 0 else 0
                
                # Calculate precision and recall
                total_predicted = df[col].sum()
                total_relevant_found = df[relevant_col].sum() if relevant_col in df.columns else 0
                
                precision = (total_relevant_found / total_predicted) if total_predicted > 0 else 0
                
                # Estimate recall (simplified)
                estimated_total_actual = 0
                for _, row in df.iterrows():
                    ground_truth = row.get('Ground_Truth_SQL', '')
                    if ground_truth:
                        estimated_total_actual += len(extract_tables_from_sql(ground_truth))
                
                recall = (total_relevant_found / estimated_total_actual) if estimated_total_actual > 0 else 0
                f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            else:
                avg_accuracy = avg_confidence = avg_relevant = precision = recall = f1_score = 0
            
            # Calculate total tables found
            total_tables = df[col].sum()
            
            # Determine method type
            method_type = "LLM" if any(x in method_name for x in ['GPT', 'Gemini', 'Azure']) else "Local"
            
            summary_data.append({
                'Method': method_name,
                'Type': method_type,
                'Success_Rate_%': round((successful_queries/total_queries)*100, 1),
                'Avg_Accuracy_%': round(avg_accuracy, 1),
                'Avg_Confidence': round(avg_confidence, 3),
                'Avg_Tables_per_Query': round(avg_count, 2),
                'Avg_Relevant_Tables': round(avg_relevant, 2),
                'Precision': round(precision, 3),
                'Recall': round(recall, 3),
                'F1_Score': round(f1_score, 3),
                'Total_Tables_Found': int(total_tables),
                'Avg_Duration_sec': round(avg_duration, 3),
                'Min_Duration_sec': round(min_duration, 3),
                'Max_Duration_sec': round(max_duration, 3),
                'Total_Queries': total_queries,
                'Successful_Queries': successful_queries
            })
        
        summary_df = pd.DataFrame(summary_data)
        # Sort by accuracy and then by method type
        summary_df = summary_df.sort_values(['Avg_Accuracy_%', 'Type'], ascending=[False, True])
        summary_df.to_excel(writer, sheet_name='Summary_with_Accuracy', index=False)
        
        # Accuracy-focused comparison sheet
        accuracy_comparison = summary_df[['Method', 'Type', 'Avg_Accuracy_%', 'Avg_Confidence', 
                                        'Precision', 'Recall', 'F1_Score', 'Avg_Duration_sec']].copy()
        accuracy_comparison = accuracy_comparison.sort_values('Avg_Accuracy_%', ascending=False)
        accuracy_comparison.to_excel(writer, sheet_name='Accuracy_Comparison', index=False)
        
        # LLM vs Local Methods Comparison
        llm_methods = summary_df[summary_df['Type'] == 'LLM'].copy()
        local_methods = summary_df[summary_df['Type'] == 'Local'].copy()
        
        if not llm_methods.empty:
            llm_methods.to_excel(writer, sheet_name='LLM_Methods', index=False)
        if not local_methods.empty:
            local_methods.to_excel(writer, sheet_name='Local_Methods', index=False)
        
        # Query-level accuracy analysis
        query_accuracy_data = []
        for _, row in df.iterrows():
            query_data = {
                'Query_ID': row['Query_ID'],
                'Question': row['Question'][:100] + '...' if len(row['Question']) > 100 else row['Question'],
                'Question_Length': row.get('Question_Length', 0),
                'Has_Ground_Truth': bool(row.get('Ground_Truth_SQL'))
            }
            
            # Add per-method accuracy
            method_accuracies = []
            for col in df.columns:
                if col.endswith('_Accuracy'):
                    method_name = col.replace('_Accuracy', '')
                    accuracy = row[col] if pd.notna(row[col]) else None
                    query_data[f'{method_name}_Accurate'] = accuracy
                    if accuracy is not None:
                        method_accuracies.append(1 if accuracy else 0)
            
            query_data['Avg_Accuracy_All_Methods'] = np.mean(method_accuracies) if method_accuracies else 0
            query_data['Methods_Correct_Count'] = sum(method_accuracies) if method_accuracies else 0
            query_data['Total_Methods_Tested'] = len(method_accuracies)
            
            query_accuracy_data.append(query_data)
        
        query_accuracy_df = pd.DataFrame(query_accuracy_data)
        query_accuracy_df.to_excel(writer, sheet_name='Query_Level_Accuracy', index=False)
        
        # Overall accuracy metrics summary
        metrics_summary = pd.DataFrame([accuracy_metrics])
        metrics_summary.to_excel(writer, sheet_name='Overall_Metrics', index=False)
        
        # Top-K accuracy analysis
        top_k_data = []
        for k in [1, 3, 5]:
            for method in summary_df['Method']:
                # Calculate top-k accuracy for each method
                correct_at_k = 0
                total_queries_with_method = 0
                
                for _, row in df.iterrows():
                    accuracy_col = f'{method}_Accuracy'
                    count_col = f'{method}_Count'
                    
                    if accuracy_col in row and pd.notna(row[accuracy_col]) and count_col in row:
                        total_queries_with_method += 1
                        tables_found = row[count_col] or 0
                        
                        # For top-k, we consider it correct if method found tables and was accurate
                        if tables_found >= k and row[accuracy_col]:
                            correct_at_k += 1
                
                top_k_accuracy = (correct_at_k / total_queries_with_method * 100) if total_queries_with_method > 0 else 0
                
                top_k_data.append({
                    'Method': method,
                    'K': k,
                    'Top_K_Accuracy_%': round(top_k_accuracy, 1),
                    'Correct_at_K': correct_at_k,
                    'Total_Queries': total_queries_with_method
                })
        
        top_k_df = pd.DataFrame(top_k_data)
        top_k_pivot = top_k_df.pivot(index='Method', columns='K', values='Top_K_Accuracy_%')
        top_k_pivot.to_excel(writer, sheet_name='Top_K_Accuracy')
    
    # Export to CSV
    csv_filename = f"enhanced_duckdb_benchmark_with_accuracy_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    
    # Create comprehensive visualizations
    print(f"\n📊 Creating accuracy visualizations...")
    try:
        create_accuracy_visualizations(results, timestamp)
        print(f"✅ Visualizations created successfully")
    except Exception as e:
        print(f"⚠️ Visualization creation failed: {e}")
    
    # Print summary statistics
    print(f"\n📈 ACCURACY ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    if accuracy_metrics:
        print(f"Overall Metrics:")
        for metric, value in accuracy_metrics.items():
            if isinstance(value, float):
                print(f"  {metric.replace('_', ' ').title()}: {value:.3f}")
            else:
                print(f"  {metric.replace('_', ' ').title()}: {value}")
    
    print(f"\nTop Performing Methods (by accuracy):")
    if summary_data:
        sorted_methods = sorted(summary_data, key=lambda x: x['Avg_Accuracy_%'], reverse=True)
        for i, method in enumerate(sorted_methods[:5], 1):
            print(f"  {i}. {method['Method']:20} - {method['Avg_Accuracy_%']:5.1f}% "
                  f"(Precision: {method['Precision']:.3f}, Recall: {method['Recall']:.3f})")
    
    logger.info(f"✅ Enhanced results with accuracy analysis exported to {excel_filename} and {csv_filename}")
    
    return excel_filename
                    '