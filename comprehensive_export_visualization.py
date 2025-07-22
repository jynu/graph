def calculate_all_accuracy_metrics(results: List[Dict]) -> Dict[str, Any]:
    """Calculate ALL accuracy metrics: Precision@K, Recall@K, Hit Rate@K, Coverage."""
    
    if not results:
        return {}
    
    # Get all methods
    methods = set()
    for result in results:
        for key in result.keys():
            if key.endswith('_Precision_At_1'):
                methods.add(key.replace('_Precision_At_1', ''))
    
    method_metrics = {}
    
    for method in methods:
        # Collect all evaluations for this method
        precision_1 = []
        precision_3 = []
        precision_5 = []
        recall_3 = []
        recall_5 = []
        recall_10 = []
        hit_rate_3 = []
        hit_rate_5 = []
        hit_rate_10 = []
        coverage_scores = []
        retrieval_success = []
        
        for result in results:
            # Precision@K metrics
            p1_key = f'{method}_Precision_At_1'
            p3_key = f'{method}_Precision_At_3'
            p5_key = f'{method}_Precision_At_5'
            
            if p1_key in result and result[p1_key] is not None:
                precision_1.append(1 if result[p1_key] else 0)
                precision_3.append(1 if result.get(p3_key, False) else 0)
                precision_5.append(1 if result.get(p5_key, False) else 0)
            
            # Recall@K metrics
            r3_key = f'{method}_Recall_At_3'
            r5_key = f'{method}_Recall_At_5'
            r10_key = f'{method}_Recall_At_10'
            
            if r3_key in result and result[r3_key] is not None:
                recall_3.append(1 if result[r3_key] else 0)
                recall_5.append(1 if result.get(r5_key, False) else 0)
                recall_10.append(1 if result.get(r10_key, False) else 0)
            
            # Hit Rate@K metrics (same as Recall@K)
            hr3_key = f'{method}_Hit_Rate_At_3'
            hr5_key = f'{method}_Hit_Rate_At_5'
            hr10_key = f'{method}_Hit_Rate_At_10'
            
            if hr3_key in result and result[hr3_key] is not None:
                hit_rate_3.append(1 if result[hr3_key] else 0)
                hit_rate_5.append(1 if result.get(hr5_key, False) else 0)
                hit_rate_10.append(1 if result.get(hr10_key, False) else 0)
            
            # Coverage and Success
            cov_key = f'{method}_Coverage_%'
            success_key = f'{method}_Retrieval_Success'
            
            if cov_key in result and result[cov_key] is not None:
                coverage_scores.append(result[cov_key])
            
            if success_key in result and result[success_key] is not None:
                retrieval_success.append(1 if result[success_key] else 0)
        
        if precision_1:  # Only calculate if we have data
            method_metrics[method] = {
                # Precision@K (Ranking-based)
                'precision_at_1': sum(precision_1) / len(precision_1),
                'precision_at_3': sum(precision_3) / len(precision_3),
                'precision_at_5': sum(precision_5) / len(precision_5),
                
                # Recall@K (Any match)
                'recall_at_3': sum(recall_3) / len(recall_3) if recall_3 else 0,
                'recall_at_5': sum(recall_5) / len(recall_5) if recall_5 else 0,
                'recall_at_10': sum(recall_10) / len(recall_10) if recall_10 else 0,
                
                # Hit Rate@K (same as Recall@K)
                'hit_rate_at_3': sum(hit_rate_3) / len(hit_rate_3) if hit_rate_3 else 0,
                'hit_rate_at_5': sum(hit_rate_5) / len(hit_rate_5) if hit_rate_5 else 0,
                'hit_rate_at_10': sum(hit_rate_10) / len(hit_rate_10) if hit_rate_10 else 0,
                
                # Coverage and Success
                'avg_coverage': sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0,
                'retrieval_success_rate': sum(retrieval_success) / len(retrieval_success) if retrieval_success else 0,
                
                # Meta info
                'total_evaluations': len(precision_1),
            }
            
            # Calculate improvements
            base_precision = method_metrics[method]['precision_at_1']
            if base_precision > 0:
                method_metrics[method]['precision_improvement_3'] = (
                    method_metrics[method]['precision_at_3'] / base_precision - 1
                ) * 100
                method_metrics[method]['precision_improvement_5'] = (
                    method_metrics[method]['precision_at_5'] / base_precision - 1
                ) * 100
            else:
                method_metrics[method]['precision_improvement_3'] = float('inf') if method_metrics[method]['precision_at_3'] > 0 else 0
                method_metrics[method]['precision_improvement_5'] = float('inf') if method_metrics[method]['precision_at_5'] > 0 else 0
            
            # Calculate Recall improvements
            base_recall = method_metrics[method]['recall_at_3']
            if base_recall > 0:
                method_metrics[method]['recall_improvement_5'] = (
                    method_metrics[method]['recall_at_5'] / base_recall - 1
                ) * 100
                method_metrics[method]['recall_improvement_10'] = (
                    method_metrics[method]['recall_at_10'] / base_recall - 1
                ) * 100
            else:
                method_metrics[method]['recall_improvement_5'] = float('inf') if method_metrics[method]['recall_at_5'] > 0 else 0
                method_metrics[method]['recall_improvement_10'] = float('inf') if method_metrics[method]['recall_at_10'] > 0 else 0
    
    # Overall statistics
    if method_metrics:
        overall_metrics = {
            'total_methods': len(method_metrics),
            'total_queries': len(results),
            
            # Average Precision@K
            'avg_precision_at_1': sum(m['precision_at_1'] for m in method_metrics.values()) / len(method_metrics),
            'avg_precision_at_3': sum(m['precision_at_3'] for m in method_metrics.values()) / len(method_metrics),
            'avg_precision_at_5': sum(m['precision_at_5'] for m in method_metrics.values()) / len(method_metrics),
            
            # Average Recall@K
            'avg_recall_at_3': sum(m['recall_at_3'] for m in method_metrics.values()) / len(method_metrics),
            'avg_recall_at_5': sum(m['recall_at_5'] for m in method_metrics.values()) / len(method_metrics),
            'avg_recall_at_10': sum(m['recall_at_10'] for m in method_metrics.values()) / len(method_metrics),
            
            # Average Coverage and Success
            'avg_coverage': sum(m['avg_coverage'] for m in method_metrics.values()) / len(method_metrics),
            'avg_retrieval_success': sum(m['retrieval_success_rate'] for m in method_metrics.values()) / len(method_metrics),
            
            # Best performing methods
            'best_precision_1': max(method_metrics.items(), key=lambda x: x[1]['precision_at_1'])[0],
            'best_recall_5': max(method_metrics.items(), key=lambda x: x[1]['recall_at_5'])[0],
            'best_coverage': max(method_metrics.items(), key=lambda x: x[1]['avg_coverage'])[0],
            'best_retrieval_success': max(method_metrics.items(), key=lambda x: x[1]['retrieval_success_rate'])[0],
        }
        
        # Add method-specific metrics
        overall_metrics['method_metrics'] = method_metrics
        
        return overall_metrics
    
    return {'error': 'No valid accuracy data found'}

def export_comprehensive_results(results: List[Dict]):
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
    
    return excel_filename, summary_df, comprehensive_metrics

def determine_method_recommendation(row: pd.Series) -> str:
    """Determine the best use case recommendation for each method."""
    
    precision_1 = row['Precision@1_%']
    recall_5 = row['Recall@5_%']
    coverage = row['Avg_Coverage_%']
    success = row['Retrieval_Success_%']
    duration = row['Avg_Duration_sec']
    
    # Determine primary strength
    if precision_1 >= 70:
        return "Excellent for precise ranking (use when you need exact best table)"
    elif recall_5 >= 80:
        return "Excellent for comprehensive retrieval (use for graph traversal)"
    elif coverage >= 75:
        return "Excellent for coverage (finds most relevant tables)"
    elif success >= 90 and duration < 1.0:
        return "Excellent for fast reliable retrieval"
    elif duration < 0.1:
        return "Best for real-time applications (very fast)"
    elif success >= 85:
        return "Good general-purpose method"
    else:
        return "Specialized use cases only"

def create_comprehensive_visualizations(results: List[Dict], summary_df: pd.DataFrame, 
                                      comprehensive_metrics: Dict, timestamp: str):
    """Create comprehensive visualizations showing ALL accuracy types."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create main comprehensive figure
    fig = plt.figure(figsize=(24, 18))
    
    # 1. Precision@K vs Recall@K Comparison (MAIN PLOT)
    ax1 = plt.subplot(3, 4, (1, 4))
    plot_precision_vs_recall_comparison(summary_df, ax1)
    
    # 2. Method Performance Radar Chart
    ax2 = plt.subplot(3, 4, 5)
    plot_comprehensive_radar_chart(summary_df, ax2)
    
    # 3. Accuracy Type Heatmap
    ax3 = plt.subplot(3, 4, 6)
    plot_accuracy_type_heatmap(summary_df, ax3)
    
    # 4. Recall@K Progression
    ax4 = plt.subplot(3, 4, 7)
    plot_recall_progression(summary_df, ax4)
    
    # 5. Success Rate Analysis
    ax5 = plt.subplot(3, 4, 8)
    plot_success_rate_analysis(summary_df, ax5)
    
    # 6. Coverage vs Speed Trade-off
    ax6 = plt.subplot(3, 4, 9)
    plot_coverage_speed_tradeoff(summary_df, ax6)
    
    # 7. Method Recommendation Matrix
    ax7 = plt.subplot(3, 4, 10)
    plot_method_recommendation_matrix(summary_df, ax7)
    
    # 8. Accuracy Improvement Analysis
    ax8 = plt.subplot(3, 4, 11)
    plot_accuracy_improvements(summary_df, ax8)
    
    # 9. Overall Performance Score
    ax9 = plt.subplot(3, 4, 12)
    plot_overall_performance_score(summary_df, ax9)
    
    plt.suptitle('COMPREHENSIVE TABLE RETRIEVAL ACCURACY ANALYSIS\n' +
                 'Precision@K (Ranking) | Recall@K (Any Match) | Coverage | Success Rate', 
                 fontsize=18, fontweight='bold')
    plt.tight_layout(pad=3.0)
    
    # Save the comprehensive plot
    filename = f"comprehensive_accuracy_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Comprehensive accuracy analysis saved to {filename}")
    
    plt.show()

def plot_precision_vs_recall_comparison(summary_df: pd.DataFrame, ax):
    """Plot comprehensive Precision@K vs Recall@K comparison."""
    
    methods = summary_df['Method'].tolist()
    
    # Precision@K data
    precision_1 = summary_df['Precision@1_%'].tolist()
    precision_3 = summary_df['Precision@3_%'].tolist()
    precision_5 = summary_df['Precision@5_%'].tolist()
    
    # Recall@K data
    recall_3 = summary_df['Recall@3_%'].tolist()
    recall_5 = summary_df['Recall@5_%'].tolist()
    recall_10 = summary_df['Recall@10_%'].tolist()
    
    x = np.arange(len(methods))
    width = 0.13
    
    # Plot Precision@K
    ax.bar(x - 2.5*width, precision_1, width, label='Precision@1 (Strict)', color='#FF6B6B', alpha=0.8)
    ax.bar(x - 1.5*width, precision_3, width, label='Precision@3', color='#FF8E8E', alpha=0.8)
    ax.bar(x - 0.5*width, precision_5, width, label='Precision@5', color='#FFB1B1', alpha=0.8)
    
    # Plot Recall@K
    ax.bar(x + 0.5*width, recall_3, width, label='Recall@3 (Any Match)', color='#4ECDC4', alpha=0.8)
    ax.bar(x + 1.5*width, recall_5, width, label='Recall@5', color='#6DD4D0', alpha=0.8)
    ax.bar(x + 2.5*width, recall_10, width, label='Recall@10 (Relaxed)', color='#8EDBD8', alpha=0.8)
    
    # Add value labels on bars
    all_data = [precision_1, precision_3, precision_5, recall_3, recall_5, recall_10]
    all_positions = [x - 2.5*width, x - 1.5*width, x - 0.5*width, x + 0.5*width, x + 1.5*width, x + 2.5*width]
    
    for data, positions in zip(all_data, all_positions):
        for pos, val in zip(positions, data):
            ax.text(pos, val + 1, f'{val:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Methods', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Precision@K (Ranking) vs Recall@K (Any Match)\nRed=Strict Ranking | Teal=Any Match', 
                 fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)

def print_comprehensive_summary(comprehensive_metrics: Dict[str, Any]):
    """Print comprehensive accuracy summary with all metrics."""
    
    if 'error' in comprehensive_metrics:
        print(f"❌ {comprehensive_metrics['error']}")
        return
    
    print(f"\n📊 COMPREHENSIVE ACCURACY SUMMARY")
    print(f"{'='*80}")
    
    # Overall statistics
    print(f"📈 Overall Performance:")
    print(f"   Total Methods: {comprehensive_metrics.get('total_methods', 0)}")
    print(f"   Total Queries: {comprehensive_metrics.get('total_queries', 0)}")
    
    # Average metrics
    print(f"\n📊 Average Accuracy Across All Methods:")
    print(f"   Precision@1 (Strict): {comprehensive_metrics.get('avg_precision_at_1', 0):.1%}")
    print(f"   Precision@3 (Moderate): {comprehensive_metrics.get('avg_precision_at_3', 0):.1%}")
    print(f"   Precision@5 (Relaxed): {comprehensive_metrics.get('avg_precision_at_5', 0):.1%}")
    print(f"   Recall@3 (Any Match): {comprehensive_metrics.get('avg_recall_at_3', 0):.1%}")
    print(f"   Recall@5 (Any Match): {comprehensive_metrics.get('avg_recall_at_5', 0):.1%}")
    print(f"   Recall@10 (Any Match): {comprehensive_metrics.get('avg_recall_at_10', 0):.1%}")
    print(f"   Coverage: {comprehensive_metrics.get('avg_coverage', 0):.1f}%")
    print(f"   Retrieval Success: {comprehensive_metrics.get('avg_retrieval_success', 0):.1%}")
    
    # Best performing methods
    print(f"\n🏆 Best Performing Methods by Category:")
    print(f"   🎯 Best Precision@1: {comprehensive_metrics.get('best_precision_1', 'N/A')} (strict ranking)")
    print(f"   🔍 Best Recall@5: {comprehensive_metrics.get('best_recall_5', 'N/A')} (any match in top-5)")
    print(f"   📊 Best Coverage: {comprehensive_metrics.get('best_coverage', 'N/A')} (finds most GT tables)")
    print(f"   ✅ Best Success Rate: {comprehensive_metrics.get('best_retrieval_success', 'N/A')} (most reliable)")
    
    # Method breakdown
    method_metrics = comprehensive_metrics.get('method_metrics', {})
    if method_metrics:
        print(f"\n📋 Detailed Method Performance:")
        print(f"{'Method':<25} {'P@1':<6} {'P@5':<6} {'R@5':<6} {'R@10':<6} {'Cov':<6} {'Success':<8} {'Recommendation'}")
        print(f"{'-'*95}")
        
        for method, stats in sorted(method_metrics.items(), 
                                  key=lambda x: x[1]['recall_at_5'], reverse=True):
            p1 = stats['precision_at_1']
            p5 = stats['precision_at_5']
            r5 = stats['recall_at_5']
            r10 = stats['recall_at_10']
            cov = stats['avg_coverage']
            success = stats['retrieval_success_rate']
            
            # Simple recommendation
            if p1 > 0.7:
                rec = "Precision Leader"
            elif r5 > 0.8:
                rec = "Recall Leader"
            elif cov > 75:
                rec = "Coverage Leader"
            elif success > 0.9:
                rec = "Reliable"
            else:
                rec = "Specialized"
            
            print(f"{method:<25} {p1:<6.1%} {p5:<6.1%} {r5:<6.1%} {r10:<6.1%} {cov:<6.1f} {success:<8.1%} {rec}")
    
    print(f"\n{'='*80}")
    print(f"📖 ACCURACY TYPE DEFINITIONS:")
    print(f"   • Precision@K: Main GT table appears in top K positions (ranking quality)")
    print(f"   • Recall@K: ANY GT table appears in top K results (coverage quality)")
    print(f"   • Coverage: % of all GT tables found (comprehensive discovery)")
    print(f"   • Success: Whether ANY useful table was found (minimum viability)")
    print(f"{'='*80}")