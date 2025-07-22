# REPLACE YOUR ACCURACY EVALUATION SECTION WITH THIS COMPREHENSIVE VERSION:

# Comprehensive accuracy evaluation (if ground truth available)
if ground_truth_sql and tables:
    try:
        print(f"      🔍 Evaluating comprehensive accuracy for {method_name}...")
        
        # Use comprehensive evaluation (Precision@K + Recall@K + Hit Rate)
        evaluation = asyncio.run(evaluate_comprehensive_table_accuracy(question, tables, ground_truth_sql))
        
        # Store ALL accuracy metrics
        
        # PRECISION@K (Ranking-based): Main table in top K positions
        query_results[f'{method_name}_Precision_At_1'] = evaluation.get('precision_at_1', False)
        query_results[f'{method_name}_Precision_At_3'] = evaluation.get('precision_at_3', False)
        query_results[f'{method_name}_Precision_At_5'] = evaluation.get('precision_at_5', False)
        
        # RECALL@K / HIT RATE@K (Any match): ANY GT table in top K results
        query_results[f'{method_name}_Recall_At_3'] = evaluation.get('recall_at_3', False)
        query_results[f'{method_name}_Recall_At_5'] = evaluation.get('recall_at_5', False)
        query_results[f'{method_name}_Recall_At_10'] = evaluation.get('recall_at_10', False)
        query_results[f'{method_name}_Hit_Rate_At_3'] = evaluation.get('hit_rate_at_3', False)
        query_results[f'{method_name}_Hit_Rate_At_5'] = evaluation.get('hit_rate_at_5', False)
        query_results[f'{method_name}_Hit_Rate_At_10'] = evaluation.get('hit_rate_at_10', False)
        
        # COVERAGE: Percentage of all GT tables found
        query_results[f'{method_name}_Coverage_%'] = round(evaluation.get('coverage_percentage', 0), 1)
        query_results[f'{method_name}_Tables_Found'] = evaluation.get('tables_found', 0)
        query_results[f'{method_name}_Tables_Needed'] = evaluation.get('tables_needed', 0)
        
        # OVERALL SUCCESS: Any useful table found
        query_results[f'{method_name}_Retrieval_Success'] = evaluation.get('retrieval_success', False)
        
        # Legacy compatibility
        query_results[f'{method_name}_Accuracy'] = evaluation.get('precision_at_1', False)  # Most strict
        query_results[f'{method_name}_Confidence'] = round(evaluation.get('confidence', 0), 3)
        query_results[f'{method_name}_Accuracy_Explanation'] = evaluation.get('explanation', '')[:150]
        
        # Additional metrics
        query_results[f'{method_name}_Ground_Truth_Tables'] = '; '.join(evaluation.get('ground_truth_tables', []))
        query_results[f'{method_name}_Matched_Tables'] = '; '.join(evaluation.get('matched_tables', []))
        query_results[f'{method_name}_Unmatched_Tables'] = '; '.join(evaluation.get('unmatched_tables', []))
        
        # Position information
        positions = evaluation.get('ground_truth_positions', {})
        if positions:
            pos_str = '; '.join([f"{table}@{pos}" for table, pos in positions.items()])
            query_results[f'{method_name}_GT_Positions'] = pos_str
        else:
            query_results[f'{method_name}_GT_Positions'] = 'No matches found'
        
        # Debug info
        if 'metric_inconsistencies' in evaluation:
            query_results[f'{method_name}_Metric_Issues'] = evaluation['metric_inconsistencies']
        
        # Store detailed evaluation
        query_results['method_results'][method_name]['evaluation'] = evaluation
        
        # Enhanced reporting with ALL metrics
        print(f"      📊 COMPREHENSIVE ACCURACY RESULTS:")
        
        # Precision@K (Ranking-based)
        p1_indicator = "✅" if evaluation.get('precision_at_1') else "❌"
        p3_indicator = "✅" if evaluation.get('precision_at_3') else "❌"
        p5_indicator = "✅" if evaluation.get('precision_at_5') else "❌"
        print(f"         🎯 Precision@K (Main table ranking):")
        print(f"            {p1_indicator} P@1: {evaluation.get('precision_at_1')} (Main table is #1)")
        print(f"            {p3_indicator} P@3: {evaluation.get('precision_at_3')} (Main table in top-3)")
        print(f"            {p5_indicator} P@5: {evaluation.get('precision_at_5')} (Main table in top-5)")
        
        # Recall@K / Hit Rate (Any match)
        r3_indicator = "✅" if evaluation.get('recall_at_3') else "❌"
        r5_indicator = "✅" if evaluation.get('recall_at_5') else "❌"
        r10_indicator = "✅" if evaluation.get('recall_at_10') else "❌"
        print(f"         🔍 Recall@K / Hit Rate (Any GT table found):")
        print(f"            {r3_indicator} R@3: {evaluation.get('recall_at_3')} (Any GT in top-3)")
        print(f"            {r5_indicator} R@5: {evaluation.get('recall_at_5')} (Any GT in top-5)")
        print(f"            {r10_indicator} R@10: {evaluation.get('recall_at_10')} (Any GT in top-10)")
        
        # Coverage and Success
        coverage = evaluation.get('coverage_percentage', 0)
        found = evaluation.get('tables_found', 0)
        needed = evaluation.get('tables_needed', 0)
        success = evaluation.get('retrieval_success', False)
        success_indicator = "✅" if success else "❌"
        
        print(f"         📈 Coverage & Success:")
        print(f"            📊 Coverage: {coverage:.1f}% ({found}/{needed} GT tables found)")
        print(f"            {success_indicator} Retrieval Success: {success} (Any useful table found)")
        
        # Ground Truth Info
        gt_tables = evaluation.get('ground_truth_tables', [])
        matched = evaluation.get('matched_tables', [])
        print(f"         🎯 Ground Truth: {gt_tables}")
        print(f"         ✅ Matched: {matched}")
        
        if positions:
            print(f"         📍 Positions: {positions}")
        
        # Log issues
        if 'metric_inconsistencies' in evaluation:
            print(f"         ⚠️  Inconsistencies: {evaluation['metric_inconsistencies']}")
        
        print(f"         🤖 Confidence: {evaluation.get('confidence', 0):.2f}")
        
    except Exception as eval_error:
        print(f"      ❌ Comprehensive accuracy evaluation failed: {eval_error}")
        # Set all metrics to None/False
        
        # Precision@K
        for k in [1, 3, 5]:
            query_results[f'{method_name}_Precision_At_{k}'] = None
        
        # Recall@K / Hit Rate@K
        for k in [3, 5, 10]:
            query_results[f'{method_name}_Recall_At_{k}'] = None
            query_results[f'{method_name}_Hit_Rate_At_{k}'] = None
        
        # Coverage and Success
        query_results[f'{method_name}_Coverage_%'] = 0
        query_results[f'{method_name}_Tables_Found'] = 0
        query_results[f'{method_name}_Tables_Needed'] = 0
        query_results[f'{method_name}_Retrieval_Success'] = False
        
        # Legacy
        query_results[f'{method_name}_Accuracy'] = None
        query_results[f'{method_name}_Confidence'] = 0
        query_results[f'{method_name}_Accuracy_Explanation'] = f"Evaluation failed: {str(eval_error)}"