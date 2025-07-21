# Add to retriever initialization section (around line 600):

try:
    retrievers["Advanced Graph Traversal"] = AdvancedGraphTraversalRetriever(DB_PATH)
    print(f"  ✅ Advanced Graph Traversal initialized")
except Exception as e:
    print(f"  ❌ AdvancedGraphTraversalRetriever failed: {e}")

# Update the main query processing loop (around line 650):

for i, query_info in enumerate(queries, 1):
    query_id = query_info['id']
    question = query_info['question']
    ground_truth_sql = query_info.get('ground_truth_sql')
    source = query_info['source']
    
    print(f"\n--- [{i}/{len(queries)}] {query_id}: {question[:80]}... ---")
    
    query_results = {
        'Query_ID': query_id,
        'Question': question,
        'Ground_Truth_SQL': ground_truth_sql[:200] + '...' if ground_truth_sql and len(ground_truth_sql) > 200 else ground_truth_sql,
        'Source': source,
        'Question_Length': len(question),
        'method_results': {}  # Store detailed results for accuracy evaluation
    }
    
    for method_name, retriever in retrievers.items():
        try:
            start_time = datetime.now()
            tables = retriever.get_tables(question)
            end_time = datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            
            # Clean and format table names
            if tables:
                unique_tables = []
                seen = set()
                for table in tables:
                    if table and table.strip() and table.strip() not in seen:
                        clean_table = table.strip()
                        unique_tables.append(clean_table)
                        seen.add(clean_table)
                tables = unique_tables
            
            tables_str = "; ".join(tables) if tables else "No tables found"
            query_results[f'{method_name}_Tables'] = tables_str
            query_results[f'{method_name}_Count'] = len(tables)
            query_results[f'{method_name}_Duration_sec'] = round(duration, 3)
            
            # Store for accuracy evaluation
            query_results['method_results'][method_name] = {
                'tables': tables,
                'count': len(tables),
                'duration': duration
            }
            
            # Accuracy evaluation using GPT (if ground truth available)
            if ground_truth_sql and tables:
                try:
                    print(f"      🔍 Evaluating accuracy for {method_name}...")
                    evaluation = await evaluate_table_accuracy_with_gpt(question, tables, ground_truth_sql)
                    
                    # Add accuracy results to query_results
                    query_results[f'{method_name}_Accuracy'] = evaluation.get('is_correct', False)
                    query_results[f'{method_name}_Confidence'] = round(evaluation.get('confidence', 0), 3)
                    query_results[f'{method_name}_Relevant_Tables'] = evaluation.get('relevant_tables_found', 0)
                    query_results[f'{method_name}_Accuracy_Explanation'] = evaluation.get('explanation', '')[:100]
                    
                    # Store detailed evaluation
                    query_results['method_results'][method_name]['evaluation'] = evaluation
                    
                    accuracy_indicator = "✅" if evaluation.get('is_correct') else "❌"
                    confidence = evaluation.get('confidence', 0)
                    print(f"      {accuracy_indicator} Accuracy: {evaluation.get('is_correct')} (confidence: {confidence:.2f})")
                    
                except Exception as eval_error:
                    print(f"      ⚠️ Accuracy evaluation failed: {eval_error}")
                    query_results[f'{method_name}_Accuracy'] = None
                    query_results[f'{method_name}_Confidence'] = 0
                    query_results[f'{method_name}_Relevant_Tables'] = 0
                    query_results[f'{method_name}_Accuracy_Explanation'] = f"Evaluation failed: {str(eval_error)}"
            
            # Update performance tracking
            method_performance[method_name]['total_time'] += duration
            method_performance[method_name]['total_tables'] += len(tables)
            if len(tables) > 0:
                method_performance[method_name]['success_count'] += 1
            
            # Enhanced output
            print(f"    🔍 {method_name:20}: {len(tables):2d} tables ({duration:5.3f}s)")
            
        except Exception as e:
            error_msg = f"ERROR: {str(e)}"
            query_results[f'{method_name}_Tables'] = error_msg
            query_results[f'{method_name}_Count'] = 0
            query_results[f'{method_name}_Duration_sec'] = 0
            query_results[f'{method_name}_Accuracy'] = False
            query_results[f'{method_name}_Confidence'] = 0
            query_results[f'{method_name}_Relevant_Tables'] = 0
            query_results[f'{method_name}_Accuracy_Explanation'] = error_msg
            print(f"    ❌ {method_name:20}: FAILED - {e}")
    
    results.append(query_results)