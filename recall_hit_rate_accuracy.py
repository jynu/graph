# ================================
# RECALL@K AND HIT RATE@K ACCURACY
# ================================

"""
ACCURACY TYPES EXPLAINED:

1. **Precision@K (Top-K)**: Main table appears in top K positions
   - Top-1: Main table is #1 result
   - Top-3: Main table is in positions 1-3
   - Top-5: Main table is in positions 1-5

2. **Recall@K (Hit Rate)**: ANY ground truth table appears in top K results  
   - Recall@3: At least 1 GT table in top 3 results
   - Recall@5: At least 1 GT table in top 5 results
   - Recall@10: At least 1 GT table in top 10 results

3. **Coverage**: Percentage of all GT tables found (regardless of position)

For graph traversal: If method finds 10 tables and ANY matches GT → Success!
"""

async def evaluate_comprehensive_table_accuracy(query: str, predicted_tables: List[str], 
                                               ground_truth_sql: str) -> Dict[str, Any]:
    """Comprehensive evaluation including Precision@K, Recall@K, and Hit Rate."""
    
    if not predicted_tables or not ground_truth_sql:
        return create_empty_comprehensive_result(len(predicted_tables))
    
    # Enhanced prompt for comprehensive evaluation
    evaluation_prompt = f"""
You are a database expert. Evaluate table retrieval with MULTIPLE accuracy types.

**TASK BREAKDOWN:**
1. Extract ALL table names from the SQL
2. Identify the MOST IMPORTANT table for answering the query
3. Calculate different accuracy metrics

**User Query:** {query}

**Ground Truth SQL:** 
{ground_truth_sql}

**Predicted Tables (in order):**
{chr(10).join([f"{i+1}. {table}" for i, table in enumerate(predicted_tables)])}

**EVALUATION TYPES:**

**A) PRECISION@K (Ranking-based):**
- Does the MOST IMPORTANT table appear in top 1/3/5 positions?

**B) RECALL@K (Hit Rate):**
- Does ANY ground truth table appear in top 3/5/10 results?

**C) COVERAGE:**
- What percentage of ALL ground truth tables were found?

**RESPOND WITH ONLY THIS JSON:**
{{
    "ground_truth_tables": ["table1", "table2", "table3"],
    "most_important_table": "main_table_name",
    
    "precision_at_1": false,
    "precision_at_3": false,
    "precision_at_5": false,
    
    "recall_at_3": false,
    "recall_at_5": false,
    "recall_at_10": false,
    
    "hit_rate_at_3": false,
    "hit_rate_at_5": false,
    "hit_rate_at_10": false,
    
    "coverage_percentage": 0.0,
    "tables_found": 0,
    "tables_needed": 0,
    
    "confidence": 0.0,
    "explanation": "Brief explanation",
    "retrieval_success": false
}}

Note: recall_at_k and hit_rate_at_k are the same metric with different names."""
    
    try:
        response = await client_manager.ask_gpt(evaluation_prompt)
        logger.info(f"GPT Comprehensive Response: {response[:200]}...")
        
        # Parse the response
        parsed_result = parse_comprehensive_gpt_response(response, predicted_tables)
        
        # Add calculated metrics
        parsed_result = enhance_with_calculated_metrics(parsed_result, predicted_tables)
        
        return parsed_result
            
    except Exception as e:
        logger.error(f"GPT comprehensive evaluation failed: {e}")
        return create_empty_comprehensive_result(len(predicted_tables))

def parse_comprehensive_gpt_response(response: str, predicted_tables: List[str]) -> Dict[str, Any]:
    """Parse GPT response for comprehensive evaluation."""
    
    # Clean response
    response_clean = response.strip()
    if response_clean.startswith('```json'):
        response_clean = response_clean.replace('```json', '').replace('```', '').strip()
    
    # Find JSON content
    json_start = response_clean.find('{')
    json_end = response_clean.rfind('}') + 1
    
    if json_start >= 0 and json_end > json_start:
        json_content = response_clean[json_start:json_end]
    else:
        json_content = response_clean
    
    try:
        import json
        result = json.loads(json_content)
        
        # Standardize and validate
        standardized = {
            # Ground truth info
            'ground_truth_tables': result.get('ground_truth_tables', []),
            'most_important_table': result.get('most_important_table', ''),
            
            # Precision@K (ranking-based)
            'precision_at_1': bool(result.get('precision_at_1', False)),
            'precision_at_3': bool(result.get('precision_at_3', False)),
            'precision_at_5': bool(result.get('precision_at_5', False)),
            
            # Recall@K / Hit Rate@K (any match)
            'recall_at_3': bool(result.get('recall_at_3', False)),
            'recall_at_5': bool(result.get('recall_at_5', False)),
            'recall_at_10': bool(result.get('recall_at_10', False)),
            'hit_rate_at_3': bool(result.get('hit_rate_at_3', result.get('recall_at_3', False))),
            'hit_rate_at_5': bool(result.get('hit_rate_at_5', result.get('recall_at_5', False))),
            'hit_rate_at_10': bool(result.get('hit_rate_at_10', result.get('recall_at_10', False))),
            
            # Coverage metrics
            'coverage_percentage': max(0.0, min(100.0, float(result.get('coverage_percentage', 0)))),
            'tables_found': int(result.get('tables_found', 0)),
            'tables_needed': int(result.get('tables_needed', 0)),
            
            # Overall metrics
            'confidence': max(0.0, min(1.0, float(result.get('confidence', 0)))),
            'explanation': str(result.get('explanation', ''))[:200],
            'retrieval_success': bool(result.get('retrieval_success', False)),
            
            # Meta info
            'total_predicted': len(predicted_tables),
            'extraction_method': 'gpt_json'
        }
        
        return standardized
        
    except json.JSONDecodeError:
        logger.warning(f"JSON parsing failed for comprehensive evaluation, using fallback")
        return parse_comprehensive_fallback(response_clean, predicted_tables)

def parse_comprehensive_fallback(response: str, predicted_tables: List[str]) -> Dict[str, Any]:
    """Fallback parsing for comprehensive evaluation."""
    
    response_lower = response.lower()
    
    # Extract ground truth tables
    gt_tables = []
    table_pattern = r'ground_truth_tables.*?\[(.*?)\]'
    match = re.search(table_pattern, response, re.DOTALL)
    if match:
        tables_str = match.group(1)
        table_names = re.findall(r'"([^"]+)"', tables_str)
        gt_tables = table_names
    
    # Extract boolean values with multiple patterns
    def extract_boolean(metric_name: str) -> bool:
        patterns = [
            f'"{metric_name}":\s*true',
            f'{metric_name}.*true',
            f'{metric_name}.*yes'
        ]
        return any(re.search(pattern, response_lower) for pattern in patterns)
    
    # Extract metrics
    result = {
        'ground_truth_tables': gt_tables,
        'most_important_table': gt_tables[0] if gt_tables else '',
        
        # Precision@K
        'precision_at_1': extract_boolean('precision_at_1'),
        'precision_at_3': extract_boolean('precision_at_3'),
        'precision_at_5': extract_boolean('precision_at_5'),
        
        # Recall@K / Hit Rate@K
        'recall_at_3': extract_boolean('recall_at_3'),
        'recall_at_5': extract_boolean('recall_at_5'),
        'recall_at_10': extract_boolean('recall_at_10'),
        'hit_rate_at_3': extract_boolean('hit_rate_at_3') or extract_boolean('recall_at_3'),
        'hit_rate_at_5': extract_boolean('hit_rate_at_5') or extract_boolean('recall_at_5'),
        'hit_rate_at_10': extract_boolean('hit_rate_at_10') or extract_boolean('recall_at_10'),
        
        # Coverage
        'coverage_percentage': 0.0,
        'tables_found': len(gt_tables),
        'tables_needed': len(gt_tables),
        
        # Overall
        'confidence': 0.5,
        'explanation': f'Fallback parsing: {response[:100]}...',
        'retrieval_success': any([
            extract_boolean('recall_at_5'),
            extract_boolean('hit_rate_at_5'),
            extract_boolean('precision_at_5')
        ]),
        
        'total_predicted': len(predicted_tables),
        'extraction_method': 'fallback'
    }
    
    # Extract numerical values
    coverage_match = re.search(r'coverage_percentage.*?([0-9.]+)', response_lower)
    if coverage_match:
        result['coverage_percentage'] = max(0.0, min(100.0, float(coverage_match.group(1))))
    
    confidence_match = re.search(r'confidence.*?([0-9.]+)', response_lower)
    if confidence_match:
        result['confidence'] = max(0.0, min(1.0, float(confidence_match.group(1))))
    
    return result

def enhance_with_calculated_metrics(result: Dict[str, Any], predicted_tables: List[str]) -> Dict[str, Any]:
    """Add calculated metrics and validation."""
    
    enhanced = result.copy()
    gt_tables = result.get('ground_truth_tables', [])
    most_important = result.get('most_important_table', '')
    
    # Calculate actual positions and matches
    positions = {}
    matches = set()
    
    for gt_table in gt_tables:
        for i, pred_table in enumerate(predicted_tables):
            # Check for exact match or partial match
            if (gt_table.lower() == pred_table.lower() or 
                gt_table.lower() in pred_table.lower() or 
                pred_table.lower() in gt_table.lower()):
                positions[gt_table] = i + 1  # 1-indexed position
                matches.add(gt_table)
                break
    
    # Calculate Precision@K (ranking-based)
    if most_important and most_important in positions:
        pos = positions[most_important]
        enhanced['precision_at_1_calculated'] = pos <= 1
        enhanced['precision_at_3_calculated'] = pos <= 3
        enhanced['precision_at_5_calculated'] = pos <= 5
    else:
        enhanced['precision_at_1_calculated'] = False
        enhanced['precision_at_3_calculated'] = False
        enhanced['precision_at_5_calculated'] = False
    
    # Calculate Recall@K / Hit Rate@K (any match)
    any_match_in_3 = any(pos <= 3 for pos in positions.values())
    any_match_in_5 = any(pos <= 5 for pos in positions.values())
    any_match_in_10 = any(pos <= 10 for pos in positions.values())
    
    enhanced['recall_at_3_calculated'] = any_match_in_3
    enhanced['recall_at_5_calculated'] = any_match_in_5
    enhanced['recall_at_10_calculated'] = any_match_in_10
    enhanced['hit_rate_at_3_calculated'] = any_match_in_3  # Same as recall
    enhanced['hit_rate_at_5_calculated'] = any_match_in_5
    enhanced['hit_rate_at_10_calculated'] = any_match_in_10
    
    # Calculate Coverage
    total_gt_tables = len(gt_tables)
    found_gt_tables = len(matches)
    enhanced['coverage_calculated'] = (found_gt_tables / total_gt_tables * 100) if total_gt_tables > 0 else 0
    enhanced['tables_found_calculated'] = found_gt_tables
    enhanced['tables_needed_calculated'] = total_gt_tables
    
    # Overall retrieval success (any GT table found)
    enhanced['retrieval_success_calculated'] = len(matches) > 0
    
    # Store positions and matches for debugging
    enhanced['ground_truth_positions'] = positions
    enhanced['matched_tables'] = list(matches)
    enhanced['unmatched_tables'] = [gt for gt in gt_tables if gt not in matches]
    
    # Check for inconsistencies
    inconsistencies = []
    
    # Check Precision@K inconsistencies
    for k in [1, 3, 5]:
        gpt_key = f'precision_at_{k}'
        calc_key = f'precision_at_{k}_calculated'
        if enhanced.get(gpt_key) != enhanced.get(calc_key):
            inconsistencies.append(f"Precision@{k}: GPT={enhanced.get(gpt_key)}, Calc={enhanced.get(calc_key)}")
    
    # Check Recall@K inconsistencies
    for k in [3, 5, 10]:
        gpt_key = f'recall_at_{k}'
        calc_key = f'recall_at_{k}_calculated'
        if enhanced.get(gpt_key) != enhanced.get(calc_key):
            inconsistencies.append(f"Recall@{k}: GPT={enhanced.get(gpt_key)}, Calc={enhanced.get(calc_key)}")
    
    if inconsistencies:
        enhanced['metric_inconsistencies'] = '; '.join(inconsistencies)
        logger.warning(f"Metric inconsistencies detected: {enhanced['metric_inconsistencies']}")
    
    return enhanced

def create_empty_comprehensive_result(total_predicted: int) -> Dict[str, Any]:
    """Create empty comprehensive result structure."""
    return {
        # Ground truth info
        'ground_truth_tables': [],
        'most_important_table': '',
        
        # Precision@K (ranking-based)
        'precision_at_1': False,
        'precision_at_3': False,
        'precision_at_5': False,
        
        # Recall@K / Hit Rate@K (any match)
        'recall_at_3': False,
        'recall_at_5': False,
        'recall_at_10': False,
        'hit_rate_at_3': False,
        'hit_rate_at_5': False,
        'hit_rate_at_10': False,
        
        # Coverage metrics
        'coverage_percentage': 0.0,
        'tables_found': 0,
        'tables_needed': 0,
        
        # Overall metrics
        'confidence': 0.0,
        'explanation': 'No predicted tables or ground truth available',
        'retrieval_success': False,
        
        # Calculated versions
        'precision_at_1_calculated': False,
        'precision_at_3_calculated': False,
        'precision_at_5_calculated': False,
        'recall_at_3_calculated': False,
        'recall_at_5_calculated': False,
        'recall_at_10_calculated': False,
        'hit_rate_at_3_calculated': False,
        'hit_rate_at_5_calculated': False,
        'hit_rate_at_10_calculated': False,
        'coverage_calculated': 0.0,
        'retrieval_success_calculated': False,
        
        # Meta info
        'total_predicted': total_predicted,
        'ground_truth_positions': {},
        'matched_tables': [],
        'unmatched_tables': [],
        'extraction_method': 'empty'
    }