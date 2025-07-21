async def evaluate_table_accuracy_with_gpt(query: str, predicted_tables: List[str], 
                                          ground_truth_sql: str) -> Dict[str, Any]:
    """Use GPT to evaluate if predicted tables are correct for the given query and ground truth SQL."""
    
    if not predicted_tables or not ground_truth_sql:
        return {
            'is_correct': False,
            'confidence': 0.0,
            'explanation': 'No predicted tables or ground truth available',
            'relevant_tables_found': 0,
            'total_predicted': len(predicted_tables)
        }
    
    # Extract table names from ground truth SQL
    actual_tables = extract_tables_from_sql(ground_truth_sql)
    
    evaluation_prompt = f"""
You are a database expert evaluating table retrieval accuracy. 

**Task**: Determine if the predicted tables are correct for answering the given query, based on the ground truth SQL.

**User Query**: {query}

**Ground Truth SQL**: {ground_truth_sql}

**Predicted Tables**: {', '.join(predicted_tables)}

**Actual Tables in SQL**: {', '.join(actual_tables) if actual_tables else 'Unable to extract'}

**Evaluation Criteria**:
1. Do the predicted tables contain the main tables needed to answer the query?
2. Are the predicted tables relevant to the business question?
3. Would these tables provide sufficient data to construct a similar SQL query?

**Response Format** (JSON):
{{
    "is_correct": true/false,
    "confidence": 0.0-1.0,
    "explanation": "Brief explanation of the evaluation",
    "relevant_tables_found": number_of_relevant_tables_in_prediction,
    "total_predicted": {len(predicted_tables)},
    "missing_key_tables": ["table1", "table2"],
    "unnecessary_tables": ["table3"],
    "overall_quality": "excellent/good/fair/poor"
}}

Provide only the JSON response."""
    
    try:
        response = await client_manager.ask_gpt(evaluation_prompt)
        
        # Parse JSON response
        import json
        try:
            result = json.loads(response.strip())
            
            # Validate required fields
            required_fields = ['is_correct', 'confidence', 'explanation', 'relevant_tables_found']
            for field in required_fields:
                if field not in result:
                    result[field] = 0 if field == 'relevant_tables_found' or field == 'confidence' else False if field == 'is_correct' else 'GPT response incomplete'
            
            # Ensure confidence is between 0 and 1
            result['confidence'] = max(0.0, min(1.0, float(result.get('confidence', 0))))
            
            return result
            
        except json.JSONDecodeError:
            # Fallback: try to extract key information from text response
            is_correct = any(word in response.lower() for word in ['true', 'correct', 'yes', 'accurate'])
            return {
                'is_correct': is_correct,
                'confidence': 0.5,
                'explanation': f'GPT response (non-JSON): {response[:200]}...',
                'relevant_tables_found': len(predicted_tables) if is_correct else 0,
                'total_predicted': len(predicted_tables)
            }
            
    except Exception as e:
        logger.error(f"GPT evaluation failed: {e}")
        return {
            'is_correct': False,
            'confidence': 0.0,
            'explanation': f'Evaluation failed: {str(e)}',
            'relevant_tables_found': 0,
            'total_predicted': len(predicted_tables)
        }

def extract_tables_from_sql(sql: str) -> List[str]:
    """Extract table names from SQL query using regex."""
    if not sql:
        return []
    
    # Common patterns for table names in SQL
    patterns = [
        r'\bFROM\s+([A-Za-z_][A-Za-z0-9_.]*)',  # FROM table_name
        r'\bJOIN\s+([A-Za-z_][A-Za-z0-9_.]*)',  # JOIN table_name
        r'\bINTO\s+([A-Za-z_][A-Za-z0-9_.]*)',  # INTO table_name
        r'\bUPDATE\s+([A-Za-z_][A-Za-z0-9_.]*)', # UPDATE table_name
    ]
    
    tables = set()
    sql_upper = sql.upper()
    
    for pattern in patterns:
        matches = re.findall(pattern, sql_upper, re.IGNORECASE)
        for match in matches:
            # Clean table name (remove schema prefix if exists)
            table_name = match.strip()
            if '.' in table_name:
                table_name = table_name.split('.')[-1]  # Take only table name, not schema
            tables.add(table_name)
    
    return list(tables)

def calculate_accuracy_metrics(evaluation_results: List[Dict]) -> Dict[str, float]:
    """Calculate various accuracy metrics from evaluation results."""
    
    if not evaluation_results:
        return {}
    
    total_queries = len(evaluation_results)
    
    # Top-K accuracy calculation
    def calculate_top_k_accuracy(k: int) -> float:
        correct = 0
        for result in evaluation_results:
            method_results = result.get('method_results', {})
            for method_name, method_data in method_results.items():
                if method_data.get('evaluation'):
                    # Check if any of the top-k predictions were correct
                    predicted_tables = method_data.get('tables', [])[:k]
                    if predicted_tables and method_data['evaluation'].get('is_correct', False):
                        correct += 1
                        break  # Count this query as correct
        return correct / total_queries if total_queries > 0 else 0.0
    
    # Precision and Recall calculation
    def calculate_precision_recall() -> Tuple[float, float]:
        total_precision = 0
        total_recall = 0
        valid_evaluations = 0
        
        for result in evaluation_results:
            method_results = result.get('method_results', {})
            for method_name, method_data in method_results.items():
                evaluation = method_data.get('evaluation')
                if evaluation:
                    relevant_found = evaluation.get('relevant_tables_found', 0)
                    total_predicted = evaluation.get('total_predicted', 0)
                    
                    # Extract actual tables from ground truth
                    ground_truth_sql = result.get('ground_truth_sql', '')
                    actual_tables = extract_tables_from_sql(ground_truth_sql)
                    total_actual = len(actual_tables)
                    
                    if total_predicted > 0:
                        precision = relevant_found / total_predicted
                        total_precision += precision
                    
                    if total_actual > 0:
                        recall = relevant_found / total_actual
                        total_recall += recall
                    
                    valid_evaluations += 1
        
        avg_precision = total_precision / valid_evaluations if valid_evaluations > 0 else 0.0
        avg_recall = total_recall / valid_evaluations if valid_evaluations > 0 else 0.0
        
        return avg_precision, avg_recall
    
    # Calculate metrics
    metrics = {
        'top_1_accuracy': calculate_top_k_accuracy(1),
        'top_3_accuracy': calculate_top_k_accuracy(3),
        'top_5_accuracy': calculate_top_k_accuracy(5),
        'total_queries_evaluated': total_queries
    }
    
    # Add precision and recall
    precision, recall = calculate_precision_recall()
    metrics.update({
        'average_precision': precision,
        'average_recall': recall,
        'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    })
    
    return metrics