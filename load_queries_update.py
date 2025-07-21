def load_queries_from_excel(file_path: str) -> List[Dict[str, str]]:
    """Load user queries and ground truth answers from Excel file."""
    try:
        # Read the feedback report sheet
        df = pd.read_excel(file_path, sheet_name='feedback_report')
        
        # Print column names for debugging
        logger.info(f"📋 Available columns in Excel: {list(df.columns)}")
        
        # Find the question column
        question_column = None
        possible_question_columns = ['QUESTION', 'Question', 'question', 'QUERY', 'Query', 'query']
        
        for col in possible_question_columns:
            if col in df.columns:
                question_column = col
                logger.info(f"✅ Found question column: '{col}'")
                break
        
        # Find the answer column
        answer_column = None
        possible_answer_columns = ['ANSWER', 'Answer', 'answer', 'SQL', 'sql', 'GROUND_TRUTH', 'Ground_Truth']
        
        for col in possible_answer_columns:
            if col in df.columns:
                answer_column = col
                logger.info(f"✅ Found answer column: '{col}'")
                break
        
        if question_column is None:
            logger.error(f"❌ No question column found. Available columns: {list(df.columns)}")
            return []
        
        # Extract questions and answers
        queries = []
        valid_questions = 0
        
        for idx, row in df.iterrows():
            question = row[question_column]
            answer = row[answer_column] if answer_column and answer_column in row and pd.notna(row[answer_column]) else None
            
            if pd.notna(question) and str(question).strip():
                question_text = str(question).strip()
                if len(question_text) > 10:  # Filter out very short questions
                    queries.append({
                        'id': f"Q{idx+1}",
                        'question': question_text,
                        'ground_truth_sql': str(answer).strip() if answer else None,
                        'source': 'Excel',
                        'row_number': idx + 1
                    })
                    valid_questions += 1
        
        logger.info(f"✅ Loaded {len(queries)} valid queries from Excel file")
        
        # Show statistics
        queries_with_ground_truth = sum(1 for q in queries if q['ground_truth_sql'])
        logger.info(f"📊 Queries with ground truth SQL: {queries_with_ground_truth}/{len(queries)}")
        
        return queries
        
    except FileNotFoundError:
        logger.error(f"❌ Excel file not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"❌ Failed to load queries from Excel: {e}")
        return []