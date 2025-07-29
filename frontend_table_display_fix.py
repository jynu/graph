# In your text_to_sql_frontend.py file, replace the existing 
# create_table_summary_with_scores function with this improved version:

def create_table_summary_with_scores(tables: List[str], table_details: Dict) -> str:
    """Create a compact summary of found tables with similarity scores - one line per table."""
    if not tables:
        return "No tables found."
    
    summary_text = f"## 📊 Found {len(tables)} Relevant Tables (Ranked by Similarity)\n\n"
    
    # Create list of tables with their scores for sorting
    table_info = []
    for table_name in tables:
        if table_name in table_details and 'error' not in table_details[table_name]:
            details = table_details[table_name]
            score = details.get('similarity_score', 0.0)
            confidence = details.get('confidence', 'Unknown')
            rank = details.get('rank', 999)
            table_type = details.get('table_type', 'unknown')
            description = details.get('description', '')
            
            table_info.append({
                'name': table_name,
                'score': score,
                'confidence': confidence,
                'rank': rank,
                'type': table_type,
                'description': description
            })
    
    # Sort by rank
    table_info.sort(key=lambda x: x['rank'])
    
    # Group by confidence level
    confidence_groups = {
        'High': [t for t in table_info if t['confidence'] == 'High'],
        'Medium': [t for t in table_info if t['confidence'] == 'Medium'],
        'Low': [t for t in table_info if t['confidence'] == 'Low'],
        'Very Low': [t for t in table_info if t['confidence'] == 'Very Low']
    }
    
    for confidence_level, tables_in_group in confidence_groups.items():
        if not tables_in_group:
            continue
            
        # Confidence level emoji
        confidence_emoji = {
            'High': '🟢',
            'Medium': '🟡', 
            'Low': '🟠',
            'Very Low': '🔴'
        }
        
        summary_text += f"### {confidence_emoji[confidence_level]} {confidence_level} Confidence ({len(tables_in_group)} tables):\n\n"
        
        for table_info_item in tables_in_group:
            table_name = table_info_item['name']
            score = table_info_item['score']
            rank = table_info_item['rank']
            table_type = table_info_item['type']
            description = table_info_item['description']
            
            # Truncate table name if too long (keep it readable)
            max_name_length = 55
            if len(table_name) > max_name_length:
                display_name = table_name[:max_name_length-3] + "..."
            else:
                display_name = table_name
            
            # Calculate remaining space for description
            base_length = len(f"#{rank} ") + len(display_name) + len(f" ({table_type}) Score: {score:.3f}")
            remaining_space = 100 - base_length  # Target total line length
            
            # Truncate description to fit
            if description and len(description) > 0:
                if len(description) > remaining_space:
                    desc_text = f" - {description[:remaining_space-6]}..."
                else:
                    desc_text = f" - {description}"
            else:
                desc_text = ""
            
            # One clean line per table with rank always at start
            summary_text += f"**#{rank}** `{display_name}` ({table_type}) *Score: {score:.3f}*{desc_text}\n"
        
        summary_text += "\n"
    
    summary_text += """💡 **Tips:**
- **Higher scores** = more relevant to your query
- **Adjust similarity threshold** above to see more/fewer results  
- **All found tables** are ranked by relevance
- Click 'Detailed Table Information' below for full names and column details
"""
    
    return summary_text