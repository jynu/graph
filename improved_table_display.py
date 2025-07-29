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
    
    # Sort by rank (already sorted, but ensure consistency)
    table_info.sort(key=lambda x: x['rank'])
    
    # Group by confidence level for better presentation
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
            
            # Truncate table name if too long
            display_name = table_name if len(table_name) <= 50 else table_name[:47] + "..."
            
            # Truncate description to fit in one line
            max_desc_length = 80  # Adjust based on your UI width
            if description and len(description) > 0:
                if len(description) > max_desc_length:
                    desc_text = f" - {description[:max_desc_length-3]}..."
                else:
                    desc_text = f" - {description}"
            else:
                desc_text = ""
            
            # Format everything in one line with consistent spacing
            summary_text += f"**#{rank}** `{display_name}` ({table_type}) *Score: {score:.3f}*{desc_text}\n"
        
        summary_text += "\n"
    
    summary_text += """
💡 **Tips:**
- **Higher scores** = more relevant to your query
- **Adjust similarity threshold** above to see more/fewer results  
- **All found tables** are ranked by relevance
- Click 'Detailed Table Information' below for column details
"""
    
    return summary_text

# Alternative version with even more compact display
def create_table_summary_ultra_compact(tables: List[str], table_details: Dict) -> str:
    """Ultra-compact table summary - minimal info per line."""
    if not tables:
        return "No tables found."
    
    summary_text = f"## 📊 Found {len(tables)} Relevant Tables\n\n"
    
    # Create table with all info in one line each
    summary_text += "| Rank | Table Name | Type | Score | Confidence |\n"
    summary_text += "|------|------------|------|-------|------------|\n"
    
    for table_name in tables:
        if table_name in table_details and 'error' not in table_details[table_name]:
            details = table_details[table_name]
            score = details.get('similarity_score', 0.0)
            confidence = details.get('confidence', 'Unknown')
            rank = details.get('rank', 999)
            table_type = details.get('table_type', 'unknown')
            
            # Truncate table name for table display
            display_name = table_name if len(table_name) <= 35 else table_name[:32] + "..."
            
            # Confidence emoji
            conf_emoji = {
                'High': '🟢', 'Medium': '🟡', 'Low': '🟠', 'Very Low': '🔴'
            }.get(confidence, '⚪')
            
            summary_text += f"| #{rank} | `{display_name}` | {table_type} | {score:.3f} | {conf_emoji} {confidence} |\n"
    
    summary_text += "\n💡 Click 'Detailed Table Information' below for full names and column details.\n"
    
    return summary_text

# Most user-friendly version - clean list format
def create_table_summary_clean_list(tables: List[str], table_details: Dict) -> str:
    """Clean list format - one clean line per table."""
    if not tables:
        return "No tables found."
    
    summary_text = f"## 📊 Found {len(tables)} Relevant Tables (Ranked by Similarity)\n\n"
    
    for i, table_name in enumerate(tables, 1):
        if table_name in table_details and 'error' not in table_details[table_name]:
            details = table_details[table_name]
            score = details.get('similarity_score', 0.0)
            confidence = details.get('confidence', 'Unknown')
            table_type = details.get('table_type', 'table')
            description = details.get('description', '')
            
            # Confidence emoji
            conf_emoji = {
                'High': '🟢', 'Medium': '🟡', 'Low': '🟠', 'Very Low': '🔴'
            }.get(confidence, '⚪')
            
            # Truncate table name if very long
            if len(table_name) > 60:
                display_name = table_name[:57] + "..."
            else:
                display_name = table_name
            
            # Truncate description to fit remainder of line
            remaining_space = 120 - len(display_name) - len(table_type) - 20  # Rough calculation
            if description and len(description) > remaining_space:
                desc_text = description[:remaining_space-3] + "..."
            else:
                desc_text = description
            
            # One clean line per table
            summary_text += f"{conf_emoji} **#{i}** `{display_name}` *({table_type}, {score:.3f})* {desc_text}\n"
        else:
            summary_text += f"❌ **#{i}** `{table_name}` *(error)*\n"
    
    summary_text += f"""
---
💡 **Legend:** 🟢 High confidence • 🟡 Medium • 🟠 Low • 🔴 Very Low  
📋 Click 'Detailed Table Information' for full names and column details
"""
    
    return summary_text