# REPLACE your existing update_table_selection function with this:

def update_table_selection(selected_tables: List[str]) -> str:
    """Update selected tables."""
    global current_session
    
    if not selected_tables:
        current_session["selected_tables"] = []
        return "⚠️ No tables selected."
    
    current_session["selected_tables"] = selected_tables
    
    return f"✅ Updated selection: {len(selected_tables)} tables selected: {', '.join([f'`{t}`' for t in selected_tables])}"

# ALSO UPDATE the text_to_table_search function to ensure proper table name handling:

def text_to_table_search(query: str, max_tables: int = 10) -> Tuple[str, pd.DataFrame, Dict]:
    """Search for relevant tables using Advanced Graph Traversal."""
    global current_session
    
    if not query.strip():
        return "Please enter a query.", pd.DataFrame(), {}
    
    logger.info(f"🔍 Text-to-Table search: {query}")
    
    try:
        # Call text-to-table API
        api_data = {
            "query": query,
            "user_id": DEFAULT_USER_ID,
            "max_tables": max_tables
        }
        
        result = call_api_sync("/text-to-table", api_data)
        
        if result.get("success", False):
            tables = result.get("tables", [])
            table_details = result.get("table_details", {})
            processing_time = result.get("processing_time", 0.0)
            
            # Ensure tables is a list of strings
            if isinstance(tables, list):
                tables = [str(table) for table in tables]
            else:
                tables = []
            
            # Update session
            current_session["query"] = query
            current_session["tables"] = tables
            current_session["selected_tables"] = tables.copy()  # Select all by default
            current_session["table_details"] = table_details
            
            # Create summary
            summary = f"""## 🗄️ Table Discovery Results

**Query:** {query}

**Method:** Advanced Graph Traversal (GNN + RL + Multi-level)

**Performance:**
- Found {len(tables)} relevant tables
- Processing time: {processing_time:.3f} seconds

**Status:** ✅ Success

### 📋 Selected Tables:
{', '.join([f'`{table}`' for table in tables]) if tables else 'No tables found'}

**Next Step:** Review the table selections below and click "Generate SQL" to create the SQL query.
"""
            
            # Create DataFrame for table display
            table_data = []
            for table_name in tables:
                details = table_details.get(table_name, {})
                table_data.append({
                    "Table Name": str(table_name),
                    "Type": details.get("table_type", ""),
                    "Description": details.get("description", "")[:100] + "..." if len(details.get("description", "")) > 100 else details.get("description", ""),
                    "Columns": len(details.get("columns", [])),
                    "Selected": "✅"
                })
            
            df = pd.DataFrame(table_data)
            
            return summary, df, table_details
            
        else:
            error_msg = result.get("error", "Unknown error")
            return f"❌ Error: {error_msg}", pd.DataFrame(), {}
            
    except Exception as e:
        logger.error(f"Text-to-table search failed: {e}")
        return f"❌ Error: {str(e)}", pd.DataFrame(), {}