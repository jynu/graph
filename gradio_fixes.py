# Replace the table selection section in your create_text_to_sql_interface() function

# Step 2: Table Selection (REPLACE the existing Step 2 section)
with gr.Row():
    with gr.Column():
        gr.Markdown("### 📋 Step 2: Table Selection")
        tables_display = gr.DataFrame(
            headers=["Table Name", "Type", "Description", "Columns", "Selected"],
            label="🗄️ Discovered Tables"
        )
        
        selected_tables_input = gr.CheckboxGroup(
            choices=[],
            value=[],
            label="✅ Select tables to use for SQL generation",
            info="Choose which tables should be used for generating the SQL query"
        )
        
        update_selection_btn = gr.Button("✅ Update Selection", variant="secondary")
        selection_status = gr.Markdown("")

# REPLACE the event handlers section with this corrected version:

# === Event Handlers ===

# Step-by-step workflow handlers
def handle_table_search(query, max_tables):
    summary, df, table_details = text_to_table_search(query, max_tables)
    
    # Extract table names for CheckboxGroup
    if df is not None and not df.empty and "Table Name" in df.columns:
        table_names = df["Table Name"].tolist()
        # Return updated CheckboxGroup with new choices and all selected by default
        return summary, df, gr.CheckboxGroup(choices=table_names, value=table_names)
    else:
        return summary, df, gr.CheckboxGroup(choices=[], value=[])

search_tables_btn.click(
    fn=handle_table_search,
    inputs=[query_input, max_tables_slider],
    outputs=[table_search_results, tables_display, selected_tables_input]
)

update_selection_btn.click(
    fn=update_table_selection,
    inputs=[selected_tables_input],
    outputs=[selection_status]
)

def handle_sql_generation(selected_tables):
    if not selected_tables:
        return "❌ No tables selected. Please select at least one table.", "", ""
    return generate_sql_from_tables(selected_tables)

generate_sql_btn.click(
    fn=handle_sql_generation,
    inputs=[selected_tables_input],
    outputs=[sql_generation_results, generated_sql_display, sql_reasoning_display]
)

evaluate_sql_btn.click(
    fn=evaluate_generated_sql,
    inputs=[ground_truth_input],
    outputs=[evaluation_summary, evaluation_metrics, evaluation_details]
)