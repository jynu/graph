Fix 1: Add Missing Example Buttons and Variables
In the create_text_to_sql_interface() function, add the missing example buttons section:
pythondef create_text_to_sql_interface():
    """Create the complete text-to-SQL Gradio interface."""
    
    # Initialize system
    system_ready, init_message = initialize_system()
    
    with gr.Blocks(
        title="DC Chat Text-to-SQL System",
        theme=gr.themes.Soft(),
        css="""
        .step-container { border: 2px solid #e5e7eb; border-radius: 8px; padding: 16px; margin: 8px 0; }
        .step-active { border-color: #3b82f6; background-color: #eff6ff; }
        .step-completed { border-color: #10b981; background-color: #ecfdf5; }
        .sql-code { font-family: 'Courier New', monospace; background-color: #f8fafc; }
        """
    ) as demo:
        
        # Header
        gr.Markdown("""
        # 🚀 DC Chat Text-to-SQL System
        
        **Intelligent SQL generation using GNN + RL + Multi-level reasoning**
        
        This system follows a three-step workflow:
        1. **🔍 Table Discovery** - Find relevant tables using Advanced Graph Traversal
        2. **⚡ SQL Generation** - Generate optimized SQL using GPT-4
        3. **📊 Quality Evaluation** - Evaluate SQL quality against ground truth
        """)
        
        # System status
        with gr.Row():
            system_status = gr.Markdown(f"**System Status:** {init_message}")
        
        # Database information
        with gr.Row():
            with gr.Column():
                db_info_btn = gr.Button("🔄 Refresh Database Info", size="sm")
                db_info_display = gr.Markdown(get_database_info_display())
        
        # Main workflow
        gr.Markdown("## 📋 Main Workflow")
        
        # Step 1: Query Input and Table Discovery
        with gr.Group():
            gr.Markdown("### 🔍 Step 1: Enter Query and Discover Tables")
            
            with gr.Row():
                with gr.Column(scale=3):
                    query_input = gr.Textbox(
                        label="💭 Natural Language Query",
                        placeholder="Example: Fetch us a list of all quotes for equities in APAC region",
                        lines=3,
                        value=""
                    )
                    
                    with gr.Row():
                        max_tables_slider = gr.Slider(
                            minimum=5,
                            maximum=20,
                            value=10,
                            step=1,
                            label="Max Tables to Find"
                        )
                        similarity_threshold_slider = gr.Slider(
                            minimum=0.05,
                            maximum=0.8,
                            value=0.2,
                            step=0.05,
                            label="Similarity Threshold",
                            info="Lower = more results, Higher = more precise"
                        )
                    
                    find_tables_btn = gr.Button("🔍 Find Tables", variant="primary")
                
                with gr.Column(scale=2):
                    # Example queries - THIS WAS MISSING!
                    gr.Markdown("**Quick Examples:**")
                    example_buttons = []
                    for i, example in enumerate(EXAMPLE_QUERIES[:4]):
                        btn = gr.Button(f"📝 {example[:40]}...", size="sm")
                        example_buttons.append(btn)
            
            # Results for Step 1 - THESE WERE MISSING!
            with gr.Row():
                with gr.Column():
                    step1_status = gr.Markdown("")
                    
                    # Quick summary of found tables
                    table_summary = gr.Markdown("", label="📊 Found Tables Summary")
                    
                    # Collapsible detailed table information
                    with gr.Accordion("📋 Detailed Table Information (Click to expand)", open=False):
                        table_details_display = gr.Markdown("", label="Table Details with Columns")
                    
                    table_selection = gr.CheckboxGroup(
                        choices=[],
                        label="✅ Select Tables for SQL Generation",
                        info="All found tables are selected by default. Uncheck to exclude."
                    )
        
        # Step 2: SQL Generation
        with gr.Group():
            gr.Markdown("### ⚡ Step 2: Generate SQL from Selected Tables")
            
            with gr.Row():
                generate_sql_btn = gr.Button("⚡ Generate SQL", variant="primary", size="lg")
                
            with gr.Row():
                with gr.Column():
                    sql_results_display = gr.Markdown("", label="🎯 Generated SQL & Reasoning")
                
                with gr.Column():
                    sql_code_display = gr.Code(
                        language="sql",
                        label="📝 SQL Code",
                        interactive=True
                    )
        
        # Step 3: SQL Evaluation
        with gr.Group():
            gr.Markdown("### 📊 Step 3: Evaluate SQL Quality (Optional)")
            
            with gr.Row():
                with gr.Column():
                    ground_truth_input = gr.Code(
                        language="sql",
                        label="🎯 Ground Truth SQL",
                        value="-- Enter the expected/correct SQL query for comparison...",
                        lines=8
                    )
                    
                    evaluate_sql_btn = gr.Button("📊 Evaluate SQL Quality", variant="secondary")
                
                with gr.Column():
                    evaluation_results_display = gr.Markdown("", label="📈 Evaluation Results")
        
        # Performance and utilities
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📈 Performance Metrics")
                performance_table = gr.DataFrame(
                    headers=["Step", "Method", "Tables Found", "Status", "Threshold", "Timestamp"],  # Added "Threshold"
                    label="🏃 Performance Tracking"
                )
            
            with gr.Column():
                gr.Markdown("### 🛠️ Utilities")
                step_guidance = gr.Markdown("Enter a query to begin the text-to-SQL workflow.")
                
                with gr.Row():
                    export_btn = gr.Button("💾 Export Results", size="sm")
                    clear_btn = gr.Button("🗑️ Clear All", size="sm")
                
                export_status = gr.Markdown("")
        
        # === Event Handlers ===
        
        # Database info refresh
        db_info_btn.click(
            fn=get_database_info_display,
            outputs=db_info_display
        )
        
        # Example query buttons - FIX THE EVENT HANDLERS
        for i, btn in enumerate(example_buttons):
            btn.click(
                fn=lambda idx=i: set_example_query(idx),
                outputs=query_input
            )
        
        # Step 1: Find tables
        find_tables_btn.click(
            fn=step1_find_tables,
            inputs=[query_input, max_tables_slider, similarity_threshold_slider],
            outputs=[
                step1_status,
                table_summary,
                table_details_display,
                table_selection,
                performance_table,
                step_guidance
            ]
        )
        
        # Step 2: Generate SQL
        generate_sql_btn.click(
            fn=step2_generate_sql,
            inputs=[table_selection],
            outputs=[
                sql_results_display,
                sql_code_display,
                step_guidance
            ]
        )
        
        # Step 3: Evaluate SQL
        evaluate_sql_btn.click(
            fn=step3_evaluate_sql,
            inputs=[ground_truth_input],
            outputs=[
                evaluation_results_display,
                step_guidance
            ]
        )
        
        # Utilities
        export_btn.click(
            fn=export_results,
            outputs=export_status
        )
        
        clear_btn.click(
            fn=clear_all_data,
            outputs=[
                query_input,
                step1_status,
                table_summary,
                table_details_display,
                table_selection,
                performance_table,
                step_guidance,
                sql_results_display,
                sql_code_display,
                ground_truth_input,
                evaluation_results_display,
                export_status
            ]
        )
        
        # Footer
        gr.Markdown("""
        ---
        ### 🔧 Technical Features:
        
        **Advanced Graph Traversal:**
        - Graph Neural Networks with attention mechanisms
        - Reinforcement Learning for path optimization  
        - Multi-level reasoning (semantic, structural, global)
        - Ensemble combination of multiple algorithms
        
        **SQL Generation:**
        - GPT-4 powered with enhanced prompting
        - Schema-aware context generation
        - Automatic SQL validation and fixing
        - Best practices enforcement
        
        **Quality Evaluation Metrics:**
        - **Execution Accuracy (EX)**: Result set comparison
        - **Logical Form Accuracy (LF)**: SQL structure analysis
        - **Partial Component Matching (PCM)**: Component-wise evaluation
        - **GPT-4 Semantic Assessment**: AI-powered quality scoring
        
        **System Requirements:**
        - DuckDB knowledge graph database
        - Optional: Client manager for GPT access
        - Python 3.8+ with required dependencies
        
        Built with ❤️ using Advanced AI techniques for intelligent SQL generation.
        """)
    
    return demo
Fix 2: Update the Performance Table Headers
In the step1_find_tables function, make sure the DataFrame matches the headers:
python# Create performance summary with similarity info
similarity_info = ""
if table_details:
    scores = [details.get('similarity_score', 0) for details in table_details.values() 
             if 'similarity_score' in details]
    if scores:
        avg_score = sum(scores) / len(scores)
        similarity_info = f" (Avg: {avg_score:.3f})"

performance_data = [{
    'Step': 'Table Discovery',
    'Method': 'Enhanced Embedding + Attention',
    'Tables Found': len(tables),
    'Status': f'✅ Success{similarity_info}' if tables else '⚠️ No tables found',
    'Threshold': f'{similarity_threshold:.2f}',
    'Timestamp': datetime.now().strftime('%H:%M:%S')
}]
Fix 3: Update Database Info Display
Update the get_database_info_display() function to show the enhanced features:
pythondef get_database_info_display():
    """Get database information for display."""
    try:
        if not app_state['backend_ready']:
            return "❌ Backend not initialized"
        
        backend = get_backend_instance()
        db_info = backend.get_database_info()
        
        if 'error' in db_info:
            return f"❌ Error getting database info: {db_info['error']}"
        
        info_text = f"""
## 📊 Database Information

- **Tables:** {db_info.get('table_count', 'N/A')}
- **Columns:** {db_info.get('column_count', 'N/A')}
- **Relationships:** {db_info.get('relationship_count', 'N/A')}
- **File Size:** {db_info.get('file_size_mb', 'N/A')} MB
- **Client Manager:** {'✅ Available' if db_info.get('client_manager_available') else '❌ Not Available'}

### 🚀 Enhanced Features:
- **✅ Embedding-based table retrieval**
- **✅ Multi-hop graph reasoning**
- **✅ Attention mechanisms**
- **✅ Column-level semantic matching**
- **✅ Dynamic similarity thresholds**

### 📋 Table Types:
"""
        
        table_types = db_info.get('table_types', {})
        for table_type, count in table_types.items():
            info_text += f"- **{table_type.title()}:** {count} tables\n"
        
        return info_text
        
    except Exception as e:
        return f"❌ Error: {str(e)}"
Summary of Fixed Issues: