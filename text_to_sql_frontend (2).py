#!/usr/bin/env python3
"""
Text-to-SQL Frontend Gradio UI

Frontend web interface for the text-to-SQL system using Gradio.
Provides a complete workflow: Query -> Tables -> SQL -> Evaluation
"""

import gradio as gr
import pandas as pd
import asyncio
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import logging

# Import the backend (ensure this is in the same directory or path)
from text_to_sql_backend import (
    get_backend_instance, 
    initialize_backend, 
    check_system_requirements,
    format_table_details_for_display,
    format_evaluation_results_for_display,
    CLIENT_MANAGER_AVAILABLE
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global state for the application
app_state = {
    'current_query': '',
    'found_tables': [],
    'selected_tables': [],
    'table_details': {},
    'generated_sql': '',
    'reasoning': '',
    'evaluation_results': {},
    'backend_ready': False
}

# === Helper Functions ===

def initialize_system():
    """Initialize the backend system and return status."""
    try:
        # Check system requirements
        system_ok, system_message = check_system_requirements()
        if not system_ok:
            return False, system_message
        
        # Initialize backend
        backend_ok, backend_message = initialize_backend()
        if not backend_ok:
            return False, backend_message
        
        app_state['backend_ready'] = True
        return True, "✅ System initialized successfully"
        
    except Exception as e:
        return False, f"❌ System initialization failed: {str(e)}"

def get_database_info_display():
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

### 📋 Table Types:
"""
        
        table_types = db_info.get('table_types', {})
        for table_type, count in table_types.items():
            info_text += f"- **{table_type.title()}:** {count} tables\n"
        
        return info_text
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

def run_async_function(async_func, *args, **kwargs):
    """Helper to run async functions in Gradio."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(async_func(*args, **kwargs))

# === Main Workflow Functions ===

def step1_find_tables(query: str, max_tables: int = 10):
    """Step 1: Find relevant tables using Advanced Graph Traversal."""
    if not query.strip():
        return "Please enter a query.", "", [], pd.DataFrame(), "Enter a query to find tables"
    
    if not app_state['backend_ready']:
        return "❌ Backend not initialized", "", [], pd.DataFrame(), "System not ready"
    
    try:
        # Update app state
        app_state['current_query'] = query
        
        # Find tables using backend
        backend = get_backend_instance()
        tables, table_details, status_message = run_async_function(
            backend.find_relevant_tables, query, max_tables
        )
        
        # Update app state
        app_state['found_tables'] = tables
        app_state['selected_tables'] = tables.copy()  # Select all by default
        app_state['table_details'] = table_details
        
        # Format table details for display
        table_details_text = format_table_details_for_display(table_details)
        
        # Create performance summary
        performance_data = [{
            'Step': 'Table Discovery',
            'Method': 'Advanced Graph Traversal',
            'Tables Found': len(tables),
            'Status': '✅ Success' if tables else '⚠️ No tables found',
            'Timestamp': datetime.now().strftime('%H:%M:%S')
        }]
        
        performance_df = pd.DataFrame(performance_data)
        
        # Create selection options for next step
        table_choices = tables if tables else []
        
        return (
            status_message,
            table_details_text,
            table_choices,  # For the checkbox group
            performance_df,
            f"Found {len(tables)} tables. Select tables and click 'Generate SQL' to proceed."
        )
        
    except Exception as e:
        error_msg = f"❌ Error in table discovery: {str(e)}"
        logger.error(error_msg)
        return error_msg, "", [], pd.DataFrame(), "Table discovery failed"

def step2_generate_sql(selected_tables: List[str]):
    """Step 2: Generate SQL from selected tables."""
    if not selected_tables:
        return "Please select at least one table.", "", "No tables selected"
    
    if not app_state['backend_ready']:
        return "❌ Backend not initialized", "", "System not ready"
    
    if not app_state['current_query']:
        return "❌ No query found. Please run table discovery first.", "", "No query available"
    
    try:
        # Update selected tables in app state
        app_state['selected_tables'] = selected_tables
        
        # Generate SQL using backend
        backend = get_backend_instance()
        sql_code, reasoning, status_message = run_async_function(
            backend.generate_sql_from_tables,
            app_state['current_query'],
            selected_tables,
            app_state['table_details']
        )
        
        # Update app state
        app_state['generated_sql'] = sql_code
        app_state['reasoning'] = reasoning
        
        # Format results
        results_text = f"""
## 🎯 Generated SQL Query

**Query:** {app_state['current_query']}

**Selected Tables:** {', '.join(selected_tables)}

**Generated SQL:**
```sql
{sql_code}
```

## 🧠 Reasoning & Analysis

{reasoning}

**Status:** {status_message}
**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        next_step_message = "SQL generated successfully. You can now evaluate it against ground truth or make modifications."
        
        return results_text, sql_code, next_step_message
        
    except Exception as e:
        error_msg = f"❌ Error in SQL generation: {str(e)}"
        logger.error(error_msg)
        return error_msg, "", "SQL generation failed"

def step3_evaluate_sql(ground_truth_sql: str):
    """Step 3: Evaluate generated SQL against ground truth."""
    if not ground_truth_sql.strip():
        return "Please enter the ground truth SQL query.", "No ground truth provided"
    
    if not app_state['generated_sql']:
        return "❌ No generated SQL found. Please generate SQL first.", "No generated SQL available"
    
    if not app_state['backend_ready']:
        return "❌ Backend not initialized", "System not ready"
    
    try:
        # Evaluate SQL using backend
        backend = get_backend_instance()
        evaluation_results, status_message = run_async_function(
            backend.evaluate_sql_quality,
            app_state['generated_sql'],
            ground_truth_sql,
            app_state['current_query']
        )
        
        # Update app state
        app_state['evaluation_results'] = evaluation_results
        
        # Format evaluation results for display
        evaluation_text = format_evaluation_results_for_display(evaluation_results)
        
        return evaluation_text, status_message
        
    except Exception as e:
        error_msg = f"❌ Error in SQL evaluation: {str(e)}"
        logger.error(error_msg)
        return error_msg, "SQL evaluation failed"

def export_results():
    """Export current results to a JSON file."""
    try:
        if not any([app_state['generated_sql'], app_state['found_tables']]):
            return "❌ No results to export"
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"text_to_sql_results_{timestamp}.json"
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'query': app_state['current_query'],
            'found_tables': app_state['found_tables'],
            'selected_tables': app_state['selected_tables'],
            'generated_sql': app_state['generated_sql'],
            'reasoning': app_state['reasoning'],
            'evaluation_results': app_state['evaluation_results'],
            'system_info': {
                'client_manager_available': CLIENT_MANAGER_AVAILABLE,
                'backend_ready': app_state['backend_ready']
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return f"✅ Results exported to {filename}"
        
    except Exception as e:
        return f"❌ Export failed: {str(e)}"

def clear_all_data():
    """Clear all application state."""
    global app_state
    app_state.update({
        'current_query': '',
        'found_tables': [],
        'selected_tables': [],
        'table_details': {},
        'generated_sql': '',
        'reasoning': '',
        'evaluation_results': {}
    })
    
    return (
        "",  # query input
        "",  # status display
        "",  # table details
        [],  # table selection
        pd.DataFrame(),  # performance table
        "",  # step guidance
        "",  # sql results
        "",  # sql code display
        "",  # ground truth input
        "",  # evaluation results
        "🗑️ All data cleared. Start with a new query."
    )

# === Example Queries ===

EXAMPLE_QUERIES = [
    "give me distinct source systems for cash ETD trades for yesterday",
    "show me EXECUTING_TRADER_SOEID EXECUTION_VENUE PRODUCT_SK where TRADE_PRICE_CURRENCY and ISSUE_CURRENCY are not the same",
    "Show me the counterparty for trade ID 18871106",
    "get me the CUSIP that was traded highest last week",
    "show me all trades by government entities",
    "find all trades with notional amount greater than 1 million",
    "list top 10 traders by trade volume this month",
    "show me all failed trades from yesterday"
]

def set_example_query(example_index: int):
    """Set an example query."""
    if 0 <= example_index < len(EXAMPLE_QUERIES):
        return EXAMPLE_QUERIES[example_index]
    return ""

# === Create Gradio Interface ===

def create_text_to_sql_interface():
    """Create the complete text-to-SQL Gradio interface."""
    
    # Initialize system
    system_ready, init_message = initialize_system()
    
    with gr.Blocks(
        title="Text-to-SQL System with Advanced Graph Traversal",
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
        # 🚀 Text-to-SQL System with Advanced Graph Traversal
        
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
                        placeholder="Example: give me distinct source systems for cash ETD trades for yesterday",
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
                        find_tables_btn = gr.Button("🔍 Find Tables", variant="primary")
                
                with gr.Column(scale=2):
                    # Example queries
                    gr.Markdown("**Quick Examples:**")
                    example_buttons = []
                    for i, example in enumerate(EXAMPLE_QUERIES[:4]):
                        btn = gr.Button(f"📝 {example[:40]}...", size="sm")
                        example_buttons.append(btn)
            
            # Results for Step 1
            with gr.Row():
                with gr.Column():
                    step1_status = gr.Markdown("")
                    
                    table_details_display = gr.Markdown("", label="📋 Found Tables Details")
                    
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
                    headers=["Step", "Method", "Tables Found", "Status", "Timestamp"],
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
        
        # Example query buttons
        for i, btn in enumerate(example_buttons):
            btn.click(
                fn=lambda idx=i: set_example_query(idx),
                outputs=query_input
            )
        
        # Step 1: Find tables
        find_tables_btn.click(
            fn=step1_find_tables,
            inputs=[query_input, max_tables_slider],
            outputs=[
                step1_status,
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

# === Main Application ===

def main():
    """Main function to launch the text-to-SQL web application."""
    print("🚀 Starting Text-to-SQL Web Interface...")
    
    # Check for required files
    if not os.path.exists("knowledge_graph.duckdb"):
        print("❌ DuckDB file 'knowledge_graph.duckdb' not found!")
        print("💡 Please ensure the knowledge graph database is available.")
        return
    
    # Create and launch interface
    demo = create_text_to_sql_interface()
    
    print("✅ Web interface created successfully!")
    print("🌐 Launching on http://localhost:7860")
    print("📊 Available features:")
    print("   - Advanced Graph Traversal table discovery")
    print("   - GPT-4 powered SQL generation")
    print("   - Comprehensive SQL quality evaluation")
    print("   - Interactive three-step workflow")
    print("   - Export and import capabilities")
    
    # Launch with custom settings
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True for public sharing
        show_error=True,
        favicon_path=None,  # Add your favicon if available
        ssl_verify=False
    )

if __name__ == "__main__":
    main()