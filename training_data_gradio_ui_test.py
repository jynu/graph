# gradio_ui_test.py - Gradio-based UI and test script
import gradio as gr
import threading
import sys
import os
from datetime import datetime
import json
import tempfile
import shutil
from typing import List, Tuple, Optional
import time

# Import our main processor
try:
    from training_data_main import TrainingDataProcessor, ConfigManager, DatabaseConfig, SystemConfig
except ImportError as e:
    print(f"Error importing training data processor: {e}")
    print("Make sure training_data_main.py and getFrequentColumn.py are in the same directory")
    sys.exit(1)


class TrainingDataGradioApp:
    """Gradio-based web interface for the Training Data Selection System"""
    
    def __init__(self):
        self.config_files = []
        self.processor = None
        self.processing_thread = None
        self.is_processing = False
        self.log_messages = []
        self.stats = {
            'tables_processed': 0,
            'columns_analyzed': 0,
            'start_time': None
        }
        
        # Load default configs if available
        self.load_default_configs()
    
    def load_default_configs(self):
        """Load default configuration files if they exist"""
        default_configs = [
            r"C:/Users/AK06306/Downloads/rsocket_workspace/olympus-dc-server/data_preprocessing/Tables_Selection_Middleoffice.conf",
            r"C:/Users/AK06306/Downloads/rsocket_workspace/olympus-dc-server/data_preprocessing/Tables_Selection.conf",
            "Tables_Selection.conf",
            "config.conf",
            "sample_config.conf"
        ]
        
        for config_file in default_configs:
            if os.path.exists(config_file):
                self.config_files.append(config_file)
                self.log(f"Loaded default config: {config_file}")
    
    def log(self, message: str):
        """Add a message to the log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        self.log_messages.append(log_message)
        
        # Keep only last 1000 messages to prevent memory issues
        if len(self.log_messages) > 1000:
            self.log_messages = self.log_messages[-1000:]
    
    def get_log_text(self) -> str:
        """Get formatted log text"""
        return "\n".join(self.log_messages[-50:])  # Show last 50 messages
    
    def add_config_file(self, file_obj) -> Tuple[str, str, str]:
        """Add a configuration file from upload"""
        if file_obj is None:
            return "No file selected", self.get_config_list(), self.get_log_text()
        
        try:
            # Save uploaded file to temporary location
            temp_dir = tempfile.gettempdir()
            config_filename = os.path.join(temp_dir, os.path.basename(file_obj.name))
            shutil.copy2(file_obj.name, config_filename)
            
            if config_filename not in self.config_files:
                self.config_files.append(config_filename)
                self.log(f"Added config file: {os.path.basename(config_filename)}")
                return f"Added: {os.path.basename(config_filename)}", self.get_config_list(), self.get_log_text()
            else:
                return "File already added", self.get_config_list(), self.get_log_text()
                
        except Exception as e:
            error_msg = f"Error adding config file: {str(e)}"
            self.log(error_msg)
            return error_msg, self.get_config_list(), self.get_log_text()
    
    def remove_config_file(self, selected_config: str) -> Tuple[str, str, str]:
        """Remove a configuration file"""
        if not selected_config or selected_config == "No config files":
            return "No file selected", self.get_config_list(), self.get_log_text()
        
        # Find the full path
        config_to_remove = None
        for config_file in self.config_files:
            if os.path.basename(config_file) == selected_config:
                config_to_remove = config_file
                break
        
        if config_to_remove:
            self.config_files.remove(config_to_remove)
            self.log(f"Removed config file: {selected_config}")
            return f"Removed: {selected_config}", self.get_config_list(), self.get_log_text()
        else:
            return "File not found", self.get_config_list(), self.get_log_text()
    
    def clear_config_files(self) -> Tuple[str, str, str]:
        """Clear all configuration files"""
        count = len(self.config_files)
        self.config_files.clear()
        self.log(f"Cleared {count} config files")
        return f"Cleared {count} files", self.get_config_list(), self.get_log_text()
    
    def get_config_list(self) -> str:
        """Get list of configuration files as string"""
        if not self.config_files:
            return "No config files"
        return "\n".join([os.path.basename(f) for f in self.config_files])
    
    def test_configuration(self, section: str) -> Tuple[str, str]:
        """Test configuration files without running full processing"""
        if not self.config_files:
            return "❌ Error: No configuration files added", self.get_log_text()
        
        try:
            self.log("Testing configuration files...")
            config_manager = ConfigManager(self.config_files)
            
            # Test getting section data
            section_data = config_manager.get_section_config(section)
            
            # Test getting tables list
            tables = config_manager.get_tables_list(section)
            
            # Test getting system config
            system_config = config_manager.get_system_config(section)
            
            # Test database config
            db_config = config_manager.get_database_config(section)
            
            self.log(f"✓ Configuration test passed for section: {section}")
            self.log(f"✓ Found {len(tables)} tables: {', '.join(tables)}")
            self.log(f"✓ Output directory: {system_config.output_directory}")
            self.log(f"✓ Base directory: {system_config.base_directory}")
            self.log(f"✓ Database host configured: {bool(db_config.host)}")
            
            return "✅ Configuration test passed!", self.get_log_text()
            
        except Exception as e:
            error_msg = f"❌ Configuration test failed: {str(e)}"
            self.log(error_msg)
            return error_msg, self.get_log_text()
    
    def start_processing(self, section: str) -> Tuple[str, str, str, str]:
        """Start the processing in a separate thread"""
        if not self.config_files:
            return "❌ Error: No configuration files added", "Ready", "0", self.get_log_text()
        
        if self.is_processing:
            return "⚠️ Processing already in progress", "Processing...", str(self.stats['tables_processed']), self.get_log_text()
        
        self.is_processing = True
        self.stats['start_time'] = datetime.now()
        self.stats['tables_processed'] = 0
        self.stats['columns_analyzed'] = 0
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self.run_processing, 
            args=(section,), 
            daemon=True
        )
        self.processing_thread.start()
        
        return "🚀 Processing started...", "Processing...", "0", self.get_log_text()
    
    def run_processing(self, section: str):
        """Run the actual processing"""
        try:
            self.log("=" * 50)
            self.log(f"Starting processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.log(f"Config files: {[os.path.basename(f) for f in self.config_files]}")
            self.log(f"Section: {section}")
            self.log("=" * 50)
            
            # Initialize processor
            self.processor = TrainingDataProcessor(self.config_files)
            
            # Run processing
            success = self.processor.process(section)
            
            if success:
                self.log("🎉 Processing completed successfully!")
                self.log(f"Total processing time: {datetime.now() - self.stats['start_time']}")
            else:
                self.log("❌ Processing failed!")
                
        except Exception as e:
            error_msg = f"💥 Error during processing: {str(e)}"
            self.log(error_msg)
        finally:
            self.is_processing = False
    
    def get_processing_status(self) -> Tuple[str, str]:
        """Get current processing status"""
        if self.is_processing:
            elapsed = datetime.now() - self.stats['start_time'] if self.stats['start_time'] else datetime.now()
            return f"Processing... (Elapsed: {elapsed})", str(self.stats['tables_processed'])
        else:
            return "Ready", str(self.stats['tables_processed'])
    
    def view_output_directory(self) -> str:
        """Get output directory information"""
        try:
            if self.processor and hasattr(self.processor, 'system_config'):
                output_dir = self.processor.system_config.output_directory
                if os.path.exists(output_dir):
                    files = os.listdir(output_dir)
                    file_list = "\n".join([f"📄 {f}" for f in files]) if files else "No files found"
                    return f"📁 Output Directory: {output_dir}\n\nFiles:\n{file_list}"
                else:
                    return f"📁 Output Directory: {output_dir}\n❌ Directory does not exist yet"
            else:
                return "No processing has been run yet"
        except Exception as e:
            return f"Error accessing output directory: {e}"
    
    def create_sample_config(self) -> Tuple[str, str, str]:
        """Create a sample configuration file"""
        sample_config_content = """[transcation]
TABLES = sample_table1,sample_table2
BASEDIRECTORY = ./sample_output
OUTPUT_DIRECTORY = ./sample_output
START_DATE = 20250101
USERID = your_username
PASSWORD = your_password

sample_table1_Columns = id,name,type,status,date_created
sample_table1_WhereCol_daycriteria = date_created
sample_table1_NoOfDays = 30

sample_table2_Columns = [generated]
sample_table2_WhereCol_daycriteria = last_updated
sample_table2_NoOfDays = 90

[risk]
TABLES = risk_positions,risk_metrics
BASEDIRECTORY = ./sample_output
OUTPUT_DIRECTORY = ./sample_output
START_DATE = 20250101
USERID = your_username
PASSWORD = your_password

risk_positions_Columns = portfolio_id,instrument_id,quantity,market_value
risk_positions_WhereCol_daycriteria = as_of_date
risk_positions_NoOfDays = 60

risk_metrics_Columns = risk_type,var_amount,stress_result
risk_metrics_WhereCol_daycriteria = calculation_date
risk_metrics_NoOfDays = 30
"""
        
        try:
            config_filename = f"sample_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.conf"
            with open(config_filename, 'w') as f:
                f.write(sample_config_content)
            
            # Add to config files list
            self.config_files.append(os.path.abspath(config_filename))
            self.log(f"Created sample config: {config_filename}")
            
            return f"✅ Created: {config_filename}", self.get_config_list(), self.get_log_text()
            
        except Exception as e:
            error_msg = f"❌ Error creating sample config: {str(e)}"
            self.log(error_msg)
            return error_msg, self.get_config_list(), self.get_log_text()
    
    def run_unit_tests(self) -> str:
        """Run unit tests and return results"""
        self.log("Starting unit tests...")
        test_results = []
        
        # Create temporary test configs
        test_config_content = """[transcation]
TABLES = test_table1,test_table2
BASEDIRECTORY = ./test_output
OUTPUT_DIRECTORY = ./test_output
START_DATE = 20250101
USERID = test_user
PASSWORD = test_pass

test_table1_Columns = col1,col2,col3
test_table1_WhereCol_daycriteria = date_col
test_table1_NoOfDays = 30
"""
        
        test_config_file = "temp_test_config.conf"
        
        try:
            with open(test_config_file, 'w') as f:
                f.write(test_config_content)
            
            # Test 1: Config Manager
            try:
                config_manager = ConfigManager([test_config_file])
                self.log("✓ ConfigManager initialization successful")
                test_results.append(True)
                
                section_data = config_manager.get_section_config("transcation")
                assert "TABLES" in section_data
                self.log("✓ Section configuration loading successful")
                test_results.append(True)
                
            except Exception as e:
                self.log(f"✗ ConfigManager test failed: {e}")
                test_results.append(False)
            
            # Test 2: System Config
            try:
                system_config = config_manager.get_system_config()
                assert system_config.base_directory == "./test_output"
                self.log("✓ SystemConfig loading successful")
                test_results.append(True)
                
            except Exception as e:
                self.log(f"✗ SystemConfig test failed: {e}")
                test_results.append(False)
            
            # Test 3: Tables List
            try:
                tables = config_manager.get_tables_list()
                assert len(tables) > 0
                self.log(f"✓ Tables list loading successful: {tables}")
                test_results.append(True)
                
            except Exception as e:
                self.log(f"✗ Tables list test failed: {e}")
                test_results.append(False)
            
            # Test 4: Database Config
            try:
                db_config = config_manager.get_database_config()
                assert db_config.user_id == "test_user"
                self.log("✓ Database configuration successful")
                test_results.append(True)
                
            except Exception as e:
                self.log(f"✗ Database config test failed: {e}")
                test_results.append(False)
            
            passed = sum(test_results)
            total = len(test_results)
            self.log(f"Unit test results: {passed}/{total} tests passed")
            
            if passed == total:
                result = "🎉 All unit tests passed!"
            else:
                result = f"⚠️ {total - passed} test(s) failed - check logs"
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Error running tests: {str(e)}"
            self.log(error_msg)
            return error_msg
        finally:
            # Cleanup
            if os.path.exists(test_config_file):
                os.remove(test_config_file)
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        with gr.Blocks(title="Training Data Selection System", theme=gr.themes.Soft()) as interface:
            
            # Header
            gr.Markdown("# 🎯 Training Data Selection System")
            gr.Markdown("### Automated database table analysis and training data preparation")
            
            with gr.Tabs():
                
                # Main Processing Tab
                with gr.TabItem("🚀 Processing", id="main"):
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            
                            # Configuration Section
                            gr.Markdown("## 📁 Configuration Files")
                            
                            with gr.Row():
                                config_file_upload = gr.File(
                                    label="Upload Config File (.conf)",
                                    file_types=[".conf"],
                                    type="filepath"
                                )
                                with gr.Column():
                                    add_config_btn = gr.Button("Add Config", variant="primary")
                                    create_sample_btn = gr.Button("Create Sample Config")
                            
                            config_list_display = gr.Textbox(
                                label="Current Config Files",
                                value=self.get_config_list(),
                                interactive=False,
                                lines=4
                            )
                            
                            with gr.Row():
                                config_dropdown = gr.Dropdown(
                                    label="Select Config to Remove",
                                    choices=[os.path.basename(f) for f in self.config_files] if self.config_files else ["No config files"],
                                    value=None
                                )
                                remove_config_btn = gr.Button("Remove Selected")
                                clear_config_btn = gr.Button("Clear All", variant="stop")
                            
                            # Processing Options
                            gr.Markdown("## ⚙️ Processing Options")
                            
                            section_dropdown = gr.Dropdown(
                                label="Processing Section",
                                choices=["transcation", "risk", "market", "ref"],
                                value="transcation"
                            )
                            
                            with gr.Row():
                                test_config_btn = gr.Button("🧪 Test Config", variant="secondary")
                                start_processing_btn = gr.Button("🚀 Start Processing", variant="primary")
                            
                            # Status Section
                            gr.Markdown("## 📊 Status")
                            status_text = gr.Textbox(
                                label="Status",
                                value="Ready - Add configuration files to start",
                                interactive=False
                            )
                            
                            with gr.Row():
                                tables_processed = gr.Number(
                                    label="Tables Processed",
                                    value=0,
                                    interactive=False
                                )
                                processing_time = gr.Textbox(
                                    label="Processing Time",
                                    value="00:00:00",
                                    interactive=False
                                )
                        
                        with gr.Column(scale=3):
                            
                            # Log Section
                            gr.Markdown("## 📝 Processing Log")
                            
                            log_display = gr.Textbox(
                                label="Real-time Log",
                                value=self.get_log_text(),
                                interactive=False,
                                lines=20,
                                max_lines=30,
                                autoscroll=True
                            )
                            
                            with gr.Row():
                                refresh_log_btn = gr.Button("🔄 Refresh Log")
                                clear_log_btn = gr.Button("🗑️ Clear Log")
                            
                            # Output Section
                            gr.Markdown("## 📤 Output")
                            
                            output_info = gr.Textbox(
                                label="Output Directory Info",
                                value="No processing completed yet",
                                interactive=False,
                                lines=8
                            )
                            
                            view_output_btn = gr.Button("📁 View Output Info")
                
                # Testing Tab
                with gr.TabItem("🧪 Testing", id="testing"):
                    
                    gr.Markdown("## 🔧 System Testing")
                    gr.Markdown("Test the system components without running full processing")
                    
                    with gr.Row():
                        with gr.Column():
                            
                            run_tests_btn = gr.Button("🧪 Run Unit Tests", variant="primary")
                            
                            test_results = gr.Textbox(
                                label="Test Results",
                                value="Click 'Run Unit Tests' to start testing",
                                interactive=False,
                                lines=5
                            )
                            
                            gr.Markdown("### 📋 Test Coverage")
                            gr.Markdown("""
                            - Configuration file loading and validation
                            - System configuration parsing
                            - Database configuration setup
                            - Tables list extraction
                            - Component initialization
                            """)
                        
                        with gr.Column():
                            
                            gr.Markdown("### 🛠️ Quick Actions")
                            
                            create_sample_config_btn = gr.Button("📄 Create Sample Config")
                            validate_current_config_btn = gr.Button("✅ Validate Current Config")
                            
                            quick_test_results = gr.Textbox(
                                label="Quick Test Results",
                                value="Ready for testing",
                                interactive=False,
                                lines=8
                            )
                
                # Help Tab
                with gr.TabItem("❓ Help", id="help"):
                    
                    gr.Markdown("""
                    ## 📖 How to Use
                    
                    ### 1. Configuration Setup
                    - Upload or create configuration files (.conf format)
                    - Each config file should contain database connection details and table specifications
                    - Use "Create Sample Config" to generate a template
                    
                    ### 2. Processing Steps
                    1. Add configuration files using the upload button
                    2. Select the appropriate processing section (transcation, risk, market, ref)
                    3. Test your configuration using "Test Config"
                    4. Start processing with "Start Processing"
                    5. Monitor progress in the log panel
                    
                    ### 3. Configuration File Format
                    ```ini
                    [transcation]
                    TABLES = table1,table2
                    BASEDIRECTORY = ./output
                    OUTPUT_DIRECTORY = ./final_output
                    START_DATE = 20250101
                    USERID = your_username
                    PASSWORD = your_password
                    
                    table1_Columns = col1,col2,col3
                    table1_WhereCol_daycriteria = date_column
                    table1_NoOfDays = 30
                    ```
                    
                    ### 4. System Requirements
                    - Python 3.8+
                    - Java 17+ (for database client)
                    - Gradio library
                    - Network access to database
                    
                    ### 5. Troubleshooting
                    - Check the log panel for detailed error messages
                    - Use "Test Config" to validate setup before processing
                    - Ensure database credentials and paths are correct
                    - Verify Java client is accessible
                    
                    ### 6. Output
                    The system generates JSON files with training data specifications including:
                    - Table metadata and column information
                    - Distinct value counts and enumeration data
                    - Business rule decisions for each column
                    """)
            
            # Event Handlers
            add_config_btn.click(
                fn=self.add_config_file,
                inputs=[config_file_upload],
                outputs=[status_text, config_list_display, log_display]
            )
            
            remove_config_btn.click(
                fn=self.remove_config_file,
                inputs=[config_dropdown],
                outputs=[status_text, config_list_display, log_display]
            )
            
            clear_config_btn.click(
                fn=self.clear_config_files,
                outputs=[status_text, config_list_display, log_display]
            )
            
            create_sample_btn.click(
                fn=self.create_sample_config,
                outputs=[status_text, config_list_display, log_display]
            )
            
            test_config_btn.click(
                fn=self.test_configuration,
                inputs=[section_dropdown],
                outputs=[status_text, log_display]
            )
            
            start_processing_btn.click(
                fn=self.start_processing,
                inputs=[section_dropdown],
                outputs=[status_text, processing_time, tables_processed, log_display]
            )
            
            refresh_log_btn.click(
                fn=lambda: self.get_log_text(),
                outputs=[log_display]
            )
            
            clear_log_btn.click(
                fn=self.clear_log,
                outputs=[log_display, status_text]
            )
            
            view_output_btn.click(
                fn=self.view_output_directory,
                outputs=[output_info]
            )
            
            run_tests_btn.click(
                fn=self.run_unit_tests,
                outputs=[test_results]
            )
            
            create_sample_config_btn.click(
                fn=self.create_sample_config,
                outputs=[quick_test_results, config_list_display, log_display]
            )
            
            validate_current_config_btn.click(
                fn=lambda section: self.test_configuration(section)[0],
                inputs=[section_dropdown],
                outputs=[quick_test_results]
            )
            
            # Auto-refresh functionality for processing status
            def auto_refresh():
                if self.is_processing:
                    status, tables = self.get_processing_status()
                    elapsed = str(datetime.now() - self.stats['start_time']).split('.')[0] if self.stats['start_time'] else "00:00:00"
                    return status, elapsed, tables, self.get_log_text()
                return gr.update(), gr.update(), gr.update(), gr.update()
            
            # Set up auto-refresh every 2 seconds when processing
            interface.load(
                fn=auto_refresh,
                outputs=[status_text, processing_time, tables_processed, log_display],
                every=2
            )
        
        return interface
    
    def clear_log(self) -> Tuple[str, str]:
        """Clear the log messages"""
        self.log_messages.clear()
        self.log("Log cleared")
        return "", "Log cleared"


def create_app():
    """Create and return the Gradio app"""
    app = TrainingDataGradioApp()
    interface = app.create_interface()
    return interface


def run_tests_cli():
    """Run tests from command line"""
    app = TrainingDataGradioApp()
    result = app.run_unit_tests()
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(result)
    print("\nLog messages:")
    print(app.get_log_text())


def create_sample_config_cli():
    """Create sample config from command line"""
    app = TrainingDataGradioApp()
    result, config_list, log_text = app.create_sample_config()
    print(result)
    print(f"Current configs: {config_list}")


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test":
            run_tests_cli()
        elif command == "sample":
            create_sample_config_cli()
        elif command == "help":
            print("Training Data Selection System - Gradio Interface")
            print("Usage:")
            print("  python gradio_ui_test.py          - Run web interface")
            print("  python gradio_ui_test.py test     - Run test suite")
            print("  python gradio_ui_test.py sample   - Create sample config file")
            print("  python gradio_ui_test.py help     - Show this help")
        else:
            print(f"Unknown command: {command}")
            print("Use 'help' to see available commands")
    else:
        # Run the Gradio interface
        interface = create_app()
        interface.launch(
            server_name="0.0.0.0",  # Allow access from other machines
            server_port=7860,       # Default Gradio port
            share=False,            # Set to True to create public link
            debug=True,             # Enable debug mode
            show_error=True         # Show detailed error messages
        )


if __name__ == "__main__":
    main()