# simple_gradio_ui.py - Simplified Gradio interface compatible with all versions
import gradio as gr
import threading
import sys
import os
from datetime import datetime
import json
import tempfile
import shutil
from typing import List, Tuple, Optional

# Import our main processor
try:
    from training_data_main import TrainingDataProcessor, ConfigManager, DatabaseConfig, SystemConfig
except ImportError as e:
    print(f"Error importing training data processor: {e}")
    print("Make sure training_data_main.py and getFrequentColumn.py are in the same directory")
    sys.exit(1)


class SimpleTrainingDataApp:
    """Simplified Gradio interface for the Training Data Selection System"""
    
    def __init__(self):
        self.config_files = []
        self.processor = None
        self.is_processing = False
        self.log_messages = []
        
        # Load default configs if available
        self.load_default_configs()
    
    def load_default_configs(self):
        """Load default configuration files if they exist"""
        default_configs = [
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
        
        # Keep only last 100 messages
        if len(self.log_messages) > 100:
            self.log_messages = self.log_messages[-100:]
    
    def get_log_text(self) -> str:
        """Get formatted log text"""
        return "\n".join(self.log_messages)
    
    def get_config_list(self) -> str:
        """Get list of configuration files"""
        if not self.config_files:
            return "No configuration files loaded"
        return "\n".join([f"📄 {os.path.basename(f)}" for f in self.config_files])
    
    def add_config_file(self, file_obj):
        """Add a configuration file"""
        if file_obj is None:
            return "❌ No file selected", self.get_config_list(), self.get_log_text()
        
        try:
            # Copy uploaded file to current directory
            config_filename = os.path.basename(file_obj.name)
            shutil.copy2(file_obj.name, config_filename)
            
            if config_filename not in [os.path.basename(f) for f in self.config_files]:
                self.config_files.append(os.path.abspath(config_filename))
                self.log(f"Added config file: {config_filename}")
                return f"✅ Added: {config_filename}", self.get_config_list(), self.get_log_text()
            else:
                return "⚠️ File already exists", self.get_config_list(), self.get_log_text()
                
        except Exception as e:
            error_msg = f"❌ Error adding config file: {str(e)}"
            self.log(error_msg)
            return error_msg, self.get_config_list(), self.get_log_text()
    
    def create_sample_config(self):
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
"""
        
        try:
            config_filename = f"sample_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.conf"
            with open(config_filename, 'w') as f:
                f.write(sample_config_content)
            
            self.config_files.append(os.path.abspath(config_filename))
            self.log(f"Created sample config: {config_filename}")
            
            return f"✅ Created: {config_filename}", self.get_config_list(), self.get_log_text()
            
        except Exception as e:
            error_msg = f"❌ Error creating sample config: {str(e)}"
            self.log(error_msg)
            return error_msg, self.get_config_list(), self.get_log_text()
    
    def test_configuration(self, section: str):
        """Test configuration files"""
        if not self.config_files:
            return "❌ No configuration files loaded", self.get_log_text()
        
        try:
            self.log("Testing configuration...")
            config_manager = ConfigManager(self.config_files)
            
            # Test configuration
            section_data = config_manager.get_section_config(section)
            tables = config_manager.get_tables_list(section)
            system_config = config_manager.get_system_config(section)
            
            self.log(f"✅ Configuration test passed for section: {section}")
            self.log(f"📊 Found {len(tables)} tables: {', '.join(tables)}")
            self.log(f"📁 Output directory: {system_config.output_directory}")
            
            return "✅ Configuration test passed!", self.get_log_text()
            
        except Exception as e:
            error_msg = f"❌ Configuration test failed: {str(e)}"
            self.log(error_msg)
            return error_msg, self.get_log_text()
    
    def start_processing(self, section: str):
        """Start processing"""
        if not self.config_files:
            return "❌ No configuration files loaded", self.get_log_text()
        
        if self.is_processing:
            return "⚠️ Processing already in progress", self.get_log_text()
        
        try:
            self.log("🚀 Starting processing...")
            self.log(f"Config files: {[os.path.basename(f) for f in self.config_files]}")
            self.log(f"Section: {section}")
            
            self.processor = TrainingDataProcessor(self.config_files)
            self.is_processing = True
            
            # Run processing in thread
            def run_processing():
                try:
                    success = self.processor.process(section)
                    if success:
                        self.log("🎉 Processing completed successfully!")
                    else:
                        self.log("❌ Processing failed!")
                except Exception as e:
                    self.log(f"💥 Error during processing: {str(e)}")
                finally:
                    self.is_processing = False
            
            thread = threading.Thread(target=run_processing, daemon=True)
            thread.start()
            
            return "🚀 Processing started...", self.get_log_text()
            
        except Exception as e:
            self.is_processing = False
            error_msg = f"❌ Error starting processing: {str(e)}"
            self.log(error_msg)
            return error_msg, self.get_log_text()
    
    def get_processing_status(self):
        """Get current processing status"""
        if self.is_processing:
            return "🔄 Processing in progress...", self.get_log_text()
        else:
            return "✅ Ready", self.get_log_text()
    
    def view_output_info(self):
        """Get output directory information"""
        try:
            if self.processor and hasattr(self.processor, 'system_config'):
                output_dir = self.processor.system_config.output_directory
                if os.path.exists(output_dir):
                    files = os.listdir(output_dir)
                    if files:
                        file_list = "\n".join([f"📄 {f}" for f in files])
                        return f"📁 Output Directory: {output_dir}\n\nFiles:\n{file_list}"
                    else:
                        return f"📁 Output Directory: {output_dir}\n📝 No files generated yet"
                else:
                    return f"📁 Output Directory: {output_dir}\n❌ Directory does not exist"
            else:
                return "No processing has been completed yet"
        except Exception as e:
            return f"❌ Error accessing output: {str(e)}"
    
    def run_tests(self):
        """Run unit tests"""
        self.log("Starting unit tests...")
        
        # Create temporary test config
        test_config_content = """[transcation]
TABLES = test_table1
BASEDIRECTORY = ./test_output
OUTPUT_DIRECTORY = ./test_output
START_DATE = 20250101
USERID = test_user
PASSWORD = test_pass

test_table1_Columns = col1,col2
test_table1_WhereCol_daycriteria = date_col
test_table1_NoOfDays = 30
"""
        
        test_config_file = "temp_test_config.conf"
        
        try:
            with open(test_config_file, 'w') as f:
                f.write(test_config_content)
            
            # Run tests
            test_results = []
            
            # Test 1: Config Manager
            try:
                config_manager = ConfigManager([test_config_file])
                self.log("✅ ConfigManager test passed")
                test_results.append(True)
            except Exception as e:
                self.log(f"❌ ConfigManager test failed: {e}")
                test_results.append(False)
            
            # Test 2: System Config
            try:
                system_config = config_manager.get_system_config()
                assert system_config.base_directory == "./test_output"
                self.log("✅ SystemConfig test passed")
                test_results.append(True)
            except Exception as e:
                self.log(f"❌ SystemConfig test failed: {e}")
                test_results.append(False)
            
            # Test 3: Tables List
            try:
                tables = config_manager.get_tables_list()
                assert len(tables) > 0
                self.log(f"✅ Tables list test passed: {tables}")
                test_results.append(True)
            except Exception as e:
                self.log(f"❌ Tables list test failed: {e}")
                test_results.append(False)
            
            passed = sum(test_results)
            total = len(test_results)
            
            if passed == total:
                result = f"🎉 All {total} tests passed!"
            else:
                result = f"⚠️ {passed}/{total} tests passed"
            
            self.log(f"Test results: {result}")
            return result, self.get_log_text()
            
        except Exception as e:
            error_msg = f"❌ Error running tests: {str(e)}"
            self.log(error_msg)
            return error_msg, self.get_log_text()
        finally:
            if os.path.exists(test_config_file):
                os.remove(test_config_file)
    
    def clear_log(self):
        """Clear log messages"""
        self.log_messages.clear()
        self.log("Log cleared")
        return "Log cleared", self.get_log_text()


def create_interface():
    """Create the Gradio interface"""
    app = SimpleTrainingDataApp()
    
    with gr.Blocks(title="Training Data Selection System") as interface:
        
        gr.Markdown("# 🎯 Training Data Selection System")
        gr.Markdown("### Automated database table analysis and training data preparation")
        
        with gr.Row():
            with gr.Column(scale=1):
                
                # Configuration Section
                gr.Markdown("## 📁 Configuration")
                
                config_file_upload = gr.File(
                    label="Upload Config File (.conf)",
                    file_types=[".conf"]
                )
                
                with gr.Row():
                    add_config_btn = gr.Button("Add Config", variant="primary")
                    create_sample_btn = gr.Button("Create Sample")
                
                config_list = gr.Textbox(
                    label="Config Files",
                    value=app.get_config_list(),
                    interactive=False,
                    lines=4
                )
                
                # Processing Section
                gr.Markdown("## 🚀 Processing")
                
                section_dropdown = gr.Dropdown(
                    label="Section",
                    choices=["transcation", "risk", "market", "ref"],
                    value="transcation"
                )
                
                with gr.Row():
                    test_btn = gr.Button("🧪 Test Config")
                    start_btn = gr.Button("▶️ Start Processing", variant="primary")
                
                status_display = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False
                )
                
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh")
                    output_btn = gr.Button("📁 View Output")
                
                # Testing Section
                gr.Markdown("## 🧪 Testing")
                
                run_tests_btn = gr.Button("Run Tests")
                test_results = gr.Textbox(
                    label="Test Results",
                    value="Click 'Run Tests' to start",
                    interactive=False,
                    lines=3
                )
            
            with gr.Column(scale=2):
                
                # Log Section
                gr.Markdown("## 📝 Processing Log")
                
                log_display = gr.Textbox(
                    label="Log Messages",
                    value=app.get_log_text(),
                    interactive=False,
                    lines=20,
                    max_lines=25
                )
                
                with gr.Row():
                    refresh_log_btn = gr.Button("🔄 Refresh Log")
                    clear_log_btn = gr.Button("🗑️ Clear Log")
                
                # Output Section
                gr.Markdown("## 📤 Output Information")
                
                output_info = gr.Textbox(
                    label="Output Details",
                    value="No processing completed yet",
                    interactive=False,
                    lines=8
                )
        
        # Help Section
        gr.Markdown("""
        ## 📖 Quick Help
        
        **Steps to use:**
        1. Upload or create a configuration file
        2. Select the processing section (transcation, risk, market, ref)
        3. Test your configuration
        4. Start processing and monitor the log
        5. View output when complete
        
        **Configuration format:**
        ```
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
        """)
        
        # Event handlers
        add_config_btn.click(
            fn=app.add_config_file,
            inputs=[config_file_upload],
            outputs=[status_display, config_list, log_display]
        )
        
        create_sample_btn.click(
            fn=app.create_sample_config,
            outputs=[status_display, config_list, log_display]
        )
        
        test_btn.click(
            fn=app.test_configuration,
            inputs=[section_dropdown],
            outputs=[status_display, log_display]
        )
        
        start_btn.click(
            fn=app.start_processing,
            inputs=[section_dropdown],
            outputs=[status_display, log_display]
        )
        
        refresh_btn.click(
            fn=app.get_processing_status,
            outputs=[status_display, log_display]
        )
        
        output_btn.click(
            fn=app.view_output_info,
            outputs=[output_info]
        )
        
        run_tests_btn.click(
            fn=app.run_tests,
            outputs=[test_results, log_display]
        )
        
        refresh_log_btn.click(
            fn=app.get_log_text,
            outputs=[log_display]
        )
        
        clear_log_btn.click(
            fn=app.clear_log,
            outputs=[status_display, log_display]
        )
    
    return interface


def run_tests_cli():
    """Run tests from command line"""
    app = SimpleTrainingDataApp()
    result, log_text = app.run_tests()
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(result)
    print("\nLog messages:")
    print(log_text)


def create_sample_config_cli():
    """Create sample config from command line"""
    app = SimpleTrainingDataApp()
    result, config_list, log_text = app.create_sample_config()
    print(result)


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test":
            run_tests_cli()
        elif command == "sample":
            create_sample_config_cli()
        elif command == "help":
            print("Training Data Selection System - Simple Gradio Interface")
            print("Usage:")
            print("  python simple_gradio_ui.py          - Run web interface")
            print("  python simple_gradio_ui.py test     - Run test suite")
            print("  python simple_gradio_ui.py sample   - Create sample config")
            print("  python simple_gradio_ui.py help     - Show this help")
        else:
            print(f"Unknown command: {command}")
            print("Use 'help' to see available commands")
    else:
        # Run the Gradio interface
        interface = create_interface()
        print("Starting Training Data Selection System...")
        print("Opening web interface at: http://localhost:7860")
        interface.launch(
            server_name="127.0.0.1",  # Local access only
            server_port=7860,
            share=False,
            debug=False,
            show_error=True
        )


if __name__ == "__main__":
    main()