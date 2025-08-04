import json
import logging
import os
import asyncio
from typing import Dict, List, Any, Tuple
from datetime import datetime
import re

# Import your existing client manager for GPT-4 calls
from app.utils.client_manager import client_manager

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BIRDDataProcessor:
    """
    BIRD Dataset Processor and Annotator
    
    Transforms BIRD schema format into your internal knowledge graph format
    with GPT-4 powered intelligent annotation for:
    - Table descriptions
    - Column descriptions
    - Business aliases
    - Categorical value mappings
    """
    
    def __init__(self, output_dir: str = "processed_bird_data"):
        self.output_dir = output_dir
        self.processed_count = 0
        self.annotation_cache = {}  # Cache to avoid re-annotating similar schemas
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"✅ BIRD Data Processor initialized. Output: {output_dir}")
    
    async def process_bird_files(self, bird_files: List[str]) -> List[str]:
        """
        Process multiple BIRD JSON files and generate annotated versions.
        
        Args:
            bird_files: List of BIRD JSON file paths
            
        Returns:
            List of generated output file paths
        """
        logger.info(f"🚀 Processing {len(bird_files)} BIRD files...")
        
        output_files = []
        
        for bird_file in bird_files:
            try:
                logger.info(f"📂 Processing: {bird_file}")
                output_file = await self.process_single_bird_file(bird_file)
                if output_file:
                    output_files.append(output_file)
                    logger.info(f"✅ Generated: {output_file}")
                else:
                    logger.warning(f"⚠️ Failed to process: {bird_file}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing {bird_file}: {e}")
                continue
        
        logger.info(f"🎉 Completed processing. Generated {len(output_files)} files.")
        return output_files
    
    async def process_single_bird_file(self, bird_file_path: str) -> str:
        """Process a single BIRD JSON file."""
        
        # Load BIRD data
        with open(bird_file_path, 'r', encoding='utf-8') as f:
            bird_data = json.load(f)
        
        if not isinstance(bird_data, list):
            logger.error(f"Expected list format in {bird_file_path}")
            return None
        
        # Transform each database schema
        training_data = []
        
        for db_schema in bird_data:
            try:
                transformed_tables = await self.transform_database_schema(db_schema)
                training_data.extend(transformed_tables)
                
            except Exception as e:
                logger.error(f"Failed to transform schema {db_schema.get('db_id', 'unknown')}: {e}")
                continue
        
        # Generate output file
        base_filename = os.path.splitext(os.path.basename(bird_file_path))[0]
        output_filename = f"bird_processed_{base_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Save in your internal format
        output_data = {
            "source": "BIRD_dataset",
            "processed_at": datetime.now().isoformat(),
            "original_file": bird_file_path,
            "total_tables": len(training_data),
            "trainingdata": training_data
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    async def transform_database_schema(self, db_schema: Dict) -> List[Dict]:
        """Transform a single BIRD database schema to internal format."""
        
        db_id = db_schema.get('db_id', 'unknown')
        table_names_original = db_schema.get('table_names_original', [])
        table_names_clean = db_schema.get('table_names', table_names_original)
        column_names_original = db_schema.get('column_names_original', [])
        column_names_clean = db_schema.get('column_names', column_names_original)
        column_types = db_schema.get('column_types', [])
        primary_keys = db_schema.get('primary_keys', [])
        foreign_keys = db_schema.get('foreign_keys', [])
        
        logger.info(f"🔄 Transforming database: {db_id} ({len(table_names_original)} tables)")
        
        transformed_tables = []
        
        # Process each table
        for table_idx, table_name_original in enumerate(table_names_original):
            table_name_clean = table_names_clean[table_idx] if table_idx < len(table_names_clean) else table_name_original
            
            # Extract columns for this table
            table_columns = []
            for col_idx, (col_table_idx, col_name_original) in enumerate(column_names_original):
                if col_table_idx == table_idx:  # Column belongs to this table
                    col_name_clean = column_names_clean[col_idx][1] if col_idx < len(column_names_clean) else col_name_original
                    col_type = column_types[col_idx] if col_idx < len(column_types) else 'unknown'
                    
                    # Create column structure
                    column_data = {
                        "columnname": col_name_original,
                        "cleaned_name": col_name_clean,
                        "datatype": col_type,
                        "mapped_col_type": self._map_column_type(col_type),
                        "nullable": True,  # Default - BIRD doesn't specify
                        "is_primary_key": col_idx in primary_keys,
                        "is_foreign_key": any(fk[0] == col_idx or fk[1] == col_idx for fk in foreign_keys),
                        # Placeholders for GPT-4 annotation
                        "columnDescription": "",
                        "columnAlias": [],
                        "provide_distinct": "NO",
                        "distinct_values": [],
                        "distinct_value_map": {}
                    }
                    
                    table_columns.append(column_data)
            
            # Create table structure
            table_data = {
                "tablename": f"{db_id}.{table_name_original}",
                "original_name": table_name_original,
                "cleaned_name": table_name_clean,
                "database_id": db_id,
                # Placeholders for GPT-4 annotation
                "tableDescription": "",
                "tableAlias": [],
                "tableSpecificRules": "",
                "columns": table_columns
            }
            
            transformed_tables.append(table_data)
        
        # Now annotate with GPT-4
        annotated_tables = await self.annotate_tables_with_gpt4(transformed_tables, db_id)
        
        return annotated_tables
    
    async def annotate_tables_with_gpt4(self, tables: List[Dict], db_id: str) -> List[Dict]:
        """Use GPT-4 to intelligently annotate table and column metadata."""
        
        logger.info(f"🤖 Starting GPT-4 annotation for {len(tables)} tables in {db_id}")
        
        annotated_tables = []
        
        for table_idx, table_data in enumerate(tables, 1):
            try:
                logger.info(f"🧠 Annotating table {table_idx}/{len(tables)}: {table_data['original_name']}")
                
                # Check cache first
                cache_key = f"{db_id}_{table_data['original_name']}"
                if cache_key in self.annotation_cache:
                    logger.info(f"💾 Using cached annotation for {table_data['original_name']}")
                    annotated_table = self.annotation_cache[cache_key].copy()
                else:
                    # Get GPT-4 annotation
                    annotated_table = await self.get_table_annotation(table_data, db_id)
                    self.annotation_cache[cache_key] = annotated_table.copy()
                
                annotated_tables.append(annotated_table)
                
                # Add small delay to avoid rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"❌ Failed to annotate table {table_data['original_name']}: {e}")
                # Use original table data as fallback
                annotated_tables.append(table_data)
                continue
        
        logger.info(f"✅ Completed GPT-4 annotation for {db_id}")
        return annotated_tables
    
    async def get_table_annotation(self, table_data: Dict, db_id: str) -> Dict:
        """Get comprehensive table and column annotations from GPT-4."""
        
        table_name = table_data['original_name']
        cleaned_name = table_data['cleaned_name']
        columns = table_data['columns']
        
        # Create comprehensive prompt
        annotation_prompt = f"""You are a database expert specializing in schema analysis and metadata generation. 

**TASK:** Analyze this table schema and provide comprehensive business-friendly annotations.

**DATABASE CONTEXT:** {db_id}
**TABLE:** {table_name} (also known as: {cleaned_name})

**COLUMNS:**
{self._format_columns_for_prompt(columns)}

**REQUIRED OUTPUT FORMAT (JSON):**
{{
    "table_description": "Clear business description of what this table contains and its purpose",
    "table_aliases": ["alias1", "alias2", "alias3"],
    "table_rules": "Any specific business rules or constraints for this table",
    "columns": [
        {{
            "original_name": "column_name",
            "description": "Business-friendly description of this column",
            "aliases": ["alias1", "alias2"],
            "should_provide_distinct": true/false,
            "sample_values": ["val1", "val2", "val3"],
            "value_mappings": {{"code1": "Description1", "code2": "Description2"}}
        }}
    ]
}}

**ANNOTATION GUIDELINES:**
1. **Table Description**: Focus on business purpose, not technical implementation
2. **Table Aliases**: Include common business terms, abbreviations, and synonyms
3. **Column Descriptions**: Explain business meaning, not just data type
4. **Column Aliases**: Include common business terms and abbreviations
5. **Distinct Values**: Set to true for categorical/code columns (status, type, category, etc.)
6. **Sample Values**: Provide realistic business examples
7. **Value Mappings**: For codes, provide human-readable descriptions

**BUSINESS CONTEXT HINTS:**
- Financial domain: account, transaction, loan, card, client, district
- E-commerce: customer, product, order, payment
- Healthcare: patient, treatment, diagnosis, medication
- Manufacturing: product, supplier, inventory, order

**RESPOND WITH ONLY THE JSON:**"""

        try:
            # Call GPT-4 using your internal client manager
            response = await client_manager.ask_gpt(annotation_prompt)
            
            # Parse the JSON response
            annotation_data = self._parse_gpt4_response(response)
            
            # Apply annotations to table data
            annotated_table = self._apply_annotations(table_data, annotation_data)
            
            return annotated_table
            
        except Exception as e:
            logger.error(f"GPT-4 annotation failed for {table_name}: {e}")
            # Return original data with minimal enhancement
            return self._apply_minimal_annotations(table_data)
    
    def _format_columns_for_prompt(self, columns: List[Dict]) -> str:
        """Format columns for GPT-4 prompt."""
        column_lines = []
        for col in columns:
            flags = []
            if col.get('is_primary_key'):
                flags.append('PK')
            if col.get('is_foreign_key'):
                flags.append('FK')
            
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            column_lines.append(f"- {col['columnname']} ({col['datatype']}){flag_str}")
        
        return "\n".join(column_lines)
    
    def _parse_gpt4_response(self, response: str) -> Dict:
        """Parse GPT-4 JSON response with error handling."""
        try:
            # Clean response
            response_clean = response.strip()
            
            # Remove code blocks if present
            if response_clean.startswith('```json'):
                response_clean = response_clean.replace('```json', '').replace('```', '').strip()
            elif response_clean.startswith('```'):
                response_clean = response_clean.replace('```', '').strip()
            
            # Find JSON content
            json_start = response_clean.find('{')
            json_end = response_clean.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = response_clean[json_start:json_end]
                return json.loads(json_content)
            else:
                raise ValueError("No valid JSON found in response")
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse GPT-4 JSON response: {e}")
            logger.debug(f"Response was: {response[:200]}...")
            return self._create_fallback_annotation()
        except Exception as e:
            logger.warning(f"Unexpected error parsing GPT-4 response: {e}")
            return self._create_fallback_annotation()
    
    def _create_fallback_annotation(self) -> Dict:
        """Create fallback annotation when GPT-4 parsing fails."""
        return {
            "table_description": "Table description not available",
            "table_aliases": [],
            "table_rules": "",
            "columns": []
        }
    
    def _apply_annotations(self, table_data: Dict, annotation_data: Dict) -> Dict:
        """Apply GPT-4 annotations to table data."""
        
        # Apply table-level annotations
        annotated_table = table_data.copy()
        annotated_table["tableDescription"] = annotation_data.get("table_description", "")
        annotated_table["tableAlias"] = annotation_data.get("table_aliases", [])
        annotated_table["tableSpecificRules"] = annotation_data.get("table_rules", "")
        
        # Apply column-level annotations
        annotated_columns = []
        column_annotations = {col["original_name"]: col for col in annotation_data.get("columns", [])}
        
        for col_data in table_data["columns"]:
            col_name = col_data["columnname"]
            annotated_col = col_data.copy()
            
            # Find annotation for this column
            col_annotation = column_annotations.get(col_name, {})
            
            if col_annotation:
                annotated_col["columnDescription"] = col_annotation.get("description", "")
                annotated_col["columnAlias"] = col_annotation.get("aliases", [])
                annotated_col["provide_distinct"] = "YES" if col_annotation.get("should_provide_distinct", False) else "NO"
                annotated_col["distinct_values"] = col_annotation.get("sample_values", [])
                
                # Handle value mappings
                value_mappings = col_annotation.get("value_mappings", {})
                if value_mappings:
                    annotated_col["distinct_value_map"] = value_mappings
                    # Add mapped values to distinct_values if not already present
                    for code in value_mappings.keys():
                        if code not in annotated_col["distinct_values"]:
                            annotated_col["distinct_values"].append(code)
            
            annotated_columns.append(annotated_col)
        
        annotated_table["columns"] = annotated_columns
        return annotated_table
    
    def _apply_minimal_annotations(self, table_data: Dict) -> Dict:
        """Apply minimal annotations when GPT-4 fails."""
        
        annotated_table = table_data.copy()
        
        # Basic table description based on name
        table_name = table_data["original_name"].lower()
        if "account" in table_name:
            annotated_table["tableDescription"] = "Account-related information"
        elif "transaction" in table_name or "trans" in table_name:
            annotated_table["tableDescription"] = "Transaction records"
        elif "customer" in table_name or "client" in table_name:
            annotated_table["tableDescription"] = "Customer/client information"
        elif "product" in table_name:
            annotated_table["tableDescription"] = "Product catalog information"
        else:
            annotated_table["tableDescription"] = f"Data table for {table_name}"
        
        # Basic aliases
        annotated_table["tableAlias"] = [table_data["cleaned_name"]] if table_data["cleaned_name"] != table_data["original_name"] else []
        
        # Apply basic column descriptions
        annotated_columns = []
        for col_data in table_data["columns"]:
            annotated_col = col_data.copy()
            col_name = col_data["columnname"].lower()
            
            # Basic column description
            if "id" in col_name:
                annotated_col["columnDescription"] = f"Unique identifier for {col_name.replace('_id', '').replace('id', '')}"
            elif "name" in col_name:
                annotated_col["columnDescription"] = f"Name field for {col_name.replace('_name', '').replace('name', '')}"
            elif "date" in col_name or "time" in col_name:
                annotated_col["columnDescription"] = f"Date/time field for {col_name}"
            elif "amount" in col_name or "price" in col_name:
                annotated_col["columnDescription"] = f"Monetary amount for {col_name}"
            else:
                annotated_col["columnDescription"] = f"Data field: {col_name}"
            
            # Basic aliases
            if col_data["cleaned_name"] != col_data["columnname"]:
                annotated_col["columnAlias"] = [col_data["cleaned_name"]]
            
            annotated_columns.append(annotated_col)
        
        annotated_table["columns"] = annotated_columns
        return annotated_table
    
    def _map_column_type(self, bird_type: str) -> str:
        """Map BIRD column types to your internal type system."""
        type_mapping = {
            'integer': 'INTEGER',
            'text': 'VARCHAR',
            'real': 'DECIMAL',
            'date': 'DATE',
            'datetime': 'TIMESTAMP',
            'boolean': 'BOOLEAN',
            'blob': 'BLOB'
        }
        return type_mapping.get(bird_type.lower(), 'VARCHAR')
    
    def generate_processing_summary(self, output_files: List[str]) -> str:
        """Generate processing summary report."""
        
        summary_data = {
            "processing_completed_at": datetime.now().isoformat(),
            "total_files_processed": len(output_files),
            "output_files": output_files,
            "annotation_cache_size": len(self.annotation_cache),
            "next_steps": [
                "Review generated files in processed_bird_data/ directory",
                "Run: python kg_builder_v2.py with new BIRD files",
                "Verify knowledge graph construction",
                "Test similarity search with BIRD data"
            ]
        }
        
        summary_file = os.path.join(self.output_dir, f"processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"📋 Processing summary saved to: {summary_file}")
        return summary_file


async def process_bird_datasets():
    """Main function to process BIRD datasets."""
    
    print("🚀 BIRD Dataset Processor & Annotator")
    print("=" * 60)
    
    # Initialize processor
    processor = BIRDDataProcessor()
    
    # Define BIRD files to process
    bird_files = [
        "train_tables_bird_short.json",
        "dev_tables_bird_short.json"
        # Add more BIRD files as needed
    ]
    
    # Check if files exist
    existing_files = [f for f in bird_files if os.path.exists(f)]
    if not existing_files:
        print("❌ No BIRD files found!")
        print("💡 Please ensure BIRD JSON files are in the current directory:")
        for f in bird_files:
            print(f"   - {f}")
        return
    
    print(f"📂 Found {len(existing_files)} BIRD files to process:")
    for f in existing_files:
        print(f"   ✅ {f}")
    
    try:
        # Process all files
        output_files = await processor.process_bird_datasets(existing_files)
        
        # Generate summary
        summary_file = processor.generate_processing_summary(output_files)
        
        print("\n🎉 BIRD Processing Completed Successfully!")
        print("=" * 60)
        print(f"📊 Results:")
        print(f"   📁 Generated {len(output_files)} processed files")
        print(f"   💾 Files saved to: processed_bird_data/")
        print(f"   📋 Summary report: {summary_file}")
        
        print(f"\n🔧 Next Steps:")
        print(f"   1. Review generated files in processed_bird_data/")
        print(f"   2. Update kg_builder_v2.py to include BIRD files:")
        print(f"      metadata_files.extend(glob.glob('processed_bird_data/bird_processed_*.json'))")
        print(f"   3. Run: python kg_builder_v2.py")
        print(f"   4. Test enhanced knowledge graph with BIRD data")
        
    except Exception as e:
        print(f"❌ BIRD processing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(process_bird_datasets())
