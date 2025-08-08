import json
import asyncio
from typing import Dict, List, Any, Optional
from app.utils.client_manager import client_manager

class BIRDDatasetEnricher:
    """
    A system to enrich BIRD dataset descriptions with detailed, contextual information
    for building knowledge graphs and improving text-to-SQL performance.
    """
    
    def __init__(self):
        self.domain_knowledge = {
            "european_football": {
                "context": "European football/soccer league data including matches, teams, divisions",
                "common_terms": ["division", "match", "team", "goal", "season", "league"]
            },
            "sales": {
                "context": "Retail/business sales data with weather correlations",
                "common_terms": ["store", "item", "sales", "units", "revenue", "weather"]
            },
            "financial": {
                "context": "Financial and trading data",
                "common_terms": ["trade", "amount", "currency", "settlement", "portfolio"]
            }
        }

    async def enrich_table_description(self, table_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a single table's description using GPT
        """
        table_name = table_data['tablename']
        database_id = table_data['database_id']
        columns = table_data['columns']
        
        # Build context for the table
        column_info = []
        for col in columns:
            col_info = f"- {col['columnname']} ({col['datatype']})"
            if col['is_primary_key']:
                col_info += " [PRIMARY KEY]"
            if col['is_foreign_key']:
                col_info += " [FOREIGN KEY]"
            column_info.append(col_info)
        
        column_summary = "\n".join(column_info)
        
        table_enrichment_prompt = f"""You are a database schema expert. Analyze this table and provide a rich, contextual description.

**Table Information:**
- Table Name: {table_name}
- Database: {database_id}
- Current Description: {table_data['tableDescription']}

**Columns:**
{column_summary}

**Task:** Create a comprehensive table description that includes:
1. What business domain this table represents
2. What real-world entities or processes it captures
3. How it might be used in queries
4. Relationships to other potential tables
5. Common business questions it could answer

**Output Format (JSON):**
{{
    "enriched_description": "detailed description here",
    "business_domain": "domain name",
    "primary_purpose": "main purpose of the table",
    "common_use_cases": ["use case 1", "use case 2", "use case 3"],
    "query_patterns": ["common query pattern 1", "common query pattern 2"],
    "business_questions": ["What business question 1?", "What business question 2?"]
}}

**Respond with only the JSON:**"""

        try:
            response = await client_manager.ask_gpt(table_enrichment_prompt)
            enrichment_data = json.loads(response)
            
            # Update the table data
            table_data['enriched_tableDescription'] = enrichment_data['enriched_description']
            table_data['business_domain'] = enrichment_data.get('business_domain', '')
            table_data['primary_purpose'] = enrichment_data.get('primary_purpose', '')
            table_data['common_use_cases'] = enrichment_data.get('common_use_cases', [])
            table_data['query_patterns'] = enrichment_data.get('query_patterns', [])
            table_data['business_questions'] = enrichment_data.get('business_questions', [])
            
            return table_data
        except Exception as e:
            print(f"Error enriching table {table_name}: {str(e)}")
            return table_data

    async def enrich_column_description(self, column_data: Dict[str, Any], table_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a single column's description using GPT
        """
        column_name = column_data['columnname']
        table_name = table_context['tablename']
        database_id = table_context['database_id']
        
        # Get table business context if available
        business_domain = table_context.get('business_domain', 'unknown')
        table_purpose = table_context.get('primary_purpose', table_context['tableDescription'])
        
        column_enrichment_prompt = f"""You are a database schema expert. Analyze this column and provide a rich, contextual description.

**Column Information:**
- Column Name: {column_name}
- Data Type: {column_data['datatype']} 
- Table: {table_name}
- Database: {database_id}
- Current Description: {column_data['columnDescription']}
- Is Primary Key: {column_data['is_primary_key']}
- Is Foreign Key: {column_data['is_foreign_key']}
- Nullable: {column_data['nullable']}

**Table Context:**
- Business Domain: {business_domain}
- Table Purpose: {table_purpose}

**Task:** Create a comprehensive column description that includes:
1. What real-world concept this column represents
2. Expected values or ranges
3. Business meaning and importance
4. How it's typically used in queries
5. Relationships to other columns
6. Data quality considerations

**Output Format (JSON):**
{{
    "enriched_description": "detailed description here",
    "business_meaning": "what this represents in business terms",
    "typical_values": "description of expected values/ranges",
    "query_usage": "how this column is typically used in SQL queries",
    "data_quality_notes": "important data quality considerations",
    "semantic_type": "semantic classification (e.g., identifier, measure, dimension, date, categorical)",
    "related_concepts": ["concept1", "concept2"]
}}

**Respond with only the JSON:**"""

        try:
            response = await client_manager.ask_gpt(column_enrichment_prompt)
            enrichment_data = json.loads(response)
            
            # Update the column data
            column_data['enriched_columnDescription'] = enrichment_data['enriched_description']
            column_data['business_meaning'] = enrichment_data.get('business_meaning', '')
            column_data['typical_values'] = enrichment_data.get('typical_values', '')
            column_data['query_usage'] = enrichment_data.get('query_usage', '')
            column_data['data_quality_notes'] = enrichment_data.get('data_quality_notes', '')
            column_data['semantic_type'] = enrichment_data.get('semantic_type', '')
            column_data['related_concepts'] = enrichment_data.get('related_concepts', [])
            
            return column_data
        except Exception as e:
            print(f"Error enriching column {column_name}: {str(e)}")
            return column_data

    async def generate_table_relationships(self, tables_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate relationship insights between tables
        """
        # Build summary of all tables
        table_summaries = []
        for table in tables_data:
            summary = f"- {table['tablename']}: {table.get('enriched_tableDescription', table['tableDescription'])}"
            table_summaries.append(summary)
        
        tables_summary = "\n".join(table_summaries)
        
        relationship_prompt = f"""You are a database schema expert. Analyze these tables and identify potential relationships and integration patterns.

**Tables in Database:**
{tables_summary}

**Task:** Identify relationships, integration patterns, and query join opportunities.

**Output Format (JSON):**
{{
    "potential_joins": [
        {{
            "table1": "table_name_1",
            "table2": "table_name_2", 
            "join_type": "inner/left/right",
            "join_columns": {{"table1_col": "table2_col"}},
            "relationship_description": "description of relationship"
        }}
    ],
    "query_patterns": [
        "Common multi-table query pattern 1",
        "Common multi-table query pattern 2"
    ],
    "business_workflows": [
        "Business workflow that spans multiple tables 1",
        "Business workflow that spans multiple tables 2"  
    ]
}}

**Respond with only the JSON:**"""

        try:
            response = await client_manager.ask_gpt(relationship_prompt)
            relationships = json.loads(response)
            return relationships
        except Exception as e:
            print(f"Error generating relationships: {str(e)}")
            return {"potential_joins": [], "query_patterns": [], "business_workflows": []}

    async def process_dataset(self, input_file: str, output_file: str, batch_size: int = 3):
        """
        Main method to process the entire BIRD dataset
        """
        print(f"Loading dataset from {input_file}...")
        
        # Load the original dataset
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        enriched_data = data.copy()
        tables_data = enriched_data['trainingdata']
        
        print(f"Processing {len(tables_data)} tables...")
        
        # Step 1: Enrich table descriptions
        print("Step 1: Enriching table descriptions...")
        for i in range(0, len(tables_data), batch_size):
            batch = tables_data[i:i+batch_size]
            tasks = []
            
            for table_data in batch:
                task = self.enrich_table_description(table_data)
                tasks.append(task)
            
            # Process batch
            enriched_batch = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update the data
            for j, enriched_table in enumerate(enriched_batch):
                if not isinstance(enriched_table, Exception):
                    tables_data[i+j] = enriched_table
                    print(f"✓ Enriched table: {enriched_table['tablename']}")
                else:
                    print(f"✗ Failed to enrich table at index {i+j}: {str(enriched_table)}")
            
            # Small delay to avoid rate limits
            await asyncio.sleep(1)
        
        # Step 2: Enrich column descriptions
        print("Step 2: Enriching column descriptions...")
        for table_idx, table_data in enumerate(tables_data):
            columns = table_data['columns']
            
            for col_idx in range(0, len(columns), batch_size):
                batch = columns[col_idx:col_idx+batch_size]
                tasks = []
                
                for column_data in batch:
                    task = self.enrich_column_description(column_data, table_data)
                    tasks.append(task)
                
                # Process batch
                enriched_batch = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Update the data
                for j, enriched_column in enumerate(enriched_batch):
                    if not isinstance(enriched_column, Exception):
                        columns[col_idx+j] = enriched_column
                        print(f"✓ Enriched column: {table_data['tablename']}.{enriched_column['columnname']}")
                    else:
                        print(f"✗ Failed to enrich column at {table_idx}.{col_idx+j}: {str(enriched_column)}")
                
                # Small delay to avoid rate limits
                await asyncio.sleep(0.5)
        
        # Step 3: Generate relationships
        print("Step 3: Generating table relationships...")
        relationships = await self.generate_table_relationships(tables_data)
        enriched_data['table_relationships'] = relationships
        
        # Step 4: Add metadata
        enriched_data['enrichment_metadata'] = {
            "enriched_at": "2025-08-08T00:00:00",
            "enrichment_version": "1.0",
            "total_tables_processed": len(tables_data),
            "total_columns_processed": sum(len(table['columns']) for table in tables_data),
            "enrichment_features": [
                "table_descriptions",
                "column_descriptions", 
                "business_context",
                "semantic_types",
                "relationship_mapping",
                "query_patterns",
                "business_questions"
            ]
        }
        
        # Save enriched dataset
        print(f"Saving enriched dataset to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(enriched_data, f, indent=2)
        
        print("✅ Dataset enrichment completed!")
        print(f"Original tables: {len(tables_data)}")
        print(f"Total columns: {sum(len(table['columns']) for table in tables_data)}")
        print(f"Relationships identified: {len(relationships.get('potential_joins', []))}")
        
        return enriched_data

    def generate_embedding_features(self, enriched_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate structured features for embedding creation and knowledge graph
        """
        embedding_features = []
        
        for table_data in enriched_data['trainingdata']:
            # Table-level features
            table_feature = {
                "type": "table",
                "identifier": table_data['tablename'],
                "database_id": table_data['database_id'],
                "name": table_data['cleaned_name'],
                "description": table_data.get('enriched_tableDescription', table_data['tableDescription']),
                "business_domain": table_data.get('business_domain', ''),
                "primary_purpose": table_data.get('primary_purpose', ''),
                "use_cases": table_data.get('common_use_cases', []),
                "query_patterns": table_data.get('query_patterns', []),
                "business_questions": table_data.get('business_questions', [])
            }
            embedding_features.append(table_feature)
            
            # Column-level features
            for column_data in table_data['columns']:
                column_feature = {
                    "type": "column", 
                    "identifier": f"{table_data['tablename']}.{column_data['columnname']}",
                    "table_id": table_data['tablename'],
                    "database_id": table_data['database_id'],
                    "name": column_data['cleaned_name'],
                    "original_name": column_data['columnname'],
                    "datatype": column_data['datatype'],
                    "description": column_data.get('enriched_columnDescription', column_data['columnDescription']),
                    "business_meaning": column_data.get('business_meaning', ''),
                    "semantic_type": column_data.get('semantic_type', ''),
                    "query_usage": column_data.get('query_usage', ''),
                    "related_concepts": column_data.get('related_concepts', []),
                    "is_primary_key": column_data['is_primary_key'],
                    "is_foreign_key": column_data['is_foreign_key'],
                    "nullable": column_data['nullable']
                }
                embedding_features.append(column_feature)
        
        return embedding_features

# Usage example
async def main():
    """
    Main execution function
    """
    enricher = BIRDDatasetEnricher()
    
    # Process the dataset
    enriched_data = await enricher.process_dataset(
        input_file="bird_processed_train_tables_bird_short.json",
        output_file="bird_enriched_train_tables.json",
        batch_size=3  # Process 3 tables/columns at a time to avoid rate limits
    )
    
    # Generate embedding features
    embedding_features = enricher.generate_embedding_features(enriched_data)
    
    # Save embedding features
    with open("bird_embedding_features.json", 'w') as f:
        json.dump(embedding_features, f, indent=2)
    
    print(f"Generated {len(embedding_features)} embedding features")
    print("Files created:")
    print("- bird_enriched_train_tables.json (full enriched dataset)")
    print("- bird_embedding_features.json (structured features for embeddings)")

# Run the enrichment
if __name__ == "__main__":
    asyncio.run(main())
