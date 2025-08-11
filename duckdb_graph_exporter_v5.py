#!/usr/bin/env python3
"""
DuckDB Knowledge Graph Exporter
Export DuckDB knowledge graph to interactive HTML (PyVis compatible)
Updated version to work with DuckDB instead of Neo4j
"""

import json
import html
import duckdb
import os
from typing import Dict, List, Any
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    print("⚠️  PyVis not available. Will create basic HTML visualization.")

class DuckDBGraphExporter:
    """DuckDB graph exporter with proper schema understanding"""
    
    def __init__(self, db_path: str = "knowledge_graph_v3.duckdb"):
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"DuckDB file not found: {db_path}")
        
        self.conn = duckdb.connect(db_path, read_only=True)
        self.db_path = db_path
        
    def _debug_graph_structure(self):
        """Debug function to understand the actual graph structure"""
        print("🔍 Debugging DuckDB graph structure...")
        
        # Check what tables exist
        tables_info = self.conn.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'main'
            ORDER BY table_name
        """).fetchall()
        
        print(f"📊 Available tables:")
        for table in tables_info:
            print(f"   {table[0]}")
        
        # Check table counts
        try:
            table_count = self.conn.execute("SELECT COUNT(*) FROM tables").fetchone()[0]
            print(f"   📋 Tables: {table_count}")
        except:
            print("   📋 Tables: Table 'tables' not found")
        
        try:
            column_count = self.conn.execute("SELECT COUNT(*) FROM columns").fetchone()[0]
            print(f"   📋 Columns: {column_count}")
        except:
            print("   📋 Columns: Table 'columns' not found")
        
        try:
            relationship_count = self.conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
            print(f"   🔗 Relationships: {relationship_count}")
        except:
            print("   🔗 Relationships: Table 'relationships' not found")
        
        # Check relationship types
        try:
            rel_types = self.conn.execute("""
                SELECT relationship_type, COUNT(*) as count
                FROM relationships 
                GROUP BY relationship_type
                ORDER BY count DESC
            """).fetchall()
            
            print(f"🔗 Relationship types:")
            for rel_type, count in rel_types:
                print(f"   {rel_type}: {count}")
        except:
            print("   Could not query relationship types")
        
        # Check table types
        try:
            table_types = self.conn.execute("""
                SELECT table_type, COUNT(*) as count
                FROM tables 
                GROUP BY table_type
                ORDER BY count DESC
            """).fetchall()
            
            print(f"📊 Table types:")
            for table_type, count in table_types:
                print(f"   {table_type}: {count}")
        except:
            print("   Could not query table types")

    def export_to_pyvis_simple(self, output_file: str = "knowledge_graph_fixed.html"):
        """Enhanced PyVis export reading from DuckDB"""
        if not PYVIS_AVAILABLE:
            return self.export_to_basic_html(output_file)
        
        print("🎨 Creating interactive HTML with PyVis from DuckDB...")
        
        # Debug the graph structure first
        self._debug_graph_structure()
        
        # Create basic network with enhanced configuration
        net = Network(
            height="800px", 
            width="100%", 
            bgcolor="#1e1e1e", 
            font_color="white",
            select_menu=True,
            filter_menu=True
        )
        
        # Get tables data
        print("  📊 Fetching tables from DuckDB...")
        try:
            tables = self.conn.execute("""
                SELECT name, 
                       COALESCE(description, '') AS description,
                       COALESCE(aliases, '[]') AS aliases,
                       COALESCE(table_type, 'unknown') AS table_type
                FROM tables 
                ORDER BY name
            """).fetchall()
            
            print(f"  Found {len(tables)} tables")
        except Exception as e:
            print(f"Error fetching columns: {e}")
            columns = []
        
        try:
            relationships = self.conn.execute("""
                SELECT from_table, to_table, 
                       COALESCE(from_column, '') AS from_column, 
                       COALESCE(to_column, '') AS to_column,
                       COALESCE(relationship_type, 'UNKNOWN') AS relationship_type,
                       COALESCE(confidence, 0.0) AS confidence
                FROM relationships
            """).fetchall()
        except Exception as e:
            print(f"Error fetching relationships: {e}")
            relationships = []
        
        # Get embedding metadata
        try:
            embedding_info = self.conn.execute("""
                SELECT provider, dimensions, model_version
                FROM embedding_metadata
                LIMIT 1
            """).fetchone()
        except:
            embedding_info = None
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DuckDB Knowledge Graph - Table View</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; }}
                .section {{ margin: 30px 0; }}
                .table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                .table th, .table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; vertical-align: top; }}
                .table th {{ background-color: #f2f2f2; font-weight: bold; }}
                .table tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .stat {{ text-align: center; padding: 20px; background: #4CAF50; color: white; border-radius: 8px; }}
                h1, h2 {{ color: #333; }}
                .description {{ max-width: 400px; word-wrap: break-word; }}
                .info {{ background: #d4edda; padding: 15px; margin: 20px 0; border-radius: 5px; border-left: 4px solid #28a745; }}
                .table-name {{ font-family: monospace; background: #f8f9fa; padding: 2px 4px; border-radius: 3px; }}
                .embedding-info {{ background: #cce5ff; padding: 15px; margin: 20px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 DuckDB Knowledge Graph Overview</h1>
                
                <div class="info">
                    <strong>Database:</strong> {self.db_path}<br/>
                    <strong>Database Size:</strong> {os.path.getsize(self.db_path) / (1024*1024):.2f} MB
        """
        
        if embedding_info:
            provider, dimensions, model_version = embedding_info
            html_content += f"""<br/>
                    <strong>Embedding Provider:</strong> {provider} ({model_version})<br/>
                    <strong>Embedding Dimensions:</strong> {dimensions}
        """
        
        html_content += """
                </div>
                
                <div class="stats">
                    <div class="stat">
                        <h3>""" + str(len(tables)) + """</h3>
                        <p>Tables</p>
                    </div>
                    <div class="stat">
                        <h3>""" + str(len(columns)) + """</h3>
                        <p>Columns</p>
                    </div>
                    <div class="stat">
                        <h3>""" + str(len(relationships)) + """</h3>
                        <p>Relationships</p>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📋 Tables</h2>
                    <table class="table">
                        <tr>
                            <th>Table Name</th>
                            <th>Short Name</th>
                            <th>Type</th>
                            <th>Description</th>
                        </tr>
        """
        
        for table in tables:
            name, description, aliases_json, table_type = table
            short_name = name.split('.')[-1] if '.' in name else name
            description_text = description[:300] if description else 'No description available'
            
            html_content += f"""
                        <tr>
                            <td class="table-name">{html.escape(name)}</td>
                            <td><strong>{html.escape(short_name)}</strong></td>
                            <td>{html.escape(table_type)}</td>
                            <td class="description">{html.escape(description_text)}</td>
                        </tr>
            """
        
        html_content += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>🔗 Table Relationships</h2>
                    <table class="table">
                        <tr>
                            <th>From Table</th>
                            <th>To Table</th>
                            <th>Join Condition</th>
                            <th>Relationship Type</th>
                            <th>Confidence</th>
                        </tr>
        """
        
        for relationship in relationships:
            from_table, to_table, from_col, to_col, rel_type, confidence = relationship
            from_short = from_table.split('.')[-1]
            to_short = to_table.split('.')[-1]
            join_condition = f"{from_col} → {to_col}" if from_col and to_col else "Unknown condition"
            
            html_content += f"""
                        <tr>
                            <td class="table-name">{html.escape(from_short)}</td>
                            <td class="table-name">{html.escape(to_short)}</td>
                            <td>{html.escape(join_condition)}</td>
                            <td>{html.escape(rel_type)}</td>
                            <td>{confidence:.2f}</td>
                        </tr>
            """
        
        html_content += f"""
                    </table>
                </div>
                
                <div class="section">
                    <h2>📊 Columns (Sample)</h2>
                    <table class="table">
                        <tr>
                            <th>Table</th>
                            <th>Column Name</th>
                            <th>Category</th>
                            <th>Data Type</th>
                            <th>Description</th>
                        </tr>
        """
        
        for column in columns[:100]:  # Limit to first 100 columns for display
            table_name, column_name, full_name, description, data_type, column_category = column
            table_short = table_name.split('.')[-1] if table_name else 'Unknown'
            description_text = description[:200] if description else 'No description'
            
            html_content += f"""
                        <tr>
                            <td class="table-name">{html.escape(table_short)}</td>
                            <td><strong>{html.escape(column_name)}</strong></td>
                            <td>{html.escape(column_category)}</td>
                            <td>{html.escape(data_type)}</td>
                            <td class="description">{html.escape(description_text)}</td>
                        </tr>
            """
        
        html_content += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>💡 Usage Notes</h2>
                    <ul>
                        <li>This is a static HTML view of your DuckDB knowledge graph</li>
                        <li>For interactive visualization, install PyVis: <code>pip install pyvis</code></li>
                        <li>Query the DuckDB database directly for advanced analysis</li>
                        <li>The graph includes semantic similarity relationships via embeddings</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Basic HTML view saved to: {output_file}")
        print(f"   📊 {len(tables)} tables")
        print(f"   📋 {len(columns)} columns")
        print(f"   🔗 {len(relationships)} relationships")
        return output_file
    
    def get_graph_statistics(self):
        """Get comprehensive statistics about the knowledge graph."""
        print("📊 Gathering DuckDB knowledge graph statistics...")
        
        stats = {}
        
        try:
            # Basic counts
            stats['table_count'] = self.conn.execute("SELECT COUNT(*) FROM tables").fetchone()[0]
            stats['column_count'] = self.conn.execute("SELECT COUNT(*) FROM columns").fetchone()[0]
            stats['relationship_count'] = self.conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
            
            try:
                stats['value_count'] = self.conn.execute("SELECT COUNT(*) FROM values").fetchone()[0]
            except:
                stats['value_count'] = 0
            
            # Table type distribution
            table_types = self.conn.execute("""
                SELECT table_type, COUNT(*) as count 
                FROM tables 
                GROUP BY table_type 
                ORDER BY count DESC
            """).fetchall()
            stats['table_types'] = dict(table_types)
            
            # Relationship type distribution  
            rel_types = self.conn.execute("""
                SELECT relationship_type, COUNT(*) as count 
                FROM relationships 
                GROUP BY relationship_type 
                ORDER BY count DESC
            """).fetchall()
            stats['relationship_types'] = dict(rel_types)
            
            # Column category distribution
            col_categories = self.conn.execute("""
                SELECT column_category, COUNT(*) as count 
                FROM columns 
                GROUP BY column_category 
                ORDER BY count DESC
            """).fetchall()
            stats['column_categories'] = dict(col_categories)
            
            # Embedding information
            try:
                embedding_info = self.conn.execute("""
                    SELECT provider, dimensions, model_version
                    FROM embedding_metadata
                    LIMIT 1
                """).fetchone()
                if embedding_info:
                    stats['embedding_provider'] = embedding_info[0]
                    stats['embedding_dimensions'] = embedding_info[1]
                    stats['embedding_model'] = embedding_info[2]
            except:
                stats['embedding_provider'] = 'Unknown'
            
        except Exception as e:
            print(f"Error gathering statistics: {e}")
        
        return stats
    
    def close(self):
        """Close the DuckDB connection."""
        if hasattr(self, 'conn'):
            self.conn.close()

def main():
    """Main export function for DuckDB knowledge graph"""
    print("🚀 DuckDB Knowledge Graph Exporter v2.0")
    print("=" * 60)
    
    # Default database path - update this if your DB is named differently
    db_path = "knowledge_graph_v3.duckdb"
    
    # Check if the default database exists, if not, look for alternatives
    if not os.path.exists(db_path):
        possible_paths = [
            "knowledge_graph.duckdb",
            "knowledge_graph_v2.duckdb", 
            "kg_v3.duckdb",
            "kg_builder_v2.duckdb"
        ]
        
        print(f"❌ Default database '{db_path}' not found.")
        print("🔍 Searching for alternative database files...")
        
        for path in possible_paths:
            if os.path.exists(path):
                db_path = path
                print(f"✅ Found database: {path}")
                break
        else:
            print(f"\n❌ No DuckDB database found. Please ensure one of these files exists:")
            for path in [db_path] + possible_paths:
                print(f"   - {path}")
            print("\n💡 Make sure you've run the DuckDB knowledge graph builder first!")
            return
    else:
        print(f"✅ Using database: {db_path}")
    
    try:
        exporter = DuckDBGraphExporter(db_path)
        
        print(f"\n🔍 Analyzing database: {db_path}")
        
        # Get and display statistics
        stats = exporter.get_graph_statistics()
        print(f"\n📊 Graph Statistics:")
        print(f"   📋 Tables: {stats.get('table_count', 0)}")
        print(f"   📊 Columns: {stats.get('column_count', 0)}")
        print(f"   🔗 Relationships: {stats.get('relationship_count', 0)}")
        print(f"   🏷️  Values: {stats.get('value_count', 0)}")
        
        if 'embedding_provider' in stats:
            print(f"   🤖 Embedding Provider: {stats['embedding_provider']}")
            if 'embedding_dimensions' in stats:
                print(f"   📏 Embedding Dimensions: {stats['embedding_dimensions']}")
        
        # Show distribution info if available
        if 'table_types' in stats and stats['table_types']:
            print(f"\n📋 Table Types:")
            for table_type, count in list(stats['table_types'].items())[:5]:
                print(f"   {table_type}: {count}")
        
        if 'relationship_types' in stats and stats['relationship_types']:
            print(f"\n🔗 Relationship Types:")
            for rel_type, count in list(stats['relationship_types'].items())[:5]:
                print(f"   {rel_type}: {count}")
        
        print("\n" + "="*60)
        print("1️⃣ Creating Interactive Graph Visualization...")
        html_file = exporter.export_to_pyvis_simple("knowledge_graph_fixed.html")
        
        print("\n2️⃣ Creating HTML Table View...")
        table_file = exporter.export_to_basic_html("knowledge_graph_table.html")
        
        print("\n" + "=" * 60)
        print("✅ EXPORT COMPLETED!")
        print("=" * 60)
        
        print(f"\n📁 Files created:")
        if html_file:
            print(f"   🌐 {html_file} - Interactive PyVis visualization")
        if table_file:
            print(f"   📋 {table_file} - Table-based overview")
        
        print(f"\n🎯 Next Steps:")
        print(f"   1. Open '{html_file}' in your web browser")
        print(f"   2. Use the interactive controls to explore relationships")
        print(f"   3. Hover over nodes for detailed information")
        print(f"   4. Use '{table_file}' for detailed data inspection")
        
        if stats.get('table_count', 0) == 0:
            print(f"\n⚠️  Warning: No tables found in the database.")
            print(f"   Make sure you've run the knowledge graph builder successfully.")
        
        exporter.close()
        
    except Exception as e:
        print(f"❌ Error during export: {e}")
        print(f"\n🔧 Troubleshooting tips:")
        print(f"   1. Ensure the DuckDB file exists and is readable")
        print(f"   2. Check that the knowledge graph builder completed successfully")
        print(f"   3. Verify the database contains the expected schema (tables, columns, relationships)")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
            print(f"  Error fetching tables: {e}")
            return None
        
        # Get relationships data
        print("  🔗 Fetching relationships from DuckDB...")
        try:
            relationships = self.conn.execute("""
                SELECT from_table, to_table, 
                       COALESCE(from_column, '') AS from_column, 
                       COALESCE(to_column, '') AS to_column,
                       COALESCE(relationship_type, 'UNKNOWN') AS relationship_type,
                       COALESCE(confidence, 0.0) AS confidence
                FROM relationships
            """).fetchall()
            
            print(f"  Found {len(relationships)} relationships")
        except Exception as e:
            print(f"  Error fetching relationships: {e}")
            relationships = []
        
        # Add table nodes
        print("  📋 Adding table nodes...")
        for table in tables:
            name, description, aliases_json, table_type = table
            
            # Parse aliases safely
            try:
                aliases = json.loads(aliases_json) if aliases_json and aliases_json != '[]' else []
                if isinstance(aliases, str):
                    aliases = [aliases]
            except:
                aliases = []
            
            short_name = name.split('.')[-1] if '.' in name else name
            description_text = description[:200] + "..." if len(description) > 200 else description
            
            # Create enhanced tooltip
            tooltip = f"<b>{name}</b><br/>"
            if description_text:
                tooltip += f"{description_text}<br/>"
            if aliases:
                aliases_str = ', '.join(aliases) if isinstance(aliases, list) else str(aliases)
                tooltip += f"<br/><b>Aliases:</b> {aliases_str}"
            tooltip += f"<br/><b>Type:</b> {table_type}"
            
            # Color coding based on table type
            if table_type == 'fact' or 'FACT' in name.upper():
                color = "#ff6b6b"  # Red for fact tables
            elif table_type == 'dimension' or 'DIM' in name.upper():
                color = "#4ecdc4"  # Teal for dimension tables
            elif table_type == 'reference' or 'REF' in name.upper():
                color = "#45b7d1"  # Blue for reference tables
            elif table_type == 'market_data' or 'MARKET' in name.upper():
                color = "#f39c12"  # Orange for market data
            else:
                color = "#96ceb4"  # Green for other tables
            
            net.add_node(
                name,
                label=short_name,
                title=tooltip,
                color=color,
                size=25,
                shape="box"
            )
        
        # Get columns data with better detection strategy
        print("  📋 Fetching columns from DuckDB...")
        try:
            # First try to get columns with a more flexible query
            columns = self.conn.execute("""
                SELECT c.table_name, c.name AS column_name, c.full_name,
                       COALESCE(c.description, '') AS description, 
                       COALESCE(c.data_type, 'unknown') AS data_type,
                       COALESCE(c.column_category, 'unknown') AS column_category
                FROM columns c
                WHERE c.table_name IN (SELECT name FROM tables)
                ORDER BY c.table_name, 
                         CASE c.column_category 
                             WHEN 'id' THEN 1 
                             WHEN 'key' THEN 2 
                             WHEN 'code' THEN 3 
                             WHEN 'measure' THEN 4
                             ELSE 5 
                         END,
                         c.name
            """).fetchall()
            
            print(f"  Found {len(columns)} columns total")
            
            # If no columns found, try without filtering
            if not columns:
                columns = self.conn.execute("""
                    SELECT c.table_name, c.name AS column_name, c.full_name,
                           COALESCE(c.description, '') AS description, 
                           COALESCE(c.data_type, 'unknown') AS data_type,
                           COALESCE(c.column_category, 'unknown') AS column_category
                    FROM columns c
                    ORDER BY c.table_name, c.name
                    LIMIT 500
                """).fetchall()
                print(f"  Fallback query found {len(columns)} columns")
                
        except Exception as e:
            print(f"  Error fetching columns: {e}")
            columns = []
        
        # Group columns by table and add columns with smart filtering
        table_columns = {}
        for column in columns:
            table_name, column_name, full_name, description, data_type, column_category = column
            if table_name not in table_columns:
                table_columns[table_name] = []
            table_columns[table_name].append({
                'name': column_name,
                'full_name': full_name,
                'description': description,
                'data_type': data_type,
                'category': column_category
            })
        
        # Add columns to the graph with intelligent selection
        print("  📋 Adding column nodes...")
        column_count = 0
        max_columns_per_table = 8  # Increased from 5
        
        for table_name, columns_list in table_columns.items():
            # Prioritize important columns
            prioritized_columns = sorted(columns_list, key=lambda x: (
                0 if x['category'] in ['id', 'key'] else 1,
                0 if 'id' in x['name'].lower() else 1,
                0 if 'key' in x['name'].lower() or 'sk' in x['name'].lower() else 1,
                0 if 'code' in x['name'].lower() else 1,
                1 if x['category'] == 'unknown' else 0,  # Show unknown categories last
                x['name']
            ))
            
            # Take top columns per table
            selected_columns = prioritized_columns[:max_columns_per_table]
            
            for column in selected_columns:
                column_id = f"{table_name}.{column['name']}"
                column_label = column['name']
                
                # Create informative tooltip
                tooltip = f"<b>{column['name']}</b><br/>"
                tooltip += f"Table: {table_name.split('.')[-1]}<br/>"
                tooltip += f"Type: {column['data_type']}<br/>"
                tooltip += f"Category: {column['category']}<br/>"
                if column['description']:
                    desc_preview = column['description'][:150]
                    if len(column['description']) > 150:
                        desc_preview += "..."
                    tooltip += f"Description: {desc_preview}"
                
                # Enhanced color coding based on column category/type
                category = column['category']
                col_name_lower = column['name'].lower()
                
                if category == 'id' or 'id' in col_name_lower:
                    col_color = "#f39c12"  # Orange for ID columns
                elif category == 'key' or any(k in col_name_lower for k in ['key', '_sk']):
                    col_color = "#e67e22"  # Dark orange for key columns
                elif category == 'code' or 'code' in col_name_lower:
                    col_color = "#9b59b6"  # Purple for code columns
                elif category == 'date' or any(d in col_name_lower for d in ['date', 'time', 'timestamp']):
                    col_color = "#1abc9c"  # Teal for date/time columns
                elif category == 'measure' or any(m in col_name_lower for m in ['amount', 'price', 'quantity', 'volume']):
                    col_color = "#e74c3c"  # Red for measures
                else:
                    col_color = "#34495e"  # Dark gray for other columns
                
                # Adjust size based on importance
                node_size = 15 if category in ['id', 'key'] else 12
                
                net.add_node(
                    column_id,
                    label=column_label,
                    title=tooltip,
                    color=col_color,
                    size=node_size,
                    shape="dot"
                )
                
                # Add HAS_COLUMN edge from table to column
                net.add_edge(table_name, column_id, color="#95a5a6", width=1, title="HAS_COLUMN")
                column_count += 1
        
        print(f"  Added {column_count} column nodes")
        
        # Add table relationships with enhanced visualization
        print(f"  🔗 Adding {len(relationships)} table relationships...")
        edge_count = 0
        for relationship in relationships:
            from_table, to_table, from_col, to_col, rel_type, confidence = relationship
            
            if from_table != to_table:  # Avoid self-loops
                # Create informative edge label
                edge_label = ""
                if from_col and to_col:
                    edge_label = f"{from_col} → {to_col}"
                
                # Enhanced edge styling based on relationship type
                if rel_type == 'BUSINESS_RULE':
                    edge_color = "#e74c3c"  # Red for business rules
                    edge_width = 5
                elif rel_type == 'FOREIGN_KEY':
                    edge_color = "#f39c12"  # Orange for FK
                    edge_width = 4
                elif rel_type == 'SURROGATE_KEY':
                    edge_color = "#9b59b6"  # Purple for surrogate keys
                    edge_width = 4
                elif rel_type == 'SMCP_MATCH':
                    edge_color = "#e67e22"  # Dark orange for SMCP
                    edge_width = 4
                elif rel_type == 'EMBEDDING_SIMILARITY':
                    edge_color = "#3498db"  # Blue for embedding similarity
                    edge_width = 2
                elif rel_type == 'ERD_DEFINED_GEMINI':
                    edge_color = "#27ae60"  # Green for ERD defined
                    edge_width = 4
                elif confidence > 0.8:
                    edge_color = "#27ae60"  # Green for high confidence
                    edge_width = 4
                else:
                    edge_color = "#95a5a6"  # Gray for others
                    edge_width = 2
                
                # Add tooltip for edge
                edge_title = f"<b>{rel_type}</b><br/>"
                edge_title += f"{from_table.split('.')[-1]} → {to_table.split('.')[-1]}<br/>"
                if edge_label:
                    edge_title += f"Join: {edge_label}<br/>"
                if confidence > 0:
                    edge_title += f"Confidence: {confidence:.2f}"
                
                net.add_edge(
                    from_table, 
                    to_table, 
                    label=edge_label,
                    title=edge_title,
                    color=edge_color,
                    width=edge_width
                )
                edge_count += 1
        
        print(f"  Added {edge_count} relationship edges")
        
        # Enhanced physics and layout options
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100},
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 95,
              "springConstant": 0.04,
              "damping": 0.09
            }
          },
          "nodes": {
            "font": {"size": 14, "color": "white"},
            "borderWidth": 2,
            "shadow": true
          },
          "edges": {
            "font": {"size": 12, "color": "white"},
            "smooth": {"type": "continuous"},
            "shadow": true
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "selectConnectedEdges": false
          },
          "configure": {
            "enabled": false
          }
        }
        """)
        
        # Save the graph with proper HTML generation
        try:
            net.save_graph(output_file)
            
            # Verify the file was created and contains PyVis content
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    content = f.read()
                    if 'vis.Network' in content and 'vis.DataSet' in content:
                        print(f"✅ Enhanced interactive graph saved to: {output_file}")
                        print(f"   📊 {len(tables)} tables")
                        print(f"   📋 {column_count} columns")
                        print(f"   🔗 {edge_count} cross-table relationships") 
                        print(f"   🌐 Open in browser to explore!")
                        return output_file
                    else:
                        print(f"⚠️ Warning: Generated file doesn't contain expected PyVis content")
                        return self._create_manual_pyvis_html(tables, relationships, columns, output_file)
            else:
                print(f"⚠️ Warning: PyVis failed to create file, creating manual version")
                return self._create_manual_pyvis_html(tables, relationships, columns, output_file)
                
        except Exception as e:
            print(f"⚠️ PyVis save failed: {e}, creating manual HTML")
            return self._create_manual_pyvis_html(tables, relationships, columns, output_file)
    
    def _create_manual_pyvis_html(self, tables, relationships, columns, output_file):
        """Create PyVis HTML manually when automatic generation fails"""
        print("📝 Creating manual PyVis HTML...")
        
        # Prepare nodes data
        nodes_data = []
        edges_data = []
        
        # Add table nodes
        for table in tables:
            name, description, aliases_json, table_type = table
            
            # Parse aliases safely
            try:
                aliases = json.loads(aliases_json) if aliases_json and aliases_json != '[]' else []
                if isinstance(aliases, str):
                    aliases = [aliases]
            except:
                aliases = []
            
            short_name = name.split('.')[-1] if '.' in name else name
            description_text = description[:200] + "..." if len(description) > 200 else description
            
            # Create enhanced tooltip
            tooltip = f"<b>{name}</b><br/>"
            if description_text:
                tooltip += f"{description_text}<br/>"
            if aliases:
                aliases_str = ', '.join(aliases) if isinstance(aliases, list) else str(aliases)
                tooltip += f"<br/><b>Aliases:</b> {aliases_str}"
            tooltip += f"<br/><b>Type:</b> {table_type}"
            
            # Color coding based on table type
            if table_type == 'fact' or 'FACT' in name.upper():
                color = "#ff6b6b"
            elif table_type == 'dimension' or 'DIM' in name.upper():
                color = "#4ecdc4"
            elif table_type == 'reference' or 'REF' in name.upper():
                color = "#45b7d1"
            elif table_type == 'market_data' or 'MARKET' in name.upper():
                color = "#f39c12"
            else:
                color = "#96ceb4"
            
            nodes_data.append({
                "id": name,
                "label": short_name,
                "title": tooltip,
                "color": color,
                "size": 25,
                "shape": "box",
                "font": {"color": "white"}
            })
        
        # Add column nodes
        try:
            # Get columns grouped by table
            table_columns = {}
            columns_query = self.conn.execute("""
                SELECT c.table_name, c.name AS column_name, c.full_name,
                       COALESCE(c.description, '') AS description, 
                       COALESCE(c.data_type, 'unknown') AS data_type,
                       COALESCE(c.column_category, 'unknown') AS column_category
                FROM columns c
                WHERE c.table_name IN (SELECT name FROM tables)
                ORDER BY c.table_name, c.name
                LIMIT 300
            """).fetchall()
            
            for column in columns_query:
                table_name, column_name, full_name, description, data_type, column_category = column
                if table_name not in table_columns:
                    table_columns[table_name] = []
                table_columns[table_name].append({
                    'name': column_name,
                    'description': description,
                    'data_type': data_type,
                    'category': column_category
                })
            
            # Add important columns for each table
            for table_name, columns_list in table_columns.items():
                # Sort by importance and take top 5
                sorted_columns = sorted(columns_list, key=lambda x: (
                    0 if x['category'] in ['id', 'key'] else 1,
                    0 if 'id' in x['name'].lower() else 1,
                    x['name']
                ))[:5]
                
                for column in sorted_columns:
                    column_id = f"{table_name}.{column['name']}"
                    
                    # Create tooltip
                    tooltip = f"<b>{column['name']}</b><br/>"
                    tooltip += f"Table: {table_name.split('.')[-1]}<br/>"
                    tooltip += f"Type: {column['data_type']}<br/>"
                    tooltip += f"Category: {column['category']}"
                    if column['description']:
                        tooltip += f"<br/>Description: {column['description'][:100]}..."
                    
                    # Color based on category
                    if column['category'] == 'id' or 'id' in column['name'].lower():
                        col_color = "#f39c12"
                    elif column['category'] == 'key' or 'key' in column['name'].lower():
                        col_color = "#e67e22"
                    elif column['category'] == 'code' or 'code' in column['name'].lower():
                        col_color = "#9b59b6"
                    elif 'date' in column['name'].lower() or 'time' in column['name'].lower():
                        col_color = "#1abc9c"
                    else:
                        col_color = "#34495e"
                    
                    nodes_data.append({
                        "id": column_id,
                        "label": column['name'],
                        "title": tooltip,
                        "color": col_color,
                        "size": 12,
                        "shape": "dot",
                        "font": {"color": "white"}
                    })
                    
                    # Add edge from table to column
                    edges_data.append({
                        "from": table_name,
                        "to": column_id,
                        "color": "#95a5a6",
                        "width": 1,
                        "title": "HAS_COLUMN"
                    })
                    
        except Exception as e:
            print(f"Warning: Could not add columns: {e}")
        
        # Add relationship edges
        for relationship in relationships:
            from_table, to_table, from_col, to_col, rel_type, confidence = relationship
            
            if from_table != to_table:
                edge_label = f"{from_col} → {to_col}" if from_col and to_col else ""
                
                # Color based on relationship type
                if rel_type == 'ERD_DEFINED_GEMINI':
                    edge_color = "#27ae60"
                    edge_width = 4
                elif rel_type == 'BUSINESS_RULE':
                    edge_color = "#e74c3c"
                    edge_width = 5
                elif rel_type == 'SURROGATE_KEY':
                    edge_color = "#9b59b6"
                    edge_width = 4
                elif confidence > 0.8:
                    edge_color = "#27ae60"
                    edge_width = 4
                else:
                    edge_color = "#3498db"
                    edge_width = 3
                
                edge_title = f"<b>{rel_type}</b><br/>"
                edge_title += f"{from_table.split('.')[-1]} → {to_table.split('.')[-1]}<br/>"
                if edge_label:
                    edge_title += f"Join: {edge_label}<br/>"
                if confidence > 0:
                    edge_title += f"Confidence: {confidence:.2f}"
                
                edges_data.append({
                    "from": from_table,
                    "to": to_table,
                    "label": edge_label,
                    "title": edge_title,
                    "color": edge_color,
                    "width": edge_width
                })
        
        # Create the HTML content
        nodes_json = json.dumps(nodes_data)
        edges_json = json.dumps(edges_data)
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css" />
    <title>Knowledge Graph - Interactive View</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        #mynetwork {{ width: 100%; height: 800px; background-color: #1e1e1e; border: 1px solid lightgray; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        h1 {{ color: #333; text-align: center; }}
        .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .stat {{ text-align: center; padding: 15px; background: #4CAF50; color: white; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Interactive Knowledge Graph</h1>
        
        <div class="stats">
            <div class="stat">
                <h3>{len(tables)}</h3>
                <p>Tables</p>
            </div>
            <div class="stat">
                <h3>{len(nodes_data) - len(tables)}</h3>
                <p>Columns</p>
            </div>
            <div class="stat">
                <h3>{len(edges_data)}</h3>
                <p>Relationships</p>
            </div>
        </div>
        
        <div id="mynetwork"></div>
    </div>

    <script type="text/javascript">
        var nodes = new vis.DataSet({nodes_json});
        var edges = new vis.DataSet({edges_json});

        var container = document.getElementById('mynetwork');
        var data = {{
            nodes: nodes,
            edges: edges
        }};
        var options = {{
            physics: {{
                enabled: true,
                stabilization: {{"iterations": 100}},
                barnesHut: {{
                    "gravitationalConstant": -8000,
                    "centralGravity": 0.3,
                    "springLength": 95,
                    "springConstant": 0.04,
                    "damping": 0.09
                }}
            }},
            nodes: {{
                font: {{"size": 14, "color": "white"}},
                borderWidth: 2,
                shadow: true
            }},
            edges: {{
                font: {{"size": 12, "color": "white"}},
                smooth: {{"type": "continuous"}},
                shadow: true
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100
            }}
        }};

        var network = new vis.Network(container, data, options);
        
        console.log('Knowledge graph loaded with', nodes.length, 'nodes and', edges.length, 'edges');
    </script>
</body>
</html>"""
        
        # Save the HTML file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Manual PyVis HTML created: {output_file}")
        print(f"   📊 {len(tables)} tables")
        print(f"   📋 {len(nodes_data) - len(tables)} columns")
        print(f"   🔗 {len(edges_data)} total edges")
        print(f"   🌐 Open in browser to explore!")
        
        return output_file
    
    def export_to_basic_html(self, output_file: str = "knowledge_graph_basic.html"):
        """Enhanced basic HTML export reading from DuckDB"""
        print("📋 Creating basic HTML table view from DuckDB...")
        
        # Debug first
        self._debug_graph_structure()
        
        # Get data from DuckDB
        try:
            tables = self.conn.execute("""
                SELECT name, 
                       COALESCE(description, '') AS description,
                       COALESCE(aliases, '[]') AS aliases,
                       COALESCE(table_type, 'unknown') AS table_type
                FROM tables 
                ORDER BY name
            """).fetchall()
        except Exception as e:
            print(f"Error fetching tables: {e}")
            tables = []
        
        try:
            columns = self.conn.execute("""
                SELECT c.table_name, c.name AS column_name, c.full_name,
                       COALESCE(c.description, '') AS description, 
                       COALESCE(c.data_type, 'unknown') AS data_type,
                       COALESCE(c.column_category, 'unknown') AS column_category
                FROM columns c
                ORDER BY c.table_name, c.name
                LIMIT 500
            """).fetchall()
        except Exception as e: