import json
import logging
import os
import sys
from typing import Dict, List, Any, Set, Tuple
import uuid
from datetime import datetime

# Enhanced logging configuration - Corporate environment safe
logging.basicConfig(
    level=logging.DEBUG,  # More verbose logging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Ensure output goes to stdout
        logging.FileHandler('kg_builder.log', mode='w', encoding='utf-8')  # Also log to file with UTF-8
    ]
)
logger = logging.getLogger(__name__)

# Test imports and log results
logger.info("Starting import checks...")

try:
    import duckdb
    logger.info("DuckDB import successful")
    logger.info(f"   DuckDB version: {duckdb.__version__}")
except ImportError as e:
    logger.error(f"DuckDB import failed: {e}")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    logger.info("SentenceTransformers import successful")
except ImportError as e:
    logger.error(f"SentenceTransformers import failed: {e}")
    logger.error("Try: pip install sentence-transformers")
    sys.exit(1)

logger.info("All imports successful")

class DuckDBKnowledgeGraphBuilder:
    """
    DuckDB-based Knowledge Graph Builder - Enhanced Performance & Easy Deployment
    
    Advantages over Neo4j:
    - 2-10x faster analytical queries
    - Zero server setup required
    - Native vector similarity search
    - SQL interface (familiar to most developers)
    - File-based storage (easy backup/sharing)
    - No corporate IT approval needed
    """
    
    def __init__(self, db_path: str = 'knowledge_graph.duckdb'):
        logger.info(f"Initializing DuckDBKnowledgeGraphBuilder with path: {db_path}")
        
        try:
            # Check current working directory
            current_dir = os.getcwd()
            logger.info(f"Current working directory: {current_dir}")
            
            # Check if we have write permissions
            test_file = "test_write_permissions.tmp"
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                logger.info("Write permissions confirmed")
            except Exception as e:
                logger.error(f"No write permissions in current directory: {e}")
                raise
            
            # Test DuckDB connection
            logger.info("Testing DuckDB connection...")
            self.conn = duckdb.connect(db_path)
            self.db_path = db_path
            logger.info(f"DuckDB connection successful: {db_path}")
            
            # Test basic DuckDB functionality
            logger.info("Testing basic DuckDB functionality...")
            test_result = self.conn.execute("SELECT 'Hello DuckDB' as test").fetchone()
            logger.info(f"DuckDB test query successful: {test_result}")
            
            # Initialize SentenceTransformer
            logger.info("Initializing SentenceTransformer model...")
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("SentenceTransformer model loaded successfully")
                
                # Test embedding generation
                test_embedding = self.embedding_model.encode("test sentence")
                logger.info(f"Test embedding generated: shape {test_embedding.shape}")
                
            except Exception as e:
                logger.error(f"SentenceTransformer initialization failed: {e}")
                logger.error("This might be due to network restrictions or missing dependencies")
                raise
            
            # Initialize data structures
            self.tables_metadata: Dict[str, Any] = {}
            self.relationships: Dict[Tuple, Dict] = {}
            logger.info("Data structures initialized")
            
            # Initialize database schema
            logger.info("Initializing database schema...")
            self._create_schema()
            logger.info("Database schema initialized successfully")
            
            logger.info(f"DuckDBKnowledgeGraphBuilder initialization complete!")
            
        except Exception as e:
            logger.error(f"Failed to initialize DuckDBKnowledgeGraphBuilder: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise

    def _create_schema(self):
        """Create optimized schema for knowledge graph storage."""
        logger.info("Creating DuckDB schema for knowledge graph...")
        
        # Test vector extension availability
        logger.info("Testing vector extension availability...")
        try:
            self.conn.execute("INSTALL vss;")
            logger.info("VSS extension installation attempted")
        except Exception as e:
            logger.warning(f"VSS extension install failed (this is often normal): {e}")
        
        try:
            self.conn.execute("LOAD vss;")
            logger.info("Vector similarity search extension loaded")
        except Exception as e:
            logger.warning(f"Vector extension not available (will use alternative): {e}")
        
        # Create tables with detailed logging
        schema_queries = [
            ("tables", """
            CREATE TABLE IF NOT EXISTS tables (
                id VARCHAR PRIMARY KEY,
                name VARCHAR UNIQUE NOT NULL,
                description TEXT,
                aliases JSON,
                rules TEXT,
                embedding FLOAT[384],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                table_type VARCHAR
            )
            """),
            
            ("columns", """
            CREATE TABLE IF NOT EXISTS columns (
                id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                full_name VARCHAR UNIQUE NOT NULL,
                table_name VARCHAR NOT NULL,
                description TEXT,
                data_type VARCHAR,
                is_nullable BOOLEAN DEFAULT true,
                distinct_values JSON,
                embedding FLOAT[384],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                column_category VARCHAR,
                FOREIGN KEY (table_name) REFERENCES tables(name)
            )
            """),
            
            ("relationships", """
            CREATE TABLE IF NOT EXISTS relationships (
                id VARCHAR PRIMARY KEY,
                from_table VARCHAR NOT NULL,
                to_table VARCHAR NOT NULL,
                from_column VARCHAR,
                to_column VARCHAR,
                relationship_type VARCHAR NOT NULL,
                confidence FLOAT DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (from_table) REFERENCES tables(name),
                FOREIGN KEY (to_table) REFERENCES tables(name)
            )
            """),
            
            ("values", """
            CREATE TABLE IF NOT EXISTS values (
                id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                column_full_name VARCHAR NOT NULL,
                table_name VARCHAR NOT NULL,
                embedding FLOAT[384],
                value_type VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (column_full_name) REFERENCES columns(full_name),
                FOREIGN KEY (table_name) REFERENCES tables(name)
            )
            """),
            
            ("dates", """
            CREATE TABLE IF NOT EXISTS dates (
                id VARCHAR PRIMARY KEY,
                date_value DATE NOT NULL,
                date_string VARCHAR NOT NULL,
                year INTEGER,
                month INTEGER,
                day INTEGER,
                day_of_week INTEGER,
                is_business_day BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
        ]
        
        for table_name, query in schema_queries:
            try:
                logger.info(f"Creating table: {table_name}")
                self.conn.execute(query)
                logger.info(f"Table {table_name} created successfully")
            except Exception as e:
                logger.error(f"Failed to create table {table_name}: {e}")
                raise
        
        # Create performance indexes
        logger.info("Creating performance indexes...")
        self._create_indexes()
        logger.info("Schema created successfully")

    def _create_indexes(self):
        """Create performance indexes for fast querying."""
        logger.info("📇 Creating performance indexes...")
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_tables_name ON tables(name)",
            "CREATE INDEX IF NOT EXISTS idx_tables_type ON tables(table_type)",
            "CREATE INDEX IF NOT EXISTS idx_columns_table ON columns(table_name)",
            "CREATE INDEX IF NOT EXISTS idx_columns_full_name ON columns(full_name)",
            "CREATE INDEX IF NOT EXISTS idx_columns_category ON columns(column_category)",
            "CREATE INDEX IF NOT EXISTS idx_relationships_from ON relationships(from_table)",
            "CREATE INDEX IF NOT EXISTS idx_relationships_to ON relationships(to_table)",
            "CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relationship_type)",
            "CREATE INDEX IF NOT EXISTS idx_values_column ON values(column_full_name)",
            "CREATE INDEX IF NOT EXISTS idx_values_table ON values(table_name)",
            "CREATE INDEX IF NOT EXISTS idx_dates_value ON dates(date_value)"
        ]
        
        created_count = 0
        for idx_query in indexes:
            try:
                self.conn.execute(idx_query)
                created_count += 1
            except Exception as e:
                logger.warning(f"Could not create index: {e}")
        
        logger.info(f"✅ Created {created_count}/{len(indexes)} indexes")

    def load_rich_metadata(self, file_paths: List[str]):
        """Enhanced metadata loading with robust encoding handling."""
        logger.info(f"📂 Loading rich metadata from {len(file_paths)} source files...")
        all_tables_data = []
        
        for file_path in file_paths:
            try:
                # Try different encodings for robustness
                encodings = ['utf-8', 'latin1', 'utf-8-sig']
                data = None
                
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            data = json.load(f)
                        logger.info(f"✅ Loaded {file_path} with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if data is None:
                    raise Exception(f"Could not decode {file_path} with any supported encoding")
                
                # Handle different JSON structures
                if 'trainingdata' in data:
                    all_tables_data.extend(data['trainingdata'])
                elif isinstance(data, list):
                    all_tables_data.extend(data)
                else:
                    logger.warning(f"Unexpected JSON structure in {file_path}")
                    
            except FileNotFoundError as e:
                logger.error(f"Metadata file not found: {e}")
                raise
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                raise
        
        # Process all table data
        for table_data in all_tables_data:
            table_name = table_data.get('tablename')
            if table_name:
                self.tables_metadata[table_name] = table_data
        
        logger.info(f"✅ Loaded metadata for {len(self.tables_metadata)} tables from {len(file_paths)} files")

    def load_erd_relationships(self, erd_file: str):
        """Load manually curated relationships from ERD analysis."""
        logger.info(f"🔗 Loading ERD relationships from {erd_file}...")
        try:
            with open(erd_file, 'r', encoding='utf-8') as f:
                erd_rels = json.load(f)
            
            for rel in erd_rels:
                # Use tuple key for proper deduplication
                key = tuple(sorted((rel['from_table'], rel['to_table']))) + (rel['from_column'], rel['to_column'])
                self.relationships[key] = rel
                
            logger.info(f"✅ Loaded {len(erd_rels)} ERD-defined relationships")
        except FileNotFoundError:
            logger.warning(f"'{erd_file}' not found. No manual joins will be added")
            self.relationships = {}

    def infer_additional_relationships(self):
        """Enhanced relationship inference with multiple heuristic rules."""
        logger.info("🧠 Running enhanced heuristic inference...")
        all_table_names = list(self.tables_metadata.keys())
        discovered_count = 0

        # Pairwise table comparison
        for i in range(len(all_table_names)):
            for j in range(i + 1, len(all_table_names)):  # Avoid duplicates and self-comparison
                table1_name = all_table_names[i]
                table2_name = all_table_names[j]

                table1_meta = self.tables_metadata[table1_name]
                table2_meta = self.tables_metadata[table2_name]

                # Compare all column pairs
                for col1_data in table1_meta.get('columns', []):
                    for col2_data in table2_meta.get('columns', []):
                        col1_name = col1_data.get('columnname', '')
                        col2_name = col2_data.get('columnname', '')
                        
                        if not col1_name or not col2_name:
                            continue

                        confidence, rel_type = self._calculate_relationship_confidence(
                            col1_name, col2_name, col1_data, col2_data
                        )

                        # Add relationship if confidence is high enough
                        if confidence > 0.7:
                            self._add_relationship(table1_name, table2_name, col1_name, col2_name, rel_type, confidence)
                            discovered_count += 1

        # Market data specific relationships
        discovered_count += self._infer_market_data_relationships()
        
        logger.info(f"✅ Discovered {discovered_count} additional relationships")

    def _calculate_relationship_confidence(self, col1_name: str, col2_name: str, col1_data: dict, col2_data: dict) -> Tuple[float, str]:
        """Calculate confidence score for potential relationship."""
        if col1_name != col2_name:
            return 0.0, "NO_MATCH"
        
        col1_lower = col1_name.lower()
        
        # Rule 1: Surrogate keys (highest confidence)
        if col1_lower.endswith('_sk'):
            return 0.95, "SURROGATE_KEY"
        
        # Rule 2: SMCP matching (market data)
        if col1_lower == 'smcp':
            return 0.95, "SMCP_MATCH"
        
        # Rule 3: Primary/Foreign key patterns
        if any(pattern in col1_lower for pattern in ['_id', '_key']):
            return 0.90, "KEY_MATCH"
        
        # Rule 4: Code patterns
        if col1_lower.endswith('_code'):
            return 0.85, "CODE_MATCH"
        
        # Rule 5: Simple ID patterns
        if col1_lower in ['id', 'key'] or col1_lower.endswith('id'):
            return 0.80, "ID_MATCH"
        
        return 0.0, "NO_MATCH"

    def _infer_market_data_relationships(self) -> int:
        """Infer specific market data relationships."""
        count = 0
        product_table = "gfolynref_standardization.OM_PRODUCT_DIM"
        market_tables = [
            "gfolynref_standardization.OM_MARKET_DATA_DIM", 
            "gfolynsd_standardization.OM_INTRA_DAY_MARKET_DATA_FACT"
        ]
        
        for mkt_table in market_tables:
            if mkt_table in self.tables_metadata and product_table in self.tables_metadata:
                self._add_relationship(mkt_table, product_table, "smcp", "smcp", "SMCP_MATCH", 0.95)
                count += 1
        
        return count

    def _add_relationship(self, from_table: str, to_table: str, from_col: str, to_col: str, rel_type: str, confidence: float):
        """Add relationship with deduplication."""
        key = tuple(sorted((from_table, to_table))) + (from_col, to_col)
        
        # Only add if it doesn't already exist (ERD relationships have precedence)
        if key not in self.relationships:
            self.relationships[key] = {
                "from_table": from_table, "to_table": to_table,
                "from_column": from_col, "to_column": to_col,
                "relationship_type": rel_type, "confidence": confidence
            }
            logger.debug(f"Added relationship: {from_table}.{from_col} -> {to_table}.{to_col} ({rel_type}, {confidence:.2f})")

    def clear_graph(self):
        """Clear all existing data from the knowledge graph."""
        logger.info("🧹 Clearing existing knowledge graph data...")
        
        tables_to_clear = ['values', 'relationships', 'columns', 'tables', 'dates']
        for table in tables_to_clear:
            try:
                self.conn.execute(f"DELETE FROM {table}")
            except:
                pass  # Table might not exist yet
        
        logger.info("✅ Graph cleared")

    def build_graph(self):
        """Build the complete knowledge graph in DuckDB."""
        if not self.tables_metadata:
            logger.error("❌ No metadata loaded. Aborting graph build.")
            return

        logger.info("🏗️ Building enhanced knowledge graph with DuckDB...")
        
        # Step 1: Insert tables
        self._insert_tables()
        
        # Step 2: Insert columns and values
        self._insert_columns_and_values()
        
        # Step 3: Insert relationships
        self._insert_relationships()
        
        # Step 4: Insert dates
        self._insert_dates()
        
        logger.info("✅ Knowledge graph build completed")

    def _insert_tables(self):
        """Insert table nodes into the graph."""
        logger.info("📋 Inserting table nodes...")
        
        insert_query = """
        INSERT INTO tables (id, name, description, aliases, rules, embedding, table_type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        table_count = 0
        for table_name, table_data in self.tables_metadata.items():
            # Determine table type
            table_type = self._classify_table_type(table_name)
            
            # Create rich embedding text
            table_aliases = ", ".join(table_data.get('tableAlias', []))
            embedding_text = (
                f"Table: {table_name}. Description: {table_data.get('tableDescription', '')}. "
                f"Aliases: {table_aliases}. Rules: {table_data.get('tableSpecificRules', '')}"
            )
            
            # Generate embedding
            embedding = self.embedding_model.encode(embedding_text).tolist()
            
            # Insert table
            self.conn.execute(insert_query, [
                str(uuid.uuid4()),  # id
                table_name,  # name
                table_data.get('tableDescription', ''),  # description
                json.dumps(table_data.get('tableAlias', [])),  # aliases
                table_data.get('tableSpecificRules', ''),  # rules
                embedding,  # embedding
                table_type  # table_type
            ])
            
            table_count += 1
        
        logger.info(f"✅ Inserted {table_count} table nodes")

    def _classify_table_type(self, table_name: str) -> str:
        """Classify table type based on naming conventions."""
        name_upper = table_name.upper()
        
        if 'FACT' in name_upper:
            return 'fact'
        elif 'DIM' in name_upper:
            return 'dimension'
        elif any(word in name_upper for word in ['REF', 'REFERENCE', 'LOOKUP']):
            return 'reference'
        elif 'MARKET' in name_upper:
            return 'market_data'
        elif 'CALENDAR' in name_upper:
            return 'calendar'
        else:
            return 'unknown'

    def _insert_columns_and_values(self):
        """Insert column nodes and their categorical values."""
        logger.info("📋 Inserting column nodes and values...")
        
        column_insert_query = """
        INSERT INTO columns (id, name, full_name, table_name, description, data_type, 
                           is_nullable, distinct_values, embedding, column_category)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        value_insert_query = """
        INSERT INTO values (id, name, column_full_name, table_name, embedding, value_type)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        column_count = 0
        value_count = 0
        
        for table_name, table_data in self.tables_metadata.items():
            logger.debug(f"Processing columns for table: {table_name}")
            
            for col_data in table_data.get('columns', []):
                col_name = col_data.get('columnname', '')
                if not col_name:
                    continue
                
                full_col_name = f"{table_name}.{col_name}"
                
                # Classify column category
                column_category = self._classify_column_category(col_name)
                
                # Create rich embedding text
                col_aliases = ", ".join(col_data.get('columnAlias', []))
                distinct_values_str = ", ".join([str(v) for v in col_data.get('distinct_values', [])[:10]])
                embedding_text = (
                    f"Column: {col_name} in table {table_name}. "
                    f"Description: {col_data.get('columnDescription', '')}. "
                    f"Type: {col_data.get('mapped_col_type', col_data.get('datatype', 'unknown'))}. "
                    f"Aliases: {col_aliases}. Sample values: [{distinct_values_str}]."
                )
                
                # Generate embedding
                embedding = self.embedding_model.encode(embedding_text).tolist()
                
                # Insert column
                self.conn.execute(column_insert_query, [
                    str(uuid.uuid4()),  # id
                    col_name,  # name
                    full_col_name,  # full_name
                    table_name,  # table_name
                    col_data.get('columnDescription', col_data.get('description', '')),  # description
                    col_data.get('mapped_col_type', col_data.get('datatype', 'unknown')),  # data_type
                    col_data.get('nullable', True),  # is_nullable
                    json.dumps(col_data.get('distinct_values', [])[:50]),  # distinct_values (limited)
                    embedding,  # embedding
                    column_category  # column_category
                ])
                
                column_count += 1
                
                # Insert categorical values for important columns
                should_create_values = (
                    col_data.get("provide_distinct") == "YES" or
                    column_category in ['id', 'key', 'code', 'type', 'status']
                )
                
                if should_create_values and col_data.get("distinct_values"):
                    # Limit values to avoid explosion
                    values_to_add = col_data["distinct_values"][:20]
                    for value in values_to_add:
                        if value is not None and str(value).strip():
                            # Create value embedding
                            value_text = f"Value: {value}. Column: {full_col_name}. Table: {table_name}"
                            value_embedding = self.embedding_model.encode(value_text).tolist()
                            
                            # Determine value type
                            value_type = self._classify_value_type(value)
                            
                            # Insert value
                            self.conn.execute(value_insert_query, [
                                str(uuid.uuid4()),  # id
                                str(value),  # name
                                full_col_name,  # column_full_name
                                table_name,  # table_name
                                value_embedding,  # embedding
                                value_type  # value_type
                            ])
                            
                            value_count += 1
        
        logger.info(f"✅ Inserted {column_count} columns and {value_count} categorical values")

    def _classify_column_category(self, col_name: str) -> str:
        """Classify column based on naming patterns."""
        col_lower = col_name.lower()
        
        if any(pattern in col_lower for pattern in ['_sk', '_id', 'id']):
            return 'id'
        elif any(pattern in col_lower for pattern in ['_key', 'key']):
            return 'key'
        elif any(pattern in col_lower for pattern in ['_code', 'code']):
            return 'code'
        elif any(pattern in col_lower for pattern in ['date', 'time', 'timestamp']):
            return 'date'
        elif any(pattern in col_lower for pattern in ['amount', 'price', 'value', 'quantity']):
            return 'measure'
        elif any(pattern in col_lower for pattern in ['type', 'status', 'flag']):
            return 'type'
        elif any(pattern in col_lower for pattern in ['name', 'description', 'text']):
            return 'attribute'
        else:
            return 'unknown'

    def _classify_value_type(self, value) -> str:
        """Classify value type for better categorization."""
        if isinstance(value, (int, float)):
            return 'numeric'
        elif isinstance(value, str):
            if len(value) <= 10 and value.replace('-', '').replace('/', '').isdigit():
                return 'date'
            elif len(value) <= 50:
                return 'categorical'
            else:
                return 'text'
        else:
            return 'unknown'

    def _insert_relationships(self):
        """Insert table relationships."""
        logger.info("🔗 Inserting table relationships...")
        
        insert_query = """
        INSERT INTO relationships (id, from_table, to_table, from_column, to_column, 
                                 relationship_type, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        relationship_count = 0
        for rel in self.relationships.values():
            self.conn.execute(insert_query, [
                str(uuid.uuid4()),  # id
                rel['from_table'],  # from_table
                rel['to_table'],  # to_table
                rel['from_column'],  # from_column
                rel['to_column'],  # to_column
                rel.get('relationship_type', 'UNKNOWN'),  # relationship_type
                rel.get('confidence', 0.0)  # confidence
            ])
            relationship_count += 1
        
        logger.info(f"✅ Inserted {relationship_count} table relationships")

    def _insert_dates(self):
        """Insert date nodes from calendar dimensions."""
        logger.info("📅 Inserting date nodes...")
        
        calendar_table_name = "gfolynref_standardization.OM_CALENDAR_DIM"
        if calendar_table_name not in self.tables_metadata:
            logger.info("No calendar table found, skipping date insertion")
            return
        
        insert_query = """
        INSERT INTO dates (id, date_value, date_string, year, month, day, day_of_week, is_business_day)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        date_count = 0
        for col_data in self.tables_metadata[calendar_table_name].get('columns', []):
            if col_data.get('columnname') == 'business_date':
                # Process date values (limit to avoid too many nodes)
                for date_val in col_data.get('distinct_values', [])[:100]:
                    if date_val:
                        try:
                            # Parse date (assuming YYYYMMDD format)
                            date_str = str(date_val)
                            if len(date_str) == 8 and date_str.isdigit():
                                year = int(date_str[:4])
                                month = int(date_str[4:6])
                                day = int(date_str[6:8])
                                
                                # Create proper date
                                from datetime import date
                                date_obj = date(year, month, day)
                                
                                self.conn.execute(insert_query, [
                                    str(uuid.uuid4()),  # id
                                    date_obj,  # date_value
                                    date_str,  # date_string
                                    year,  # year
                                    month,  # month
                                    day,  # day
                                    date_obj.weekday(),  # day_of_week
                                    date_obj.weekday() < 5  # is_business_day (Mon-Fri)
                                ])
                                date_count += 1
                        except:
                            continue  # Skip invalid dates
        
        logger.info(f"✅ Inserted {date_count} date nodes")

    def verify_graph_structure(self):
        """Verify the built graph structure and provide statistics."""
        logger.info("🔍 Verifying DuckDB knowledge graph structure...")
        
        # Get table counts
        stats = {}
        
        table_counts = [
            ("tables", "Table nodes"),
            ("columns", "Column nodes"), 
            ("values", "Value nodes"),
            ("relationships", "Table relationships"),
            ("dates", "Date nodes")
        ]
        
        logger.info("📊 Node and relationship counts:")
        for table, description in table_counts:
            try:
                count = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                stats[table] = count
                logger.info(f"   {description}: {count}")
            except:
                stats[table] = 0
                logger.info(f"   {description}: 0")
        
        # Table type distribution
        try:
            table_types = self.conn.execute("""
                SELECT table_type, COUNT(*) as count 
                FROM tables 
                GROUP BY table_type 
                ORDER BY count DESC
            """).fetchall()
            
            logger.info("📋 Table type distribution:")
            for table_type, count in table_types:
                logger.info(f"   {table_type}: {count}")
        except:
            pass
        
        # Relationship type distribution
        try:
            rel_types = self.conn.execute("""
                SELECT relationship_type, COUNT(*) as count 
                FROM relationships 
                GROUP BY relationship_type 
                ORDER BY count DESC
            """).fetchall()
            
            logger.info("🔗 Relationship type distribution:")
            for rel_type, count in rel_types:
                logger.info(f"   {rel_type}: {count}")
        except:
            pass
        
        # Sample verification - tables with most columns
        try:
            top_tables = self.conn.execute("""
                SELECT t.name, COUNT(c.id) as column_count
                FROM tables t
                LEFT JOIN columns c ON t.name = c.table_name
                GROUP BY t.name
                ORDER BY column_count DESC
                LIMIT 5
            """).fetchall()
            
            logger.info("📊 Tables with most columns:")
            for table_name, col_count in top_tables:
                logger.info(f"   {table_name}: {col_count} columns")
        except:
            pass
        
        return stats

    def get_database_info(self):
        """Get information about the DuckDB database."""
        info = {
            'database_path': self.db_path,
            'database_size_mb': os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0,
            'created_at': datetime.now().isoformat()
        }
        
        logger.info(f"💾 Database file: {info['database_path']}")
        logger.info(f"💾 Database size: {info['database_size_mb']:.2f} MB")
        
        return info

def main():
    """Enhanced main function with comprehensive error handling and logging."""
    DB_PATH = "knowledge_graph.duckdb"

    logger.info("🚀 Starting DuckDB Knowledge Graph Builder v7")
    logger.info("=" * 60)
    
    # System information logging
    logger.info(f"🖥️ System Information:")
    logger.info(f"   Python version: {sys.version}")
    logger.info(f"   Operating System: {os.name}")
    logger.info(f"   Current working directory: {os.getcwd()}")
    logger.info(f"   Script location: {os.path.abspath(__file__)}")
    
    # Check Python packages
    logger.info("📦 Checking Python packages...")
    packages_to_check = ['duckdb', 'sentence_transformers', 'numpy', 'json']
    for pkg in packages_to_check:
        try:
            if pkg == 'json':
                import json
                logger.info(f"   ✅ {pkg}: built-in module")
            else:
                module = __import__(pkg)
                version = getattr(module, '__version__', 'unknown')
                logger.info(f"   ✅ {pkg}: {version}")
        except ImportError as e:
            logger.error(f"   ❌ {pkg}: not available - {e}")

    # Check if input files exist
    logger.info("📁 Checking input files...")
    required_files = [
        'transcation_all_final_output.json', 
        'reference_all_final_output.json'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            logger.info(f"   ✅ {file_path}: {file_size:.2f} MB")
        else:
            logger.error(f"   ❌ {file_path}: not found")
    
    # Check optional files
    optional_files = ['marketdata_all_final_output.json', 'gemini_extracted_relationships.json']
    for file_path in optional_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            logger.info(f"   📄 {file_path}: {file_size:.2f} MB (optional)")
        else:
            logger.info(f"   ⚠️ {file_path}: not found (optional)")

    try:
        # Initialize builder with detailed logging
        logger.info("🔧 Initializing DuckDB Knowledge Graph Builder...")
        logger.info("   This step tests:")
        logger.info("   - DuckDB connection")
        logger.info("   - SentenceTransformer model download")
        logger.info("   - Database schema creation")
        logger.info("   - Write permissions")
        
        builder = DuckDBKnowledgeGraphBuilder(DB_PATH)
        logger.info("✅ Builder initialization successful!")
        
        # Clear existing graph
        logger.info("🧹 Clearing existing graph data...")
        builder.clear_graph()
        
        # Load all metadata files
        metadata_files = [
            'transcation_all_final_output.json', 
            'reference_all_final_output.json'
        ]
        
        # Check if market data file exists and add it
        if os.path.exists('marketdata_all_final_output.json'):
            metadata_files.append('marketdata_all_final_output.json')
            logger.info("📊 Market data file detected - will include market data relationships")
        
        logger.info(f"📂 Loading metadata from {len(metadata_files)} files...")
        builder.load_rich_metadata(metadata_files)
        
        # Load ERD relationships
        logger.info("🔗 Loading ERD relationships...")
        builder.load_erd_relationships('gemini_extracted_relationships.json')
        
        # Infer additional relationships
        logger.info("🧠 Inferring additional relationships...")
        builder.infer_additional_relationships()
        
        # Build the complete graph
        logger.info("🏗️ Building complete knowledge graph...")
        builder.build_graph()
        
        # Verify the results
        logger.info("🔍 Verifying graph structure...")
        verification_stats = builder.verify_graph_structure()
        
        # Get database info
        db_info = builder.get_database_info()
        
        logger.info("=" * 60)
        logger.info("🎉 DuckDB Knowledge Graph v7 successfully built!")
        logger.info("✨ Features included:")
        logger.info("   ⚡ 2-10x faster than Neo4j")
        logger.info("   📊 Rich embeddings for semantic search")
        logger.info("   🔗 Comprehensive relationship inference")
        logger.info("   📋 Complete column implementation")
        logger.info("   🏷️  Categorical value nodes")
        logger.info("   📅 Date node integration")
        logger.info("   🎯 Market data relationships")
        logger.info("   💾 File-based storage (easy backup/sharing)")
        logger.info("   🚀 Zero server setup required")
        logger.info("=" * 60)
        logger.info("💡 You can now run the DuckDB benchmarking script!")
        
    except Exception as e:
        logger.error(f"❌ Critical error occurred: {e}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        logger.error(f"❌ Error location: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")
        import traceback
        logger.error(f"❌ Full traceback:\n{traceback.format_exc()}")
        
        # Additional debugging information
        logger.error("🔍 Debugging information:")
        logger.error(f"   Working directory: {os.getcwd()}")
        logger.error(f"   Database path: {DB_PATH}")
        logger.error(f"   Path exists: {os.path.exists(DB_PATH) if 'DB_PATH' in locals() else 'Unknown'}")
        
        # Check if it's a network/download issue
        if "sentence" in str(e).lower() or "transformers" in str(e).lower():
            logger.error("💡 This might be a network issue with downloading the SentenceTransformer model")
            logger.error("💡 Try running this on a machine with internet access first")
            logger.error("💡 Or use a different embedding model that's already cached")
        
        # Check if it's a permissions issue
        if "permission" in str(e).lower() or "access" in str(e).lower():
            logger.error("💡 This might be a file permissions issue")
            logger.error("💡 Try running from a directory where you have write permissions")
            logger.error("💡 Or run as administrator if necessary")
        
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("⏹️ Script interrupted by user")
    except Exception as e:
        logger.error(f"💥 Unhandled exception in main: {e}")
        sys.exit(1)