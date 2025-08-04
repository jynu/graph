import json
import logging
import os
import glob
from typing import Dict, List, Any, Set, Tuple, Optional
import duckdb
import uuid
from datetime import datetime
from enum import Enum
import asyncio

# Import your existing components
from app.utils.client_manager import client_manager
from app.rag.embedding import embedding

# Import the base KG builder functionality
from kg_builder_v2 import DuckDBKnowledgeGraphBuilder, EmbeddingProvider

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedDuckDBKnowledgeGraphBuilder(DuckDBKnowledgeGraphBuilder):
    """
    Enhanced DuckDB Knowledge Graph Builder
    Supports both internal dataset format and processed BIRD dataset format
    
    Features:
    - Automatic data source detection
    - Unified schema handling
    - Enhanced relationship inference for BIRD data
    - Cross-dataset relationship discovery
    """
    
    def __init__(self, db_path: str = 'enhanced_knowledge_graph.duckdb', 
                 embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI,
                 embedding_dimensions: int = 1536):
        
        super().__init__(db_path, embedding_provider, embedding_dimensions)
        
        # Additional tracking for enhanced features
        self.data_sources = {}  # Track which tables come from which sources
        self.cross_dataset_relationships = {}
        self.bird_specific_patterns = {}
        
        logger.info(f"🚀 Enhanced KG Builder initialized with BIRD support")
    
    def load_mixed_metadata(self, internal_files: List[str], bird_files: List[str] = None):
        """
        Load metadata from both internal format and processed BIRD format files.
        
        Args:
            internal_files: Your existing internal JSON files
            bird_files: Processed BIRD JSON files (auto-detected if None)
        """
        logger.info(f"📂 Loading mixed metadata sources...")
        
        # Load internal files (existing functionality)
        if internal_files:
            logger.info(f"📋 Loading {len(internal_files)} internal files...")
            self.load_rich_metadata(internal_files)
            
            # Track source
            for table_name in self.tables_metadata.keys():
                self.data_sources[table_name] = "internal"
        
        # Auto-detect BIRD files if not specified
        if bird_files is None:
            bird_files = glob.glob("processed_bird_data/bird_processed_*.json")
            logger.info(f"🔍 Auto-detected {len(bird_files)} processed BIRD files")
        
        # Load BIRD files
        if bird_files:
            logger.info(f"🐦 Loading {len(bird_files)} BIRD files...")
            bird_tables_count = self._load_bird_metadata(bird_files)
            logger.info(f"✅ Loaded {bird_tables_count} BIRD tables")
        
        total_tables = len(self.tables_metadata)
        internal_count = sum(1 for source in self.data_sources.values() if source == "internal")
        bird_count = sum(1 for source in self.data_sources.values() if source.startswith("bird"))
        
        logger.info(f"📊 Total metadata loaded:")
        logger.info(f"   📋 Internal tables: {internal_count}")
        logger.info(f"   🐦 BIRD tables: {bird_count}")
        logger.info(f"   📈 Total tables: {total_tables}")
    
    def _load_bird_metadata(self, bird_files: List[str]) -> int:
        """Load processed BIRD metadata files."""
        
        bird_tables_loaded = 0
        
        for file_path in bird_files:
            try:
                # Try different encodings for robustness
                encodings = ['utf-8', 'latin1', 'utf-8-sig']
                data = None
                
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            data = json.load(f)
                        logger.debug(f"✅ Loaded {file_path} with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if data is None:
                    raise Exception(f"Could not decode {file_path} with any supported encoding")
                
                # Extract source info
                original_file = data.get('original_file', file_path)
                source_id = f"bird_{os.path.basename(original_file).replace('.json', '')}"
                
                # Process training data
                training_data = data.get('trainingdata', [])
                
                for table_data in training_data:
                    table_name = table_data.get('tablename')
                    if table_name:
                        # Add BIRD-specific metadata
                        enhanced_table_data = self._enhance_bird_table_data(table_data, source_id)
                        
                        self.tables_metadata[table_name] = enhanced_table_data
                        self.data_sources[table_name] = source_id
                        bird_tables_loaded += 1
                        
                        # Track BIRD-specific patterns
                        self._analyze_bird_patterns(enhanced_table_data)
                
                logger.info(f"✅ Loaded {len(training_data)} tables from {file_path}")
                
            except Exception as e:
        logger.error(f"❌ Enhanced knowledge graph build failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
                logger.error(f"❌ Failed to load BIRD file {file_path}: {e}")
                continue
        
        return bird_tables_loaded
    
    def _enhance_bird_table_data(self, table_data: Dict, source_id: str) -> Dict:
        """Enhance BIRD table data with additional metadata for better KG integration."""
        
        enhanced_data = table_data.copy()
        
        # Add source tracking
        enhanced_data['data_source'] = source_id
        enhanced_data['source_type'] = 'bird_dataset'
        
        # Enhance table classification
        table_name = table_data.get('original_name', '').lower()
        database_id = table_data.get('database_id', '')
        
        # Classify table type based on BIRD patterns
        enhanced_data['table_type'] = self._classify_bird_table_type(table_name, database_id)
        
        # Enhance column data
        enhanced_columns = []
        for col_data in table_data.get('columns', []):
            enhanced_col = self._enhance_bird_column_data(col_data, table_name, database_id)
            enhanced_columns.append(enhanced_col)
        
        enhanced_data['columns'] = enhanced_columns
        
        # Add domain-specific rules based on database type
        enhanced_data['domain_rules'] = self._infer_domain_rules(database_id, table_name)
        
        return enhanced_data
    
    def _classify_bird_table_type(self, table_name: str, database_id: str) -> str:
        """Classify BIRD table types based on naming patterns and domain."""
        
        # Domain-specific classification
        if 'financial' in database_id.lower():
            if any(term in table_name for term in ['account', 'client', 'card', 'loan']):
                return 'dimension'
            elif any(term in table_name for term in ['trans', 'transaction', 'order']):
                return 'fact'
            elif any(term in table_name for term in ['district', 'disp']):
                return 'reference'
        
        elif 'debit_card' in database_id.lower():
            if any(term in table_name for term in ['customer', 'gasstation', 'product']):
                return 'dimension'
            elif any(term in table_name for term in ['transaction', 'yearmonth']):
                return 'fact'
        
        elif 'european_football' in database_id.lower():
            if 'division' in table_name:
                return 'reference'
            elif 'match' in table_name:
                return 'fact'
        
        elif 'sales' in database_id.lower() and 'weather' in database_id.lower():
            if 'weather' in table_name:
                return 'dimension'
            elif 'sales' in table_name:
                return 'fact'
            elif 'relation' in table_name:
                return 'reference'
        
        # Generic classification
        if any(term in table_name for term in ['dim', 'dimension', 'master', 'ref']):
            return 'dimension'
        elif any(term in table_name for term in ['fact', 'trans', 'event', 'log']):
            return 'fact'
        else:
            return 'reference'
    
    def _enhance_bird_column_data(self, col_data: Dict, table_name: str, database_id: str) -> Dict:
        """Enhance BIRD column data with domain-specific insights."""
        
        enhanced_col = col_data.copy()
        col_name = col_data.get('columnname', '').lower()
        
        # Enhance categorical detection for BIRD data
        is_categorical = self._is_bird_categorical_column(col_name, col_data, database_id)
        if is_categorical:
            enhanced_col['provide_distinct'] = 'YES'
            
            # Generate sample values if not present
            if not enhanced_col.get('distinct_values'):
                enhanced_col['distinct_values'] = self._generate_sample_values(col_name, col_data, database_id)
        
        # Add domain-specific column classification
        enhanced_col['column_domain'] = self._classify_column_domain(col_name, database_id)
        
        # Enhance business importance scoring
        enhanced_col['business_importance'] = self._score_column_importance(col_name, col_data, table_name)
        
        return enhanced_col
    
    def _is_bird_categorical_column(self, col_name: str, col_data: Dict, database_id: str) -> bool:
        """Determine if a BIRD column should be treated as categorical."""
        
        data_type = col_data.get('datatype', '').lower()
        
        # Type-based detection
        if data_type == 'text' and any(term in col_name for term in [
            'type', 'status', 'category', 'code', 'flag', 'gender', 'symbol'
        ]):
            return True
        
        # Domain-specific detection
        if 'financial' in database_id:
            if col_name in ['type', 'frequency', 'operation', 'k_symbol', 'status']:
                return True
        
        elif 'football' in database_id:
            if col_name in ['ftr', 'div', 'country']:
                return True
        
        elif 'debit_card' in database_id:
            if col_name in ['segment', 'currency', 'country']:
                return True
        
        # Primary/Foreign key detection
        if col_data.get('is_primary_key') or col_data.get('is_foreign_key'):
            return False  # IDs shouldn't be categorical for distinct values
        
        return False
    
    def _generate_sample_values(self, col_name: str, col_data: Dict, database_id: str) -> List[str]:
        """Generate realistic sample values for BIRD columns."""
        
        # Domain-specific sample generation
        if 'financial' in database_id:
            if col_name == 'type':
                return ['OWNER', 'DISPONENT', 'AUTHORIZED']
            elif col_name == 'frequency':
                return ['POPLATEK MESICNE', 'POPLATEK TYDNE', 'POPLATEK PO OBRATU']
            elif col_name == 'operation':
                return ['VYBER KARTOU', 'VKLAD', 'PREVOD Z UCTU', 'VYBER']
            elif col_name == 'status':
                return ['A', 'B', 'C', 'D']
        
        elif 'football' in database_id:
            if col_name == 'ftr':
                return ['H', 'A', 'D']  # Home, Away, Draw
            elif col_name == 'country':
                return ['England', 'Spain', 'Germany', 'Italy', 'France']
        
        elif 'debit_card' in database_id:
            if col_name == 'segment':
                return ['Premium', 'Standard', 'Basic']
            elif col_name == 'currency':
                return ['USD', 'EUR', 'GBP', 'JPY']
        
        # Generic samples based on column name patterns
        if 'gender' in col_name:
            return ['M', 'F']
        elif 'status' in col_name:
            return ['Active', 'Inactive', 'Pending']
        elif 'type' in col_name:
            return ['Type A', 'Type B', 'Type C']
        
        return []
    
    def _classify_column_domain(self, col_name: str, database_id: str) -> str:
        """Classify column domain for better semantic understanding."""
        
        if any(term in col_name for term in ['id', 'key']):
            return 'identifier'
        elif any(term in col_name for term in ['date', 'time']):
            return 'temporal'
        elif any(term in col_name for term in ['amount', 'price', 'value', 'balance']):
            return 'monetary'
        elif any(term in col_name for term in ['name', 'description', 'text']):
            return 'textual'
        elif any(term in col_name for term in ['count', 'number', 'quantity']):
            return 'numerical'
        elif any(term in col_name for term in ['type', 'status', 'category', 'flag']):
            return 'categorical'
        else:
            return 'general'
    
    def _score_column_importance(self, col_name: str, col_data: Dict, table_name: str) -> float:
        """Score column business importance (0.0 to 1.0)."""
        
        score = 0.5  # Base score
        
        # Primary key gets high importance
        if col_data.get('is_primary_key'):
            score += 0.3
        
        # Foreign key gets medium importance
        if col_data.get('is_foreign_key'):
            score += 0.2
        
        # Business-critical column names
        if any(term in col_name for term in ['amount', 'price', 'value', 'balance']):
            score += 0.2
        
        if any(term in col_name for term in ['date', 'time', 'created', 'updated']):
            score += 0.15
        
        if any(term in col_name for term in ['name', 'title', 'description']):
            score += 0.1
        
        # Table-specific importance
        if 'transaction' in table_name and col_name in ['amount', 'date', 'type']:
            score += 0.1
        
        return min(1.0, score)
    
    def _infer_domain_rules(self, database_id: str, table_name: str) -> str:
        """Infer domain-specific business rules."""
        
        rules = []
        
        if 'financial' in database_id:
            if 'account' in table_name:
                rules.append("Each account must have a valid district")
                rules.append("Account creation date must be valid")
            elif 'transaction' in table_name:
                rules.append("Transaction amount must be non-zero")
                rules.append("Transaction must reference valid account")
            elif 'loan' in table_name:
                rules.append("Loan amount must be positive")
                rules.append("Loan duration must be reasonable (1-360 months)")
        
        elif 'debit_card' in database_id:
            if 'transaction' in table_name:
                rules.append("Transaction must have valid customer and product")
                rules.append("Price should be positive for purchases")
        
        elif 'football' in database_id:
            if 'match' in table_name:
                rules.append("Home and away teams must be different")
                rules.append("Goals must be non-negative integers")
        
        return "; ".join(rules)
    
    def _analyze_bird_patterns(self, table_data: Dict):
        """Analyze and store BIRD-specific patterns for relationship inference."""
        
        database_id = table_data.get('database_id', '')
        table_name = table_data.get('original_name', '')
        
        if database_id not in self.bird_specific_patterns:
            self.bird_specific_patterns[database_id] = {
                'tables': [],
                'common_columns': set(),
                'key_patterns': []
            }
        
        pattern_data = self.bird_specific_patterns[database_id]
        pattern_data['tables'].append(table_name)
        
        # Analyze column patterns
        for col_data in table_data.get('columns', []):
            col_name = col_data.get('columnname', '')
            pattern_data['common_columns'].add(col_name)
            
            # Track key patterns
            if col_data.get('is_primary_key'):
                pattern_data['key_patterns'].append(f"PK:{col_name}")
            if col_data.get('is_foreign_key'):
                pattern_data['key_patterns'].append(f"FK:{col_name}")
    
    def infer_enhanced_relationships(self):
        """Enhanced relationship inference for mixed internal + BIRD data."""
        
        logger.info("🧠 Running enhanced relationship inference...")
        
        # Run base relationship inference
        self.infer_additional_relationships()
        
        # Add BIRD-specific relationship patterns
        self._infer_bird_relationships()
        
        # Add cross-dataset relationships
        self._infer_cross_dataset_relationships()
        
        logger.info(f"✅ Enhanced relationship inference completed")
    
    def _infer_bird_relationships(self):
        """Infer relationships specific to BIRD datasets."""
        
        logger.info("🐦 Inferring BIRD-specific relationships...")
        
        discovered_count = 0
        
        # Process each BIRD database
        for database_id, patterns in self.bird_specific_patterns.items():
            logger.debug(f"Processing BIRD database: {database_id}")
            
            # Get tables from this database
            db_tables = [name for name, source in self.data_sources.items() 
                        if source.startswith('bird') and database_id in source]
            
            # Apply database-specific relationship rules
            if 'financial' in database_id:
                discovered_count += self._infer_financial_relationships(db_tables)
            elif 'debit_card' in database_id:
                discovered_count += self._infer_debit_card_relationships(db_tables)
            elif 'football' in database_id:
                discovered_count += self._infer_football_relationships(db_tables)
            elif 'sales' in database_id and 'weather' in database_id:
                discovered_count += self._infer_sales_weather_relationships(db_tables)
        
        logger.info(f"✅ Discovered {discovered_count} BIRD-specific relationships")
    
    def _infer_financial_relationships(self, tables: List[str]) -> int:
        """Infer relationships for financial domain BIRD data."""
        
        count = 0
        
        # Find relevant tables
        account_table = next((t for t in tables if 'account' in t.lower()), None)
        client_table = next((t for t in tables if 'client' in t.lower()), None)
        card_table = next((t for t in tables if 'card' in t.lower()), None)
        trans_table = next((t for t in tables if 'trans' in t.lower()), None)
        loan_table = next((t for t in tables if 'loan' in t.lower()), None)
        district_table = next((t for t in tables if 'district' in t.lower()), None)
        disp_table = next((t for t in tables if 'disp' in t.lower()), None)
        
        # Account -> District relationship
        if account_table and district_table:
            self._add_relationship(account_table, district_table, 'district_id', 'district_id', 'DISTRICT_REFERENCE', 0.95)
            count += 1
        
        # Client -> District relationship
        if client_table and district_table:
            self._add_relationship(client_table, district_table, 'district_id', 'district_id', 'CLIENT_DISTRICT', 0.95)
            count += 1
        
        # Transaction -> Account relationship
        if trans_table and account_table:
            self._add_relationship(trans_table, account_table, 'account_id', 'account_id', 'ACCOUNT_TRANSACTION', 0.95)
            count += 1
        
        # Loan -> Account relationship
        if loan_table and account_table:
            self._add_relationship(loan_table, account_table, 'account_id', 'account_id', 'ACCOUNT_LOAN', 0.95)
            count += 1
        
        # Card -> Disposition relationship
        if card_table and disp_table:
            self._add_relationship(card_table, disp_table, 'disp_id', 'disp_id', 'CARD_DISPOSITION', 0.95)
            count += 1
        
        # Disposition relationships
        if disp_table:
            if client_table:
                self._add_relationship(disp_table, client_table, 'client_id', 'client_id', 'DISPOSITION_CLIENT', 0.95)
                count += 1
            if account_table:
                self._add_relationship(disp_table, account_table, 'account_id', 'account_id', 'DISPOSITION_ACCOUNT', 0.95)
                count += 1
        
        return count
    
    def _infer_debit_card_relationships(self, tables: List[str]) -> int:
        """Infer relationships for debit card domain BIRD data."""
        
        count = 0
        
        # Find relevant tables
        customer_table = next((t for t in tables if 'customer' in t.lower()), None)
        gasstation_table = next((t for t in tables if 'gasstation' in t.lower() or 'gas' in t.lower()), None)
        product_table = next((t for t in tables if 'product' in t.lower()), None)
        transaction_table = next((t for t in tables if 'transaction' in t.lower()), None)
        yearmonth_table = next((t for t in tables if 'yearmonth' in t.lower() or 'year' in t.lower()), None)
        
        # Transaction relationships
        if transaction_table:
            if customer_table:
                self._add_relationship(transaction_table, customer_table, 'CustomerID', 'CustomerID', 'TRANSACTION_CUSTOMER', 0.95)
                count += 1
            if gasstation_table:
                self._add_relationship(transaction_table, gasstation_table, 'GasStationID', 'GasStationID', 'TRANSACTION_GASSTATION', 0.95)
                count += 1
            if product_table:
                self._add_relationship(transaction_table, product_table, 'ProductID', 'ProductID', 'TRANSACTION_PRODUCT', 0.95)
                count += 1
        
        # YearMonth -> Customer relationship
        if yearmonth_table and customer_table:
            self._add_relationship(yearmonth_table, customer_table, 'CustomerID', 'CustomerID', 'CONSUMPTION_CUSTOMER', 0.90)
            count += 1
        
        return count
    
    def _infer_football_relationships(self, tables: List[str]) -> int:
        """Infer relationships for football domain BIRD data."""
        
        count = 0
        
        # Find relevant tables
        divisions_table = next((t for t in tables if 'division' in t.lower()), None)
        matches_table = next((t for t in tables if 'match' in t.lower()), None)
        
        # Match -> Division relationship
        if matches_table and divisions_table:
            self._add_relationship(matches_table, divisions_table, 'Div', 'division', 'MATCH_DIVISION', 0.95)
            count += 1
        
        return count
    
    def _infer_sales_weather_relationships(self, tables: List[str]) -> int:
        """Infer relationships for sales-weather domain BIRD data."""
        
        count = 0
        
        # Find relevant tables
        sales_table = next((t for t in tables if 'sales' in t.lower()), None)
        weather_table = next((t for t in tables if 'weather' in t.lower()), None)
        relation_table = next((t for t in tables if 'relation' in t.lower()), None)
        
        # Sales -> Weather via Relation
        if relation_table:
            if sales_table:
                self._add_relationship(sales_table, relation_table, 'store_nbr', 'store_nbr', 'SALES_STORE_RELATION', 0.90)
                count += 1
            if weather_table:
                self._add_relationship(weather_table, relation_table, 'station_nbr', 'station_nbr', 'WEATHER_STATION_RELATION', 0.90)
                count += 1
        
        return count
    
    def _infer_cross_dataset_relationships(self):
        """Infer relationships between internal data and BIRD data."""
        
        logger.info("🔗 Inferring cross-dataset relationships...")
        
        discovered_count = 0
        
        # Get internal and BIRD tables
        internal_tables = [name for name, source in self.data_sources.items() if source == "internal"]
        bird_tables = [name for name, source in self.data_sources.items() if source.startswith("bird")]
        
        if not internal_tables or not bird_tables:
            logger.info("⚠️ Skipping cross-dataset relationships (missing data types)")
            return
        
        # Look for semantic similarities between internal and BIRD data
        for internal_table in internal_tables:
            for bird_table in bird_tables:
                
                similarity_score = self._calculate_cross_dataset_similarity(internal_table, bird_table)
                
                if similarity_score > 0.6:  # Threshold for cross-dataset relationships  
                    rel_type = "CROSS_DATASET_SIMILARITY"
                    self._add_relationship(internal_table, bird_table, "semantic_similarity", "semantic_similarity", rel_type, similarity_score)
                    discovered_count += 1
                    
                    logger.debug(f"Cross-dataset relationship: {internal_table} <-> {bird_table} (score: {similarity_score:.2f})")
        
        logger.info(f"✅ Discovered {discovered_count} cross-dataset relationships")
    
    def _calculate_cross_dataset_similarity(self, internal_table: str, bird_table: str) -> float:
        """Calculate semantic similarity between internal and BIRD tables."""
        
        try:
            internal_data = self.tables_metadata.get(internal_table, {})
            bird_data = self.tables_metadata.get(bird_table, {})
            
            if not internal_data or not bird_data:
                return 0.0
            
            # Compare table descriptions
            internal_desc = (internal_data.get('tableDescription', '') + ' ' + 
                           ' '.join(internal_data.get('tableAlias', []))).lower()
            bird_desc = (bird_data.get('tableDescription', '') + ' ' + 
                        bird_data.get('original_name', '') + ' ' +
                        bird_data.get('cleaned_name', '')).lower()
            
            if not internal_desc.strip() or not bird_desc.strip():
                return 0.0
            
            # Simple word overlap similarity
            internal_words = set(internal_desc.split())
            bird_words = set(bird_desc.split())
            
            if not internal_words or not bird_words:
                return 0.0
            
            intersection = internal_words.intersection(bird_words)
            union = internal_words.union(bird_words)
            
            jaccard_similarity = len(intersection) / len(union) if union else 0.0
            
            # Boost similarity for domain matches
            domain_keywords = ['financial', 'transaction', 'account', 'customer', 'product', 'trade', 'market']
            
            for keyword in domain_keywords:
                if keyword in internal_desc and keyword in bird_desc:
                    jaccard_similarity += 0.1  # Boost for domain match
            
            return min(1.0, jaccard_similarity)
            
        except Exception as e:
            logger.debug(f"Error calculating cross-dataset similarity: {e}")
            return 0.0
    
    def verify_enhanced_graph_structure(self):
        """Enhanced graph structure verification with source breakdown."""
        
        logger.info("🔍 Verifying enhanced knowledge graph structure...")
        
        # Run base verification
        base_stats = super().verify_graph_structure()
        
        # Add enhanced statistics
        enhanced_stats = base_stats.copy()
        
        # Source breakdown
        source_breakdown = {}
        for table_name, source in self.data_sources.items():
            if source not in source_breakdown:
                source_breakdown[source] = 0
            source_breakdown[source] += 1
        
        logger.info("📊 Data source breakdown:")
        for source, count in source_breakdown.items():
            source_type = "🏢 Internal" if source == "internal" else f"🐦 BIRD ({source})"
            logger.info(f"   {source_type}: {count} tables")
        
        # BIRD domain analysis
        if self.bird_specific_patterns:
            logger.info("🐦 BIRD domain analysis:")
            for database_id, patterns in self.bird_specific_patterns.items():
                table_count = len(patterns['tables'])
                common_cols = len(patterns['common_columns'])
                logger.info(f"   {database_id}: {table_count} tables, {common_cols} unique columns")
        
        # Relationship type breakdown
        try:
            rel_types = self.conn.execute("""
                SELECT relationship_type, COUNT(*) as count 
                FROM relationships 
                GROUP BY relationship_type 
                ORDER BY count DESC
            """).fetchall()
            
            logger.info("🔗 Enhanced relationship type distribution:")
            for rel_type, count in rel_types:
                if 'BIRD' in rel_type or 'CROSS_DATASET' in rel_type:
                    prefix = "🐦"
                elif any(term in rel_type for term in ['SMCP', 'MARKET', 'TRADE']):
                    prefix = "🏢"
                else:
                    prefix = "⚡"
                logger.info(f"   {prefix} {rel_type}: {count}")
        except:
            pass
        
        enhanced_stats.update({
            'source_breakdown': source_breakdown,
            'bird_domains': len(self.bird_specific_patterns),
            'cross_dataset_relationships': len([k for k in self.relationships.keys() 
                                              if self.relationships[k].get('relationship_type') == 'CROSS_DATASET_SIMILARITY'])
        })
        
        return enhanced_stats
    
    def export_enhanced_metadata(self, output_file: str = None):
        """Export enhanced metadata including source tracking."""
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"enhanced_kg_metadata_{timestamp}.json"
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "embedding_provider": self.embedding_provider.value,
            "embedding_dimensions": self.embedding_dimensions,
            "total_tables": len(self.tables_metadata),
            "data_sources": self.data_sources,
            "bird_patterns": self.bird_specific_patterns,
            "source_statistics": {},
            "tables_metadata": self.tables_metadata
        }
        
        # Calculate source statistics
        for source in set(self.data_sources.values()):
            source_tables = [name for name, src in self.data_sources.items() if src == source]
            export_data["source_statistics"][source] = {
                "table_count": len(source_tables),
                "tables": source_tables
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Enhanced metadata exported to: {output_file}")
        return output_file


def main():
    """Enhanced main function supporting both internal and BIRD data."""
    
    logger.info("🚀 Enhanced DuckDB Knowledge Graph Builder with BIRD Support")
    logger.info("=" * 80)
    
    # Configuration
    DB_PATH = "enhanced_knowledge_graph.duckdb"
    EMBEDDING_PROVIDER = EmbeddingProvider.OPENAI
    EMBEDDING_DIMENSIONS = 1536
    
    try:
        # Initialize enhanced builder
        builder = EnhancedDuckDBKnowledgeGraphBuilder(
            db_path=DB_PATH,
            embedding_provider=EMBEDDING_PROVIDER,
            embedding_dimensions=EMBEDDING_DIMENSIONS
        )
        
        # Clear existing graph
        builder.clear_graph()
        
        # Load mixed metadata
        logger.info("📂 Loading mixed metadata sources...")
        
        # Internal files (your existing data)
        internal_files = [
            'transcation_all_final_output.json', 
            'reference_all_final_output.json'
        ]
        
        # Add market data if available
        if os.path.exists('marketdata_all_final_output.json'):
            internal_files.append('marketdata_all_final_output.json')
        
        # BIRD files (auto-detected from processed_bird_data/)
        bird_files = None  # Auto-detect
        
        # Load mixed metadata
        builder.load_mixed_metadata(internal_files, bird_files)
        
        # Load ERD relationships (your existing relationships)
        builder.load_erd_relationships('gemini_extracted_relationships.json')
        
        # Enhanced relationship inference
        builder.infer_enhanced_relationships()
        
        # Build the enhanced knowledge graph
        builder.build_graph()
        
        # Enhanced verification
        verification_stats = builder.verify_enhanced_graph_structure()
        
        # Export enhanced metadata
        metadata_file = builder.export_enhanced_metadata()
        
        # Get database info
        db_info = builder.get_database_info()
        
        logger.info("=" * 80)
        logger.info("🎉 Enhanced Knowledge Graph Successfully Built!")
        logger.info("✨ Features:")
        logger.info("   🏢 Internal data integration")
        logger.info("   🐦 BIRD dataset integration") 
        logger.info("   🔗 Cross-dataset relationship discovery")
        logger.info("   🧠 Domain-specific relationship inference")
        logger.info(f"   🔧 {EMBEDDING_PROVIDER.value} embeddings ({EMBEDDING_DIMENSIONS}D)")
        logger.info("   📊 Enhanced similarity search")
        logger.info("   ⚡ 2-10x faster than Neo4j")
        logger.info("   💾 Zero server setup required")
        logger.info("=" * 80)
        
        logger.info("📊 Final Statistics:")
        total_tables = verification_stats.get('tables', 0)
        total_relationships = verification_stats.get('relationships', 0)
        total_columns = verification_stats.get('columns', 0)
        
        logger.info(f"   📋 Total tables: {total_tables}")
        logger.info(f"   🔗 Total relationships: {total_relationships}")
        logger.info(f"   📝 Total columns: {total_columns}")
        logger.info(f"   💾 Database size: {db_info['database_size_mb']:.2f} MB")
        logger.info(f"   📄 Metadata export: {metadata_file}")
        
        logger.info("\n💡 Next Steps:")
        logger.info("   1. Test similarity search: builder.similarity_search('your query')")
        logger.info("   2. Run benchmarks: python duckdb_benchmark_llm_v3.py")
        logger.info("   3. Explore relationships in enhanced_knowledge_graph.duckdb")
        logger.info("   4. Fine-tune cross-dataset relationship thresholds if needed")
        
    except Exception as e: