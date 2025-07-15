def load_erd_relationships(self, erd_file: str):
    """Load manually curated relationships from ERD analysis with robust error handling."""
    logger.info(f"🔗 Loading ERD relationships from {erd_file}...")
    try:
        # Try different encodings for robustness
        encodings = ['utf-8', 'utf-8-sig', 'latin1']
        erd_rels = None
        
        for encoding in encodings:
            try:
                with open(erd_file, 'r', encoding=encoding) as f:
                    content = f.read()
                    # Clean any potential BOM or extra characters
                    content = content.strip()
                    if content.startswith('\ufeff'):  # Remove BOM if present
                        content = content[1:]
                    erd_rels = json.loads(content)
                logger.info(f"✅ Loaded {erd_file} with {encoding} encoding")
                break
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load with {encoding}: {e}")
                continue
        
        if erd_rels is None:
            raise Exception(f"Could not parse {erd_file} with any supported encoding")
        
        # Validate the structure
        if not isinstance(erd_rels, list):
            raise Exception(f"Expected list in {erd_file}, got {type(erd_rels)}")
        
        for i, rel in enumerate(erd_rels):
            if not isinstance(rel, dict):
                logger.warning(f"Skipping invalid relationship at index {i}: not a dictionary")
                continue
            
            # Check required fields
            required_fields = ['from_table', 'to_table', 'from_column', 'to_column']
            if not all(field in rel for field in required_fields):
                logger.warning(f"Skipping incomplete relationship at index {i}: missing required fields")
                continue
            
            # Use tuple key for proper deduplication
            key = tuple(sorted((rel['from_table'], rel['to_table']))) + (rel['from_column'], rel['to_column'])
            self.relationships[key] = rel
            
        logger.info(f"✅ Loaded {len(self.relationships)} ERD-defined relationships")
        
    except FileNotFoundError:
        logger.warning(f"'{erd_file}' not found. No manual joins will be added")
        self.relationships = {}
    except Exception as e:
        logger.error(f"Error loading ERD relationships: {e}")
        logger.warning("Continuing without ERD relationships...")
        self.relationships = {}