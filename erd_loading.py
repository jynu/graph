def load_erd_relationships(self, erd_file: str):
    """Load manually curated relationships from ERD analysis with robust error handling."""
    logger.info(f"🔗 Loading ERD relationships from {erd_file}...")
    try:
        # Simplified approach - try UTF-8 first, then fallback
        erd_rels = None
        
        try:
            # Try UTF-8 with explicit error handling
            with open(erd_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # Remove any BOM characters
                if content.startswith('\ufeff'):
                    content = content[1:]
                erd_rels = json.loads(content)
            logger.info(f"✅ Loaded {erd_file} with utf-8 encoding")
            
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.warning(f"UTF-8 failed: {e}, trying alternative approach...")
            
            # Alternative approach: read as binary and decode
            try:
                with open(erd_file, 'rb') as f:
                    raw_content = f.read()
                    # Try to decode with different encodings
                    for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
                        try:
                            content = raw_content.decode(encoding)
                            # Clean the content
                            content = content.strip()
                            if content.startswith('\ufeff'):
                                content = content[1:]
                            erd_rels = json.loads(content)
                            logger.info(f"✅ Loaded {erd_file} with {encoding} encoding")
                            break
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue
                            
            except Exception as e2:
                logger.error(f"Binary read approach failed: {e2}")
        
        if erd_rels is None:
            raise Exception(f"Could not parse {erd_file} with any supported method")
        
        # Validate the structure
        if not isinstance(erd_rels, list):
            raise Exception(f"Expected list in {erd_file}, got {type(erd_rels)}")
        
        # Process relationships
        relationships_added = 0
        for i, rel in enumerate(erd_rels):
            try:
                if not isinstance(rel, dict):
                    logger.warning(f"Skipping invalid relationship at index {i}: not a dictionary")
                    continue
                
                # Check required fields
                required_fields = ['from_table', 'to_table', 'from_column', 'to_column']
                missing_fields = [field for field in required_fields if field not in rel]
                
                if missing_fields:
                    logger.warning(f"Skipping incomplete relationship at index {i}: missing {missing_fields}")
                    continue
                
                # Clean and validate field values
                from_table = str(rel['from_table']).strip()
                to_table = str(rel['to_table']).strip()
                from_column = str(rel['from_column']).strip()
                to_column = str(rel['to_column']).strip()
                
                if not all([from_table, to_table, from_column, to_column]):
                    logger.warning(f"Skipping relationship at index {i}: empty field values")
                    continue
                
                # Create cleaned relationship
                cleaned_rel = {
                    'from_table': from_table,
                    'to_table': to_table,
                    'from_column': from_column,
                    'to_column': to_column,
                    'relationship_type': rel.get('relationship_type', 'ERD_DEFINED'),
                    'confidence': float(rel.get('confidence', 1.0))
                }
                
                # Use tuple key for proper deduplication
                key = tuple(sorted((from_table, to_table))) + (from_column, to_column)
                self.relationships[key] = cleaned_rel
                relationships_added += 1
                
            except Exception as rel_error:
                logger.warning(f"Error processing relationship at index {i}: {rel_error}")
                continue
                
        logger.info(f"✅ Successfully loaded {relationships_added} ERD-defined relationships")
        
        # Log a sample for verification
        if relationships_added > 0:
            sample_rel = next(iter(self.relationships.values()))
            logger.info(f"📋 Sample relationship: {sample_rel['from_table']}.{sample_rel['from_column']} -> {sample_rel['to_table']}.{sample_rel['to_column']}")
            
    except FileNotFoundError:
        logger.warning(f"'{erd_file}' not found. No manual joins will be added")
        self.relationships = {}
    except Exception as e:
        logger.error(f"Error loading ERD relationships: {e}")
        logger.warning("Continuing without ERD relationships...")
        self.relationships = {}
        
        # Log detailed error information for debugging
        import traceback
        logger.debug(f"Detailed error trace: {traceback.format_exc()}")