def _insert_columns_and_values(self):
    """Insert column nodes and their categorical values."""
    logger.info("📋 Inserting column nodes and values...")
    
    column_insert_query = f"""
    INSERT INTO columns (id, name, full_name, table_name, description, data_type, 
                       is_nullable, distinct_values, embedding, column_category, embedding_provider)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    value_insert_query = f"""
    INSERT INTO values (id, name, column_full_name, table_name, embedding, value_type, embedding_provider)
    VALUES (?, ?, ?, ?, ?, ?, ?)
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
            embedding_vector = self._generate_embedding(embedding_text)
            
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
                embedding_vector,  # embedding
                column_category,  # column_category
                self.embedding_provider.value  # embedding_provider
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
                        value_embedding = self._generate_embedding(value_text)
                        
                        # Determine value type
                        value_type = self._classify_value_type(value)
                        
                        # Insert value
                        self.conn.execute(value_insert_query, [
                            str(uuid.uuid4()),  # id
                            str(value),  # name
                            full_col_name,  # column_full_name
                            table_name,  # table_name
                            value_embedding,  # embedding
                            value_type,  # value_type
                            self.embedding_provider.value  # embedding_provider
                        ])
                        
                        value_count += 1
    
    logger.info(f"✅ Inserted {column_count} columns and {value_count} categorical values")