Find this code in the _parse_log_file method (around line 340-350):
pythondef _parse_log_file(self, log_file: str) -> List[Dict[str, Any]]:
    """Parse log file and extract structured data"""
    log_data = []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
Replace with:
pythondef _parse_log_file(self, log_file: str) -> List[Dict[str, Any]]:
    """Parse log file and extract structured data"""
    log_data = []
    
    try:
        # Try different encodings to handle binary data
        content = None
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'utf-8-sig']:
            try:
                with open(log_file, 'r', encoding=encoding) as f:
                    content = f.read()
                self.logger.debug(f"File read successfully with {encoding} encoding")
                break
            except UnicodeDecodeError:
                self.logger.debug(f"Failed to read with {encoding}, trying next...")
                continue
        
        if not content:
            self.logger.error("Could not decode file with any encoding")
            return []
        
        # Clean any problematic characters
        import re
        content = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', ' ', content)
        
        lines = content.strip().split('\n')
🔧 Fix 2: Add better error handling in the parsing loop
Find this section in the same method (after the lines processing):
python        for i, line in enumerate(lines):
            if i == 0:  # Skip header
                continue
            
            line = line.strip()
            if not line:
                continue
            
            # Split CSV line
            parts = self._split_csv_line(line)
            
            if len(parts) >= 3:
                log_entry = {
                    'dwh_business_date': parts[0].strip(),
                    'client_id': parts[1].strip(),
                    'query_text': parts[2].strip()
                }
                
                if log_entry['query_text']:
                    log_data.append(log_entry)
Replace with:
python        # Process data lines
        processed_count = 0
        for i, line in enumerate(lines):
            if i == 0:  # Skip header
                continue
            
            line = line.strip()
            if not line:
                continue
            
            try:
                # Split CSV line with better error handling
                parts = self._split_csv_line(line)
                
                # Ensure we have at least 3 parts
                if len(parts) >= 3:
                    # Clean the parts to remove any problematic characters
                    log_entry = {
                        'dwh_business_date': self._clean_text(parts[0]),
                        'client_id': self._clean_text(parts[1]),
                        'query_text': self._clean_text(parts[2])
                    }
                    
                    # Only include if query_text is not empty and looks valid
                    if log_entry['query_text'] and len(log_entry['query_text']) > 5:
                        log_data.append(log_entry)
                        processed_count += 1
                        
                        # Debug first few entries
                        if processed_count <= 3:
                            self.logger.debug(f"Parsed entry {processed_count}: {log_entry['query_text'][:50]}...")
                
            except Exception as parse_error:
                self.logger.debug(f"Failed to parse line {i}: {line[:50]}... - Error: {parse_error}")
                continue
        
        self.logger.info(f"Successfully parsed {len(log_data)} valid log entries from {len(lines)} total lines")
🔧 Fix 3: Add a helper method for cleaning text
Add this new method to the QueryLogAnalyzer class:
pythondef _clean_text(self, text: str) -> str:
    """Clean text to remove problematic characters"""
    if not text:
        return ""
    
    # Remove non-printable characters except newlines and tabs
    import re
    cleaned = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', ' ', str(text))
    
    # Replace multiple spaces with single space
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()
🔧 Fix 4: Update the _split_csv_line method for better handling
Find the _split_csv_line method and add better error handling:
pythondef _split_csv_line(self, line: str) -> List[str]:
    """Split CSV line handling quoted fields and problematic characters"""
    parts = []
    current_part = ""
    in_quotes = False
    
    try:
        for char in line:
            if char == '"':
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                parts.append(current_part.strip())
                current_part = ""
            else:
                # Only add printable characters
                if ord(char) >= 32 or char in ['\t', '\n']:
                    current_part += char
        
        if current_part:
            parts.append(current_part.strip())
        
        return parts
        
    except Exception as e:
        self.logger.debug(f"Error splitting CSV line: {e}")
        # Fallback: simple split by comma
        return line.split(',')
📝 Summary of Changes: