Solutions to Fix Gemini Results:
Fix 1: Improve Schema Summary (Lines ~460-490)
Make the schema more concise and focused for Gemini:
pythondef _get_enhanced_schema_summary(self) -> str:
    """Get detailed schema with business context - optimized for Gemini."""
    try:
        sql = """
        SELECT t.name as table_name,
               t.description as table_desc,
               t.table_type
        FROM tables t
        ORDER BY 
            CASE t.table_type 
                WHEN 'fact' THEN 1 
                WHEN 'dimension' THEN 2 
                WHEN 'reference' THEN 3 
                ELSE 4 
            END, t.name
        LIMIT 30
        """
        
        results = self.conn.execute(sql).fetchall()
        
        schema_sections = {"fact": [], "dimension": [], "reference": []}
        
        for table_name, table_desc, table_type in results:
            # Keep only essential info for Gemini
            table_line = f"{table_name}"
            if table_desc and len(table_desc) > 10:
                table_line += f" - {table_desc[:60]}"
            
            section = table_type if table_type in schema_sections else "fact"
            schema_sections[section].append(table_line)
        
        # Create more focused schema text
        schema_text = []
        if schema_sections["fact"]:
            schema_text.append("FACT TABLES:")
            schema_text.extend(schema_sections["fact"][:8])  # Limit to 8 tables
        
        if schema_sections["dimension"]:
            schema_text.append("\nDIMENSION TABLES:")
            schema_text.extend(schema_sections["dimension"][:8])
        
        if schema_sections["reference"]:
            schema_text.append("\nREFERENCE TABLES:")
            schema_text.extend(schema_sections["reference"][:6])
        
        return "\n".join(schema_text)
    except Exception as e:
        logger.error(f"Failed to get schema summary: {e}")
        return "Schema unavailable"
Fix 2: Improve Gemini Prompt (Lines ~510-540)
Make the prompt much more explicit about using exact table names:
pythondef get_tables(self, query: str) -> List[str]:
    logger.info(f"💎 Running Enhanced Google Gemini search for: '{query[:50]}...'")
    
    prompt = f"""You are a database expert. Your task is to select the most relevant table names from the provided schema.

**CRITICAL INSTRUCTIONS:**
- You MUST use ONLY the EXACT table names listed in the schema below
- Do NOT create new table names or use partial names
- Return ONLY actual table names that exist in the schema
- Separate multiple table names with commas
- Maximum 5 table names

**User Query:**
{query}

**Available Tables Schema:**
{self.schema_summary}

**Task:** Select the most relevant EXACT table names from the schema above that would help answer the user query.

**Response Format:** table_name1, table_name2, table_name3

**Selected Table Names:**"""
    
    try:
        import requests
        
        files = {
            'message': (None, prompt)
        }
        
        headers = {
            'accept': 'application/json'
        }
        
        response = requests.post(
            "https://olympus-dc-server-icg-isg-olympus-high-volume-api-167969.apps.namicg39024u.ecs.dyn.nsroot.net/api/v1/ask_vertexai",
            files=files,
            headers=headers,
            timeout=30,
            verify="CitiInternalCAChain_PROD.pem"
        )
        
        logger.info(f"Gemini Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            if 'response' in result:
                response_text = result['response'].strip()
                logger.info(f"Gemini Raw Response: {response_text[:200]}...")
                
                # Enhanced parsing with validation
                tables = self._parse_and_validate_tables(response_text)
                
                return tables[:10]
            else:
                logger.warning(f"Unexpected Gemini response format: {list(result.keys())}")
                return []
        else:
            logger.error(f"Gemini API returned status {response.status_code}: {response.text}")
            return []
            
    except Exception as e:
        logger.error(f"Gemini retrieval failed: {e}")
        return []
Fix 3: Add Table Validation (Add this new method)
Add this method to the GeminiRetriever class:
pythondef _parse_and_validate_tables(self, response_text: str) -> List[str]:
    """Parse response and validate table names against actual schema."""
    # Get all actual table names from database
    try:
        sql = "SELECT DISTINCT name FROM tables"
        actual_tables = set([row[0] for row in self.conn.execute(sql).fetchall()])
    except:
        actual_tables = set()
    
    # Parse the response
    tables = []
    
    # Split by common separators
    for separator in [',', '\n', ';']:
        if separator in response_text:
            candidate_names = response_text.split(separator)
            break
    else:
        candidate_names = [response_text]
    
    for name in candidate_names:
        clean_name = name.strip().strip('"\'').strip()
        
        # Skip empty or explanatory text
        if not clean_name or len(clean_name) < 3:
            continue
            
        # Skip common explanatory words
        skip_words = ['table', 'based', 'the', 'to', 'for', 'with', 'contains', 
                     'provides', 'analysis', 'framework', 'selected', 'names',
                     'response', 'format', 'task', 'tables:']
        
        if any(word in clean_name.lower() for word in skip_words):
            continue
        
        # Check if it's an actual table name (exact match)
        if clean_name in actual_tables:
            tables.append(clean_name)
        else:
            # Try to find close matches (in case of minor variations)
            for actual_table in actual_tables:
                if actual_table.upper() == clean_name.upper():
                    tables.append(actual_table)
                    break
                elif clean_name.upper() in actual_table.upper():
                    tables.append(actual_table)
                    break
    
    logger.info(f"Validated Gemini tables: {tables}")
    return tables
Fix 4: Alternative - Use Internal Client Manager (Recommended)
Since you're in the same folder as client_manager.py, the cleanest solution is to use the internal client:
python# At the top of your script, add:
import asyncio
try:
    from client_manager import client_manager
    INTERNAL_CLIENT_AVAILABLE = True
except ImportError:
    INTERNAL_CLIENT_AVAILABLE = False

# Then replace the Gemini get_tables method with:
def get_tables(self, query: str) -> List[str]:
    logger.info(f"💎 Running Internal Gemini Vertex search for: '{query[:50]}...'")
    
    prompt = f"""You are a database expert. Select EXACT table names from the schema below.

**CRITICAL:** Use ONLY the exact table names listed in the schema. Do NOT create new names.

**Query:** {query}

**Available Tables:**
{self.schema_summary}

**Instructions:** Return only the exact table names from the schema above, separated by commas.

**Table Names:**"""
    
    try:
        if INTERNAL_CLIENT_AVAILABLE:
            # Use internal client manager
            response = asyncio.run(client_manager.ask_vertexai(prompt))
            
            logger.info(f"Internal Gemini Response: {response[:200]}...")
            
            # Validate against actual tables
            tables = self._parse_and_validate_tables(response)
            
            return tables[:10]
        else:
            # Fallback to external API calls
            return self._external_api_call(prompt)
            
    except Exception as e:
        logger.error(f"Gemini retrieval failed: {e}")
        return []
Summary of Changes Needed:

Shorter Schema: Limit to 30 tables with essential info only
Clearer Prompt: Emphasize exact table names and proper format
Table Validation: Verify returned names exist in actual schema
Better Parsing: Handle various response formats and filter out explanatory text