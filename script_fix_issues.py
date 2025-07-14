Fix 1: Update OpenAI Retriever Response Parsing (Lines ~380-420)
The main issue is in how you parse the response. Based on others.py, the response format is different:
pythondef get_tables(self, query: str) -> List[str]:
    logger.info(f"🤖 Running Enhanced OpenAI GPT-4 search for: '{query[:50]}...'")
    
    # Your existing prompt logic...
    
    try:
        import requests     

        # Combine system and user prompts into a single message
        full_message = f"{system_prompt}\n\n{user_prompt}"
        
        # Use multipart/form-data as shown in curl command
        files = {
            'message': (None, full_message)
        }
        
        headers = {
            'accept': 'application/json'
        }
        
        response = requests.post(
            "https://olympus-dc-server-icg-isg-olympus-high-volume-api-167969.apps.namicg39024u.ecs.dyn.nsroot.net/api/v1/ask_gpt",
            files=files,
            headers=headers,
            timeout=30,
            verify="CitiInternalCAChain_PROD.pem"
        )
        
        logger.info(f"OpenAI Status Code: {response.status_code}")
        logger.info(f"OpenAI Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            
            # FIX: Based on others.py, the response format is {"response": "content"}
            if 'response' in result:
                response_text = result['response'].strip()
            else:
                logger.warning(f"Unexpected response format: {list(result.keys())}")
                response_text = str(result)
            
            # Clean up the response
            tables = []
            for name in response_text.split(','):
                clean_name = name.strip().strip('"\'')
                if clean_name and not clean_name.lower().startswith(('based', 'the', 'to', 'for')):
                    tables.append(clean_name)
            
            return tables[:10]
        else:
            logger.error(f"OpenAI API returned status {response.status_code}: {response.text}")
            return []
            
    except Exception as e:
        logger.error(f"OpenAI retrieval failed: {e}")
        return []
Fix 2: Update Gemini Retriever Response Parsing (Lines ~480-520)
Same issue - fix the response parsing:
pythondef get_tables(self, query: str) -> List[str]:
    logger.info(f"💎 Running Enhanced Google Gemini search for: '{query[:50]}...'")
    
    # Your existing prompt logic...
    
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
        logger.info(f"Gemini Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            
            # FIX: Based on others.py, the response format is {"response": "content"}
            if 'response' in result:
                response_text = result['response'].strip()
            else:
                logger.warning(f"Unexpected Gemini response format: {list(result.keys())}")
                response_text = str(result)
            
            # Clean and parse the response
            tables = []
            for name in response_text.split(','):
                clean_name = name.strip().strip('"\'').strip()
                if clean_name and not any(word in clean_name.lower() for word in 
                    ['table', 'based', 'the', 'to', 'for', 'with', 'contains', 'provides']):
                    tables.append(clean_name)
            
            return tables[:10]
        else:
            logger.error(f"Gemini API returned status {response.status_code}: {response.text}")
            return []
            
    except Exception as e:
        logger.error(f"Gemini retrieval failed: {e}")
        return []
Fix 3: Fix URL Syntax Error in Gemini (Line ~507)
You have a syntax error in the Gemini URL:
python# WRONG (has extra quote):
response = requests.post(f"'https://olympus-dc-server-icg-isg-olympus-high-volume-api-167969.apps.namicg39024u.ecs.dyn.nsroot.net/api/v1/ask_vertexai",

# CORRECT:
response = requests.post(
    "https://olympus-dc-server-icg-isg-olympus-high-volume-api-167969.apps.namicg39024u.ecs.dyn.nsroot.net/api/v1/ask_vertexai",
Fix 4: Add Better Error Handling and Debug Info
Add this helper function for both retrievers:
pythondef _fallback_keyword_search(self, query: str) -> List[str]:
    """Fallback to simple keyword search if vector search fails."""
    try:
        search_term = f"%{query.lower()}%"
        sql = """
        SELECT DISTINCT name FROM tables
        WHERE LOWER(name) LIKE ? OR LOWER(description) LIKE ?
        LIMIT 5
        """
        results = self.conn.execute(sql, [search_term, search_term]).fetchall()
        return [row[0] for row in results]
    except:
        return []
Fix 5: Test Function (Add this for debugging)
Add this test function to verify your APIs work:
pythondef test_both_apis():
    """Test both GPT and Gemini APIs with simple messages."""
    import requests
    
    # Test GPT
    print("=== Testing GPT API ===")
    try:
        files = {'message': (None, "List 3 common database table types")}
        headers = {'accept': 'application/json'}
        
        response = requests.post(
            "https://olympus-dc-server-icg-isg-olympus-high-volume-api-167969.apps.namicg39024u.ecs.dyn.nsroot.net/api/v1/ask_gpt",
            files=files,
            headers=headers,
            verify="CitiInternalCAChain_PROD.pem",
            timeout=30
        )
        
        print(f"GPT Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"GPT Response Keys: {list(result.keys())}")
            print(f"GPT Content: {result.get('response', 'No response key')[:100]}...")
        else:
            print(f"GPT Error: {response.text}")
            
    except Exception as e:
        print(f"GPT Test Failed: {e}")
    
    # Test Gemini
    print("\n=== Testing Gemini API ===")
    try:
        files = {'message': (None, "List 3 common database table types")}
        headers = {'accept': 'application/json'}
        
        response = requests.post(
            "https://olympus-dc-server-icg-isg-olympus-high-volume-api-167969.apps.namicg39024u.ecs.dyn.nsroot.net/api/v1/ask_vertexai",
            files=files,
            headers=headers,
            verify="CitiInternalCAChain_PROD.pem",
            timeout=30
        )
        
        print(f"Gemini Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Gemini Response Keys: {list(result.keys())}")
            print(f"Gemini Content: {result.get('response', 'No response key')[:100]}...")
        else:
            print(f"Gemini Error: {response.text}")
            
    except Exception as e:
        print(f"Gemini Test Failed: {e}")

# Add this call in your main function to test first
# test_both_apis()
Summary of Key Changes:

Response Parsing: Change from result['choices'][0]['message']['content'] to result['response']
URL Fix: Remove extra quote in Gemini URL
Error Handling: Add better logging and fallback options
Testing: Add test function to verify API connectivity

The main issue was that you were trying to parse the response as if it came from OpenAI's format, but your internal APIs return a simpler {"response": "content"} format.