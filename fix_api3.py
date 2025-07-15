Fix: Update to Use Query Parameters Instead of Form Data
Here's how to fix both retrievers:
Fix 1: Update OpenAI Retriever (Lines ~380-420)
pythondef get_tables(self, query: str) -> List[str]:
    logger.info(f"🤖 Running Enhanced OpenAI GPT-4 search for: '{query[:50]}...'")
    
    # Your existing prompt logic...
    system_prompt = """You are an expert database analyst..."""
    user_prompt = f"""**Financial Data Query:** {query}..."""
    
    try:
        import requests     

        # Combine system and user prompts into a single message
        full_message = f"{system_prompt}\n\n{user_prompt}"
        
        # FIX: Use query parameters instead of form data
        params = {
            'message': full_message
        }
        
        headers = {
            'accept': 'application/json'
        }
        
        # FIX: Use GET request with params instead of POST with files
        response = requests.get(  # Changed from POST to GET
            "https://olympus-dc-server-icg-isg-olympus-high-volume-api-167969.apps.namicg39024u.ecs.dyn.nsroot.net/api/v1/ask_gpt",
            params=params,  # Changed from files to params
            headers=headers,
            timeout=30,
            verify="CitiInternalCAChain_PROD.pem"
        )
        
        logger.info(f"OpenAI Status Code: {response.status_code}")
        logger.info(f"OpenAI Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Based on others.py, the response format is {"response": "content"}
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
Fix 2: Update Gemini Retriever (Lines ~480-520)
pythondef get_tables(self, query: str) -> List[str]:
    logger.info(f"💎 Running Enhanced Google Gemini search for: '{query[:50]}...'")
    
    # Your existing prompt logic...
    prompt = f"""You are a financial data expert..."""
    
    try:
        import requests
        
        # FIX: Use query parameters instead of form data
        params = {
            'message': prompt
        }
        
        headers = {
            'accept': 'application/json'
        }
        
        # FIX: Use GET request with params instead of POST with files
        response = requests.get(  # Changed from POST to GET
            "https://olympus-dc-server-icg-isg-olympus-high-volume-api-167969.apps.namicg39024u.ecs.dyn.nsroot.net/api/v1/ask_vertexai",
            params=params,  # Changed from files to params
            headers=headers,
            timeout=30,
            verify="CitiInternalCAChain_PROD.pem"
        )
        
        logger.info(f"Gemini Status Code: {response.status_code}")
        logger.info(f"Gemini Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Based on others.py, the response format is {"response": "content"}
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
Alternative: If Still Using POST
If the API still expects POST, try this format:
python# Alternative: POST with JSON body instead of form data
headers = {
    'Content-Type': 'application/json',
    'accept': 'application/json'
}

payload = {
    'message': full_message
}

response = requests.post(
    "https://olympus-dc-server-icg-isg-olympus-high-volume-api-167969.apps.namicg39024u.ecs.dyn.nsroot.net/api/v1/ask_gpt",
    json=payload,  # Use json instead of files or params
    headers=headers,
    timeout=30,
    verify="CitiInternalCAChain_PROD.pem"
)
Test Function to Verify
Add this test function to check which format works:
pythondef test_api_formats():
    """Test different API request formats."""
    import requests
    
    test_message = "List common database table types"
    
    # Test 1: GET with query params
    print("=== Testing GET with params ===")
    try:
        response = requests.get(
            "https://olympus-dc-server-icg-isg-olympus-high-volume-api-167969.apps.namicg39024u.ecs.dyn.nsroot.net/api/v1/ask_gpt",
            params={'message': test_message},
            headers={'accept': 'application/json'},
            verify="CitiInternalCAChain_PROD.pem",
            timeout=10
        )
        print(f"GET Status: {response.status_code}")
        print(f"GET Response: {response.text[:200]}...")
    except Exception as e:
        print(f"GET Failed: {e}")
    
    # Test 2: POST with JSON
    print("\n=== Testing POST with JSON ===")
    try:
        response = requests.post(
            "https://olympus-dc-server-icg-isg-olympus-high-volume-api-167969.apps.namicg39024u.ecs.dyn.nsroot.net/api/v1/ask_gpt",
            json={'message': test_message},
            headers={'Content-Type': 'application/json', 'accept': 'application/json'},
            verify="CitiInternalCAChain_PROD.pem",
            timeout=10
        )
        print(f"POST JSON Status: {response.status_code}")
        print(f"POST JSON Response: {response.text[:200]}...")
    except Exception as e:
        print(f"POST JSON Failed: {e}")

# Run this to see which format works
test_api_formats()
The key changes are: