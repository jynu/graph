Update 1: Fix the API URL and Endpoint
Replace in your configuration section:
python# Update this URL based on the curl command
UAT_AZURE_API_URL = "https://olympus-dc-server-icg-isg-olympus-high-volume-api-167969.apps.namicg39024u.ecs.dyn.nsroot.net"
Update 2: Fix OpenAI Retriever API Call
Replace the entire API call section in OpenAIRetriever.get_tables():
pythontry:
    import requests
    
    # Combine system and user prompts into a single message
    full_message = f"{system_prompt}\n\n{user_prompt}"
    
    # Use multipart/form-data as shown in curl command
    files = {
        'message': (None, full_message)
    }
    
    headers = {
        'accept': 'application/json'
        # Note: Don't set Content-Type header when using files parameter
        # requests will set it automatically with boundary
    }
    
    response = requests.post(
        f"{self.api_url}/v1/ask_gpt",  # Use the correct endpoint
        files=files,
        headers=headers,
        timeout=30,
        verify="PROD.pem"
    )
    
    if response.status_code == 200:
        result = response.json()
        # The response structure might be different, check what's returned
        # You may need to adjust this based on actual response format
        if 'choices' in result:
            response_text = result['choices'][0]['message']['content'].strip()
        elif 'response' in result:
            response_text = result['response'].strip()
        else:
            response_text = str(result).strip()
        
        # Your existing parsing logic...
        tables = []
        for name in response_text.split(','):
            clean_name = name.strip().strip('"\'')
            if clean_name and not clean_name.lower().startswith(('based', 'the', 'to', 'for')):
                tables.append(clean_name)
        
        return tables[:10]
    else:
        logger.error(f"API returned status {response.status_code}: {response.text}")
        return []
        
except Exception as e:
    logger.error(f"OpenAI retrieval failed: {e}")
    return []
Update 3: Test the API First
Before running the full benchmark, test the API call with a simple test:
pythondef test_api_connection():
    """Test API connection with a simple message."""
    try:
        import requests
        
        url = "https://olympus-dc-server-icg-isg-olympus-high-volume-api-167969.apps.namicg39024u.ecs.dyn.nsroot.net/v1/ask_gpt"
        
        files = {
            'message': (None, "Tell me a story")
        }
        
        headers = {
            'accept': 'application/json'
        }
        
        response = requests.post(
            url,
            files=files,
            headers=headers,
            timeout=30,
            verify="PROD.pem"
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"API test failed: {e}")
        return False

# Add this call in your main function for testing
Update 4: Handle Different Response Format
Since the API structure is different, you might need to adjust how you parse the response. Add some debugging to see the actual response format:
pythonif response.status_code == 200:
    result = response.json()
    logger.info(f"API Response structure: {list(result.keys())}")  # Debug line
    
    # Try different response formats
    if 'choices' in result:
        response_text = result['choices'][0]['message']['content']
    elif 'response' in result:
        response_text = result['response']
    elif 'answer' in result:
        response_text = result['answer']
    elif 'text' in result:
        response_text = result['text']
    else:
        logger.warning(f"Unknown response format: {result}")
        response_text = str(result)
The key changes are:

Use the correct URL from the curl command
Use /v1/ask_gpt endpoint instead of /chat/completions
Use multipart/form-data format with files parameter
Remove JSON payload and use form data instead

Try these updates and let me know what response you get!