1. Update OpenAI Retriever (Lines ~450-520)
Replace the __init__ and API call sections:

class OpenAIRetriever(BaseDuckDBRetriever):
    def __init__(self, db_path: str = "knowledge_graph.duckdb"):
        super().__init__(db_path)
        # Use your UAT H2O API key and Azure endpoint
        self.api_key = UAT_H2O_API_KEY
        self.api_url = UAT_AZURE_API_URL
        self.schema_summary = self._get_enhanced_schema_summary()
    
    def get_tables(self, query: str) -> List[str]:
        logger.info(f"🤖 Running Enhanced OpenAI GPT-4 search for: '{query[:50]}...'")
        
        # Your existing prompt logic here...
        
        try:
            import requests
            
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "model": "Citi-GPT4-UAT",  # Based on your config
                "temperature": 0.1,
                "max_tokens": 300,
                "top_p": 0.9
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            response = requests.post(
                f"{self.api_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=30,
                verify=False  # Based on your SSL settings
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result['choices'][0]['message']['content'].strip()
                # Your existing parsing logic...
                
        except Exception as e:
            logger.error(f"OpenAI retrieval failed: {e}")
            return []
            
            
            
2. Update Gemini Retriever (Lines ~580-650)
Replace the __init__ and API call sections:

class GeminiRetriever(BaseDuckDBRetriever):
    def __init__(self, db_path: str = "knowledge_graph.duckdb"):
        super().__init__(db_path)
        # Use your Vertex AI endpoint
        self.api_url = UAT_VERTEX_API_URL
        self.api_key = UAT_H2O_API_KEY
        self.schema_summary = self._get_enhanced_schema_summary()
    
    def get_tables(self, query: str) -> List[str]:
        logger.info(f"💎 Running Enhanced Google Gemini search for: '{query[:50]}...'")
        
        # Your existing prompt logic here...
        
        try:
            import requests
            
            payload = {
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "model": "gemini-1.5-pro-002",  # Based on your Vertex config
                "max_output_tokens": 300,
                "temperature": 0.1,
                "top_p": 0.9
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            response = requests.post(
                f"{self.api_url}/chat/completions",  # Or appropriate Vertex endpoint
                json=payload,
                headers=headers,
                timeout=30,
                verify=False
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result['choices'][0]['message']['content'].strip()
                # Your existing parsing logic...
                
        except Exception as e:
            logger.error(f"Gemini retrieval failed: {e}")
            return []
            
            
3. Update Configuration Section (Lines ~30-50)
Replace the API configuration:
python# Use your UAT environment settings directly
UAT_H2O_API_KEY = "sk-rerYub6aZ0yptPg7FMQwbfe129h3oh1UeIA0UNX5Z7yVUyS"
UAT_AZURE_API_URL = "https://r2d2-c3po-icg-msst-genaihub-178909.apps.namicg39023u.ecs.dyn.nsroot.net/azure"
UAT_VERTEX_API_URL = "https://r2d2-c3po-icg-msst-genaihub-178909.apps.namicg39023u.ecs.dyn.nsroot.net/vertex"

# Remove the external API dependencies
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "test")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-key-here")


4. Update Azure OpenAI Retriever (Lines ~700-750)
This one should already work, but update the model name:
pythonpayload = {
    "messages": [...],
    "model": "Citi-GPT4-UAT",  # Use your specific model
    "max_tokens": 200,
    "temperature": 0.1
}

5. Update Initialization Logic (Lines ~800-850)
python# Update the LLM method initialization
# Replace the OPENAI_API_KEY and GEMINI_API_KEY checks with:

# Always try to initialize since you have internal endpoints
try:
    retrievers["OpenAI GPT-4"] = OpenAIRetriever(DB_PATH)
    print(f"  ✅ Enhanced OpenAI GPT-4 initialized (UAT)")
except Exception as e:
    print(f"  ❌ OpenAIRetriever failed: {e}")

try:
    retrievers["Gemini Vertex"] = GeminiRetriever(DB_PATH)
    print(f"  ✅ Enhanced Gemini Vertex initialized (UAT)")
except Exception as e:
    print(f"  ❌ GeminiRetriever failed: {e}")