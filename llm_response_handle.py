1. Update the _extract_columns_llm method
Find this method in the QueryLogAnalyzer class and modify the response processing part:
Current problematic code (around line 570):
python# Call LLM
print(extraction_prompt)
response = await client_manager.ask_vertexai(extraction_prompt)
print(response)
result = json.loads(response)  # This line fails
Replace with:
python# Call LLM
print(extraction_prompt)
response = await client_manager.ask_vertexai(extraction_prompt)
print(response)

# Clean the response to extract JSON
cleaned_response = self._clean_llm_response(response)
result = json.loads(cleaned_response)
2. Add a new helper method _clean_llm_response
Add this new method to the QueryLogAnalyzer class (after the _extract_columns_llm method):
pythondef _clean_llm_response(self, response: str) -> str:
    """Clean LLM response to extract valid JSON"""
    if not response:
        raise ValueError("Empty response from LLM")
    
    # Remove markdown code blocks if present
    import re
    
    # Pattern to match ```json ... ``` or ``` ... ```
    json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(json_pattern, response, re.DOTALL | re.IGNORECASE)
    
    if match:
        # Extract JSON from code blocks
        json_content = match.group(1).strip()
        self.logger.debug("Extracted JSON from markdown code blocks")
        return json_content
    
    # If no code blocks, try to find JSON object directly
    # Look for { ... } pattern
    json_object_pattern = r'\{.*\}'
    match = re.search(json_object_pattern, response, re.DOTALL)
    
    if match:
        json_content = match.group(0).strip()
        self.logger.debug("Extracted JSON object directly")
        return json_content
    
    # If still no JSON found, return the response as-is and let json.loads handle the error
    self.logger.warning("Could not extract JSON from LLM response, returning raw response")
    return response.strip()
3. Update the LLM prompt for better formatting
Modify the extraction prompt in the _extract_columns_llm method to be more explicit about JSON format:
Find this part (around line 520):
python**Required Output Format (JSON):**
{
    "is_valid_sql": true/false,
    "extracted_columns": ["column1", "column2", "column3"],
    "confidence": 0.0-1.0
}

**Examples:**
- "SELECT t.trade_id, t.amount FROM trades t" → ["trade_id", "amount"]
- "WHERE settlement_date > '2024-01-01'" → ["settlement_date"]
- "GROUP BY currency, status" → ["currency", "status"]

**Respond with only the JSON:**"""
Replace with:
python**Required Output Format:**
Return ONLY a valid JSON object (no markdown, no code blocks, no explanations):

{
    "is_valid_sql": true,
    "extracted_columns": ["column1", "column2", "column3"],
    "confidence": 0.85
}

**Examples:**
- "SELECT t.trade_id, t.amount FROM trades t" → ["trade_id", "amount"]
- "WHERE settlement_date > '2024-01-01'" → ["settlement_date"]  
- "GROUP BY currency, status" → ["currency", "status"]

**IMPORTANT: Respond with ONLY the JSON object, no additional text or formatting:**"""
4. Add better error handling
Update the exception handling in _extract_columns_llm method:
Find this part:
pythonexcept Exception as e:
    self.logger.error(f"LLM extraction failed: {e}")
    return set()
Replace with:
pythonexcept json.JSONDecodeError as e:
    self.logger.error(f"Failed to parse LLM JSON response: {e}")
    self.logger.debug(f"Raw LLM response: {response[:200]}...")
    return set()
except Exception as e:
    self.logger.error(f"LLM extraction failed: {e}")
    return set()
📋 Summary of Changes: