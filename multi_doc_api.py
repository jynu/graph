#!/usr/bin/env python3
"""
Advanced Graph Traversal Text-to-SQL API Service with Multiple Documentation Sets

A standalone FastAPI service with separate documentation for external and internal APIs.

Usage:
    python text_to_sql_api.py

Documentation:
    External API: http://localhost:8000/external-docs
    Internal API: http://localhost:8000/internal-docs
    Full API: http://localhost:8000/docs (disabled by default)

Requirements:
    pip install fastapi uvicorn duckdb networkx numpy pydantic
"""

import asyncio
import datetime
import json
import logging
import os
import time
import uuid
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from pydantic import BaseModel, Field
import duckdb
import networkx as nx
import numpy as np
from collections import defaultdict
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
DB_PATH = "knowledge_graph.duckdb"

# === Multiple App Configuration ===

# Main app with disabled docs (we'll create custom ones)
app = FastAPI(
    title="Advanced Graph Text-to-SQL API",
    description="Intelligent table discovery and SQL generation using Advanced Graph Traversal",
    version="2.0.0",
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable default redoc
    openapi_url=None  # We'll create custom openapi endpoints
)

# Tags for organizing endpoints
EXTERNAL_TAGS = ["External API"]
INTERNAL_TAGS = ["Internal API", "Database Management", "Utilities"]

# === Client Manager Integration ===
try:
    from app.utils.client_manager import client_manager
    CLIENT_MANAGER_AVAILABLE = True
    logger.info("✅ Client manager imported successfully")
except ImportError:
    CLIENT_MANAGER_AVAILABLE = False
    logger.warning("⚠️ Client manager not available - GPT features will be limited")

# === Pydantic Models ===

class TextToTableRequest(BaseModel):
    query: str = Field(..., description="Natural language query to find relevant tables")
    user_id: str = Field(..., description="User identifier")
    request_id: str = Field(default_factory=lambda: f"req_{int(time.time())}_{uuid.uuid4().hex[:8]}", description="Request identifier")
    max_tables: Optional[int] = Field(10, description="Maximum number of tables to return")

class TableToSQLRequest(BaseModel):
    selected_tables: List[str] = Field(..., description="List of selected table names")
    original_query: str = Field(..., description="Original natural language query")
    user_id: str = Field(..., description="User identifier") 
    request_id: str = Field(default_factory=lambda: f"req_{int(time.time())}_{uuid.uuid4().hex[:8]}", description="Request identifier")

class TextToSQLRequest(BaseModel):
    query: str = Field(..., description="Natural language query for complete text-to-SQL")
    user_id: str = Field(..., description="User identifier")
    request_id: str = Field(default_factory=lambda: f"req_{int(time.time())}_{uuid.uuid4().hex[:8]}", description="Request identifier")
    max_tables: Optional[int] = Field(8, description="Maximum number of tables to discover")
    include_reasoning: Optional[bool] = Field(True, description="Include reasoning in response")

class TableInfo(BaseModel):
    name: str
    description: str
    table_type: str
    record_count: Optional[int]
    columns: List[Dict]

class TextToTableResponse(BaseModel):
    success: bool
    message: str
    tables: List[str]
    table_details: Dict[str, TableInfo]
    processing_time: float
    method: str = "AdvancedGraphTraversal"

class TableToSQLResponse(BaseModel):
    success: bool
    message: str
    sql: str
    reasoning: str
    tables_used: List[str]
    processing_time: float
    validation_status: str

class TextToSQLResponse(BaseModel):
    success: bool
    message: str
    sql: str
    reasoning: str
    tables_found: List[str]
    table_details: Dict[str, TableInfo]
    processing_time: float
    method: str = "AdvancedGraphTraversal"

# === Simplified Implementation Classes ===
# (Using simplified versions for this example)

class AdvancedGraphTraversalRetriever:
    """Simplified version of the graph traversal retriever."""
    
    def __init__(self, db_path: str = DB_PATH):
        self.conn = duckdb.connect(db_path)
        logger.info("✅ Graph retriever initialized")
    
    def get_tables_with_details(self, query: str) -> Tuple[List[str], Dict]:
        """Simplified table discovery."""
        # Mock implementation for demonstration
        tables = ["users", "orders", "products"]
        details = {
            "users": {
                "description": "User information table",
                "table_type": "table",
                "columns": [{"name": "id", "data_type": "INTEGER", "description": "User ID"}]
            },
            "orders": {
                "description": "Order transactions",
                "table_type": "table", 
                "columns": [{"name": "id", "data_type": "INTEGER", "description": "Order ID"}]
            },
            "products": {
                "description": "Product catalog",
                "table_type": "table",
                "columns": [{"name": "id", "data_type": "INTEGER", "description": "Product ID"}]
            }
        }
        return tables, details
    
    def _get_table_details(self, table_names: List[str]) -> Dict:
        """Get table details."""
        return {name: {"description": f"Details for {name}", "columns": []} for name in table_names}

class SQLGenerator:
    """Simplified SQL generator."""
    
    def __init__(self):
        self.conn = duckdb.connect(DB_PATH)
    
    async def generate_sql(self, query: str, tables: List[str], table_details: Dict) -> Tuple[str, str]:
        """Generate SQL query."""
        sql = f"SELECT * FROM {tables[0]} LIMIT 10;" if tables else "SELECT 1;"
        reasoning = f"Generated SQL for query: {query}"
        return sql, reasoning
    
    def _validate_sql(self, sql: str) -> Dict:
        """Validate SQL."""
        return {'is_valid': True, 'error': None}

# Global instances
graph_retriever = None
sql_generator = None

def get_graph_retriever():
    global graph_retriever
    if graph_retriever is None:
        graph_retriever = AdvancedGraphTraversalRetriever(DB_PATH)
    return graph_retriever

def get_sql_generator():
    global sql_generator
    if sql_generator is None:
        sql_generator = SQLGenerator()
    return sql_generator

# === EXTERNAL API ENDPOINTS ===

@app.post("/text-to-sql", 
          response_model=TextToSQLResponse,
          tags=EXTERNAL_TAGS,
          summary="Convert natural language to SQL",
          description="""
          **Main External API Endpoint**
          
          Convert natural language queries directly to SQL using our Advanced Graph Traversal method.
          This is the primary endpoint for external users.
          
          **Features:**
          - Intelligent table discovery
          - Optimized SQL generation
          - Validation and error handling
          - Complete reasoning chain
          
          **Example queries:**
          - "Show me all customers who ordered more than $1000 last month"
          - "What are the top selling products by category?"
          - "Find users who haven't logged in for 30 days"
          """)
async def text_to_sql(request: TextToSQLRequest):
    """Complete text-to-SQL pipeline for external users."""
    start_time = time.time()
    
    try:
        logger.info(f"[{request.user_id}|{request.request_id}] External Text-to-SQL: {request.query}")
        
        # Find relevant tables
        retriever = get_graph_retriever()
        tables, table_details = retriever.get_tables_with_details(request.query)
        selected_tables = tables[:request.max_tables]
        
        # Generate SQL
        generator = get_sql_generator()
        sql_code, reasoning = await generator.generate_sql(
            request.query, selected_tables, table_details
        )
        
        processing_time = time.time() - start_time
        
        # Convert table details to response format
        formatted_details = {}
        for table_name, details in table_details.items():
            if table_name in selected_tables and 'error' not in details:
                formatted_details[table_name] = TableInfo(
                    name=table_name,
                    description=details.get('description', ''),
                    table_type=details.get('table_type', ''),
                    record_count=details.get('record_count'),
                    columns=details.get('columns', [])
                )
        
        return TextToSQLResponse(
            success=True,
            message="SQL generated successfully",
            sql=sql_code,
            reasoning=reasoning if request.include_reasoning else "Reasoning hidden",
            tables_found=selected_tables,
            table_details=formatted_details,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Text-to-SQL failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text-to-SQL processing failed: {str(e)}")

# === INTERNAL API ENDPOINTS ===

@app.post("/text-to-table", 
          response_model=TextToTableResponse,
          tags=INTERNAL_TAGS,
          summary="Internal: Find relevant tables",
          description="Internal endpoint for table discovery using Advanced Graph Traversal")
async def text_to_table(request: TextToTableRequest):
    """Internal endpoint: Find relevant tables only."""
    start_time = time.time()
    
    try:
        retriever = get_graph_retriever()
        tables, table_details = retriever.get_tables_with_details(request.query)
        tables = tables[:request.max_tables]
        
        processing_time = time.time() - start_time
        
        formatted_details = {}
        for table_name, details in table_details.items():
            if table_name in tables and 'error' not in details:
                formatted_details[table_name] = TableInfo(
                    name=table_name,
                    description=details.get('description', ''),
                    table_type=details.get('table_type', ''),
                    record_count=details.get('record_count'),
                    columns=details.get('columns', [])
                )
        
        return TextToTableResponse(
            success=True,
            message=f"Found {len(tables)} relevant tables",
            tables=tables,
            table_details=formatted_details,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/table-to-sql", 
          response_model=TableToSQLResponse,
          tags=INTERNAL_TAGS,
          summary="Internal: Generate SQL from tables",
          description="Internal endpoint for SQL generation from pre-selected tables")
async def table_to_sql(request: TableToSQLRequest):
    """Internal endpoint: Generate SQL from selected tables."""
    start_time = time.time()
    
    try:
        generator = get_sql_generator()
        retriever = get_graph_retriever()
        table_details = retriever._get_table_details(request.selected_tables)
        
        sql_code, reasoning = await generator.generate_sql(
            request.original_query, request.selected_tables, table_details
        )
        
        processing_time = time.time() - start_time
        validation = generator._validate_sql(sql_code)
        validation_status = "valid" if validation['is_valid'] else f"invalid: {validation['error']}"
        
        return TableToSQLResponse(
            success=True,
            message="SQL generated from selected tables",
            sql=sql_code,
            reasoning=reasoning,
            tables_used=request.selected_tables,
            processing_time=processing_time,
            validation_status=validation_status
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === UTILITY ENDPOINTS (Internal) ===

@app.get("/health",
         tags=INTERNAL_TAGS,
         summary="Internal: Health check",
         description="Internal health check endpoint")
async def health_check():
    """Health check endpoint."""
    try:
        conn = duckdb.connect(DB_PATH)
        table_count = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
        conn.close()
        
        return {
            "status": "healthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "database": {"connected": True, "table_count": table_count},
            "version": "2.0.0"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.get("/database/info",
         tags=INTERNAL_TAGS,
         summary="Internal: Database information",
         description="Get detailed database statistics and information")
async def get_database_info():
    """Get database information."""
    return {
        "database_path": DB_PATH,
        "tables_by_type": {"table": 10, "view": 2},
        "total_columns": 45,
        "total_relationships": 8,
        "last_updated": datetime.datetime.now().isoformat()
    }

@app.get("/tables/search",
         tags=INTERNAL_TAGS,
         summary="Internal: Search tables",
         description="Search tables by name or description")
async def search_tables(query: str, limit: int = 20, table_type: Optional[str] = None):
    """Search tables endpoint."""
    return {
        "query": query,
        "results": [
            {"name": "users", "description": "User data", "table_type": "table"},
            {"name": "orders", "description": "Order data", "table_type": "table"}
        ],
        "count": 2
    }

@app.post("/sql/validate",
          tags=INTERNAL_TAGS,
          summary="Internal: Validate SQL",
          description="Validate SQL query syntax")
async def validate_sql(sql_query: str):
    """Validate SQL query."""
    generator = get_sql_generator()
    validation = generator._validate_sql(sql_query)
    
    return {
        "sql": sql_query,
        "is_valid": validation['is_valid'],
        "error": validation['error'],
        "timestamp": datetime.datetime.now().isoformat()
    }

# === CUSTOM OPENAPI AND DOCUMENTATION ENDPOINTS ===

def create_external_openapi():
    """Create OpenAPI schema for external endpoints only."""
    if app.openapi_schema:
        return app.openapi_schema
    
    # Get the full OpenAPI schema
    full_openapi_schema = get_openapi(
        title="Text-to-SQL API - External",
        version="2.0.0",
        description="""
        ## Public Text-to-SQL API
        
        Convert natural language queries to SQL using advanced AI techniques.
        
        ### Main Features:
        - **Natural Language Processing**: Convert plain English to SQL
        - **Intelligent Table Discovery**: Automatically find relevant database tables  
        - **Query Optimization**: Generate efficient, validated SQL queries
        - **Error Handling**: Comprehensive validation and error reporting
        
        ### Getting Started:
        1. Use the `/text-to-sql` endpoint with your natural language query
        2. Receive optimized SQL code with reasoning
        3. Execute the SQL on your database
        
        ### Example Usage:
        ```json
        {
          "query": "Show me customers who spent more than $1000 last month",
          "user_id": "your-user-id"
        }
        ```
        """,
        routes=app.routes,
    )
    
    # Filter to only include external endpoints
    filtered_paths = {}
    for path, methods in full_openapi_schema["paths"].items():
        for method, details in methods.items():
            if details.get("tags") and any(tag in EXTERNAL_TAGS for tag in details["tags"]):
                if path not in filtered_paths:
                    filtered_paths[path] = {}
                filtered_paths[path][method] = details
    
    # Create external schema
    external_schema = full_openapi_schema.copy()
    external_schema["paths"] = filtered_paths
    external_schema["info"]["title"] = "Text-to-SQL API - External"
    external_schema["info"]["description"] = external_schema["info"]["description"]
    
    return external_schema

def create_internal_openapi():
    """Create OpenAPI schema for internal endpoints only."""
    # Get the full OpenAPI schema
    full_openapi_schema = get_openapi(
        title="Text-to-SQL API - Internal",
        version="2.0.0", 
        description="""
        ## Internal Text-to-SQL API
        
        Internal endpoints for advanced text-to-SQL operations and system management.
        
        ### Internal Features:
        - **Table Discovery**: Advanced graph-based table finding
        - **SQL Generation**: Sophisticated SQL generation from selected tables
        - **Database Management**: Health checks and database information
        - **Validation Tools**: SQL syntax validation and debugging
        - **Search Functions**: Advanced table and column search capabilities
        
        ### Internal Endpoints:
        - **Text-to-Table**: Find relevant tables using graph traversal
        - **Table-to-SQL**: Generate SQL from pre-selected tables
        - **Database Info**: Get detailed database statistics
        - **Health Check**: System status monitoring
        - **Table Search**: Advanced table discovery
        - **SQL Validation**: Query syntax checking
        
        ⚠️ **Internal Use Only**: These endpoints are for system administrators and internal applications.
        """,
        routes=app.routes,
    )
    
    # Filter to only include internal endpoints
    filtered_paths = {}
    for path, methods in full_openapi_schema["paths"].items():
        for method, details in methods.items():
            if details.get("tags") and any(tag in INTERNAL_TAGS for tag in details["tags"]):
                if path not in filtered_paths:
                    filtered_paths[path] = {}
                filtered_paths[path][method] = details
    
    # Create internal schema
    internal_schema = full_openapi_schema.copy()
    internal_schema["paths"] = filtered_paths
    internal_schema["info"]["title"] = "Text-to-SQL API - Internal"
    
    return internal_schema

# Custom OpenAPI endpoints
@app.get("/external-openapi.json", include_in_schema=False)
async def get_external_openapi():
    """Get OpenAPI schema for external endpoints."""
    return create_external_openapi()

@app.get("/internal-openapi.json", include_in_schema=False)
async def get_internal_openapi():
    """Get OpenAPI schema for internal endpoints."""
    return create_internal_openapi()

# Custom documentation endpoints
@app.get("/external-docs", include_in_schema=False)
async def get_external_documentation():
    """External API documentation."""
    return get_swagger_ui_html(
        openapi_url="/external-openapi.json",
        title="Text-to-SQL API - External Documentation",
        swagger_ui_parameters={
            "defaultModelsExpandDepth": 2,
            "defaultModelExpandDepth": 2,
            "displayRequestDuration": True,
            "filter": True,
            "showExtensions": True,
            "tryItOutEnabled": True
        }
    )

@app.get("/internal-docs", include_in_schema=False)
async def get_internal_documentation():
    """Internal API documentation."""
    return get_swagger_ui_html(
        openapi_url="/internal-openapi.json",
        title="Text-to-SQL API - Internal Documentation",
        swagger_ui_parameters={
            "defaultModelsExpandDepth": 2,
            "defaultModelExpandDepth": 2,
            "displayRequestDuration": True,
            "filter": True,
            "showExtensions": True,
            "tryItOutEnabled": True
        }
    )

# Optional: ReDoc versions
@app.get("/external-redoc", include_in_schema=False)
async def get_external_redoc():
    """External API ReDoc documentation."""
    return get_redoc_html(
        openapi_url="/external-openapi.json",
        title="Text-to-SQL API - External Documentation (ReDoc)"
    )

@app.get("/internal-redoc", include_in_schema=False)
async def get_internal_redoc():
    """Internal API ReDoc documentation."""
    return get_redoc_html(
        openapi_url="/internal-openapi.json",
        title="Text-to-SQL API - Internal Documentation (ReDoc)"
    )

# === ROOT ENDPOINT FOR NAVIGATION ===

@app.get("/", include_in_schema=False)
async def root():
    """API navigation page."""
    return {
        "message": "Advanced Graph Text-to-SQL API",
        "version": "2.0.0",
        "documentation": {
            "external_users": {
                "swagger": "/external-docs",
                "redoc": "/external-redoc",
                "openapi": "/external-openapi.json"
            },
            "internal_users": {
                "swagger": "/internal-docs", 
                "redoc": "/internal-redoc",
                "openapi": "/internal-openapi.json"
            }
        },
        "endpoints": {
            "external": ["/text-to-sql"],
            "internal": ["/text-to-table", "/table-to-sql", "/health", "/database/info", "/tables/search", "/sql/validate"]
        }
    }

# === ERROR HANDLERS ===

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.datetime.now().isoformat(),
            "status_code": exc.status_code
        }
    )

# === STARTUP EVENT ===

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("🚀 Starting Advanced Graph Text-to-SQL API...")
    logger.info("📚 External Documentation: http://localhost:8000/external-docs")
    logger.info("🔧 Internal Documentation: http://localhost:8000/internal-docs")
    logger.info("🏠 API Navigation: http://localhost:8000/")

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Text-to-SQL API Server with Multiple Documentation...")
    logger.info("📍 External API Docs: http://localhost:8000/external-docs")
    logger.info("🔧 Internal API Docs: http://localhost:8000/internal-docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")