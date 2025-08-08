__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import threading
import os
import logging
from typing import Any, Dict, Union
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

# Import your existing routers
from app.api.api_v1.endpoints import query, data_processing, vanna, duckdb, evaluation, others, agent_slice, liveness
from app.api.api_v1.endpoints.internal_router import internal_router
from app.api.api_v1.endpoints.external_router import external_router
from vanna.flask import VannaFlaskApp
from app.web.chatui import DCChatUI
from app.rag.vanna import vn
from app.core.runner.database_manager import DatabaseManager
from app.core.middleware import context_cleanup_middleware

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS configuration
CORS_CONFIG = {
    "allow_origins": ["*"],
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}

# === CREATE SUB-APPLICATIONS FIRST ===

# External App (DC Lite Access)
external_app = FastAPI(
    title="DC Lite Access",
    description="Public API endpoints for external users",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS to external app
external_app.add_middleware(CORSMiddleware, **CORS_CONFIG)

# Include external router
external_app.include_router(external_router)
logger.info("✅ External app created with external_router")

# Internal App (DC Fullstack Access)
internal_app = FastAPI(
    title="DC Fullstack Access", 
    description="Internal API endpoints with full system access",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS to internal app
internal_app.add_middleware(CORSMiddleware, **CORS_CONFIG)

# Include internal router
internal_app.include_router(internal_router)

# ALSO include all the individual routers for backward compatibility
internal_app.include_router(query.router)
internal_app.include_router(data_processing.router)
internal_app.include_router(vanna.router)
internal_app.include_router(duckdb.router)
internal_app.include_router(evaluation.router)
internal_app.include_router(agent_slice.router)
internal_app.include_router(others.router)
internal_app.include_router(liveness.router)

logger.info("✅ Internal app created with all routers")

# === CREATE MAIN APPLICATION ===

# Main app - acts as a router/gateway
main_app = FastAPI(
    title="DC Chat Platform",
    description="Multi-tenant chat platform with public and internal APIs",
    version="1.0.0",
    docs_url=None,  # Disable main app docs
    redoc_url=None,  # Disable main app redoc
    openapi_url=None  # Disable main app openapi
)

# Add CORS to main app
main_app.add_middleware(CORSMiddleware, **CORS_CONFIG)

# Add context cleanup middleware
main_app.middleware("http")(context_cleanup_middleware)

# Mount static files
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
if os.path.exists(static_dir):
    main_app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"✅ Static files mounted from {static_dir}")
else:
    logger.warning(f"⚠️ Static directory not found: {static_dir}")

# === MOUNT SUB-APPLICATIONS ===

# Mount external app at /public
main_app.mount("/public", external_app, name="public_api")
logger.info("✅ External app mounted at /public")

# Mount internal app at /sandbox  
main_app.mount("/sandbox", internal_app, name="internal_api")
logger.info("✅ Internal app mounted at /sandbox")

# === ROOT ENDPOINT FOR NAVIGATION ===

@main_app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Landing page with navigation to different API docs."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DC Chat Platform</title>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 800px; 
                margin: 50px auto; 
                padding: 20px;
                background: #f5f5f5;
            }
            .container {
                background: white;
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 { color: #333; text-align: center; margin-bottom: 30px; }
            h2 { color: #555; border-bottom: 2px solid #eee; padding-bottom: 10px; }
            .api-section {
                margin: 30px 0;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 8px;
                background: #fafafa;
            }
            .links { margin: 15px 0; }
            .links a {
                display: inline-block;
                margin: 5px 10px 5px 0;
                padding: 8px 16px;
                background: #007bff;
                color: white;
                text-decoration: none;
                border-radius: 4px;
                font-size: 14px;
            }
            .links a:hover { background: #0056b3; }
            .external { border-left: 4px solid #28a745; }
            .internal { border-left: 4px solid #dc3545; }
            .status { 
                text-align: center; 
                margin: 20px 0; 
                padding: 10px;
                background: #d4edda;
                border: 1px solid #c3e6cb;
                border-radius: 4px;
                color: #155724;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 DC Chat Platform</h1>
            
            <div class="status">
                ✅ All services are running and available
            </div>
            
            <div class="api-section external">
                <h2>🌐 Public API (DC Lite Access)</h2>
                <p><strong>For external users and public integrations</strong></p>
                <p>Limited set of endpoints for safe external usage.</p>
                <div class="links">
                    <a href="/public/docs" target="_blank">📚 Swagger Docs</a>
                    <a href="/public/redoc" target="_blank">📖 ReDoc</a>
                    <a href="/public/openapi.json" target="_blank">🔧 OpenAPI JSON</a>
                </div>
            </div>
            
            <div class="api-section internal">
                <h2>🔧 Internal API (DC Fullstack Access)</h2>
                <p><strong>For internal systems and development</strong></p>
                <p>Complete access to all endpoints including admin functions.</p>
                <div class="links">
                    <a href="/sandbox/docs" target="_blank">📚 Swagger Docs</a>
                    <a href="/sandbox/redoc" target="_blank">📖 ReDoc</a>
                    <a href="/sandbox/openapi.json" target="_blank">🔧 OpenAPI JSON</a>
                </div>
            </div>
            
            <hr style="margin: 40px 0;">
            
            <div style="text-align: center; color: #666; font-size: 14px;">
                <p>🏠 <strong>Direct Access URLs:</strong></p>
                <p>Public API: <code>/public/*</code> | Internal API: <code>/sandbox/*</code></p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

# === HEALTH CHECK ENDPOINTS ===

@main_app.get("/health", include_in_schema=False)
async def health_check():
    """Main health check endpoint."""
    return {
        "status": "healthy",
        "service": "DC Chat Platform",
        "apps": {
            "main": "running",
            "external": "mounted at /public",
            "internal": "mounted at /sandbox"
        },
        "endpoints": {
            "public_docs": "/public/docs",
            "internal_docs": "/sandbox/docs",
            "landing": "/"
        }
    }

@main_app.get("/debug/routes", include_in_schema=False)
async def debug_routes():
    """Debug endpoint to check mounted routes."""
    routes_info = []
    
    for route in main_app.routes:
        route_info = {
            "path": getattr(route, 'path', 'N/A'),
            "name": getattr(route, 'name', 'N/A'),
            "methods": getattr(route, 'methods', 'N/A')
        }
        routes_info.append(route_info)
    
    return {
        "main_app_routes": routes_info,
        "mounted_apps": [
            {"path": "/public", "name": "public_api"},
            {"path": "/sandbox", "name": "internal_api"}
        ]
    }

# === VANNA WEB THREAD ===

def run_vanna_web():
    """Run Vanna web interface in separate thread."""
    try:
        logger.info("🌐 Starting Vanna web interface...")
        DCChatUI(vn, allow_llm_to_see_data=True).run()
    except Exception as e:
        logger.error(f"❌ Vanna web interface failed: {e}")

# Start Vanna web server only on POSIX systems
if os.name == "posix":
    try:
        thread = threading.Thread(target=run_vanna_web, daemon=True)
        thread.start()
        logger.info("✅ Vanna web thread started")
        
        # Write thread PID
        with open('thread.pid', 'w') as f:
            f.write(str(thread.ident))
        logger.info(f"📝 Thread PID written: {thread.ident}")
            
    except Exception as e:
        logger.error(f"❌ Failed to start Vanna thread: {e}")

# === STARTUP EVENT ===

@main_app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info("🚀 DC Chat Platform Starting...")
    logger.info("🌍 Main App: http://127.0.0.1:8000/")
    logger.info("🌐 Public API Docs: http://127.0.0.1:8000/public/docs")
    logger.info("🔧 Internal API Docs: http://127.0.0.1:8000/sandbox/docs")
    logger.info("🏥 Health Check: http://127.0.0.1:8000/health")
    logger.info("🐛 Debug Routes: http://127.0.0.1:8000/debug/routes")

# === ERROR HANDLERS ===

@main_app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Custom 404 handler with helpful navigation."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested endpoint was not found",
            "available_docs": {
                "public_api": "/public/docs",
                "internal_api": "/sandbox/docs",
                "landing_page": "/"
            },
            "requested_path": str(request.url.path)
        }
    )

# Export the main app for uvicorn
app = main_app

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting DC Chat Platform...")
    logger.info("📊 Environment: Development")
    logger.info("🔗 Access URLs:")
    logger.info("   Main: http://127.0.0.1:8000/")
    logger.info("   Public: http://127.0.0.1:8000/public/docs")
    logger.info("   Internal: http://127.0.0.1:8000/sandbox/docs")
    
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8000,
        log_level="info",
        access_log=True
    )