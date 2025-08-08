#!/usr/bin/env python3
"""
Debug script to diagnose FastAPI mounting issues across environments
"""

import os
import sys
import platform
import subprocess
from pathlib import Path

def debug_environment():
    """Debug current environment setup."""
    print("=" * 60)
    print("🔍 ENVIRONMENT DEBUG INFORMATION")
    print("=" * 60)
    
    # Basic system info
    print(f"🖥️  Platform: {platform.platform()}")
    print(f"🐍 Python Version: {sys.version}")
    print(f"📁 Current Directory: {os.getcwd()}")
    print(f"🔧 OS Name: {os.name}")
    
    # Check Python path
    print(f"\n📚 Python Path:")
    for i, path in enumerate(sys.path):
        print(f"   {i}: {path}")
    
    # Check installed packages
    print(f"\n📦 Key Package Versions:")
    try:
        import fastapi
        print(f"   FastAPI: {fastapi.__version__}")
    except ImportError as e:
        print(f"   FastAPI: ❌ Not found - {e}")
    
    try:
        import uvicorn
        print(f"   Uvicorn: {uvicorn.__version__}")
    except ImportError as e:
        print(f"   Uvicorn: ❌ Not found - {e}")
    
    # Check file structure
    print(f"\n📁 Project Structure:")
    current_dir = Path(".")
    
    # Check for key files/directories
    key_paths = [
        "app/",
        "app/api/",
        "app/api/api_v1/",
        "app/api/api_v1/endpoints/",
        "app/api/api_v1/endpoints/internal_router.py",
        "app/api/api_v1/endpoints/external_router.py",
        "static/",
    ]
    
    for path in key_paths:
        full_path = current_dir / path
        if full_path.exists():
            print(f"   ✅ {path}")
        else:
            print(f"   ❌ {path} (missing)")
    
    # Check environment variables
    print(f"\n🌍 Environment Variables:")
    env_vars = [
        "PYTHONPATH",
        "FASTAPI_ENV", 
        "UVICORN_HOST",
        "UVICORN_PORT"
    ]
    
    for var in env_vars:
        value = os.environ.get(var, "Not Set")
        print(f"   {var}: {value}")
    
    print("\n" + "=" * 60)

def test_imports():
    """Test all critical imports."""
    print("🧪 TESTING IMPORTS")
    print("=" * 60)
    
    imports_to_test = [
        ("fastapi", "FastAPI"),
        ("fastapi.middleware.cors", "CORSMiddleware"),
        ("app.api.api_v1.endpoints.internal_router", "internal_router"),
        ("app.api.api_v1.endpoints.external_router", "external_router"),
        ("app.api.api_v1.endpoints", "query"),
        ("app.api.api_v1.endpoints", "data_processing"),
        ("app.web.chatui", "DCChatUI"),
        ("app.rag.vanna", "vn"),
    ]
    
    for module_name, import_name in imports_to_test:
        try:
            module = __import__(module_name, fromlist=[import_name])
            obj = getattr(module, import_name)
            print(f"   ✅ {module_name}.{import_name}")
        except ImportError as e:
            print(f"   ❌ {module_name}.{import_name} - ImportError: {e}")
        except AttributeError as e:
            print(f"   ❌ {module_name}.{import_name} - AttributeError: {e}")
        except Exception as e:
            print(f"   ❌ {module_name}.{import_name} - Error: {e}")

def test_app_creation():
    """Test FastAPI app creation and mounting."""
    print(f"\n🏗️  TESTING APP CREATION")
    print("=" * 60)
    
    try:
        from fastapi import FastAPI
        
        # Test external app
        print("   Creating external app...")
        external_app = FastAPI(title="Test External", docs_url="/docs")
        print(f"   ✅ External app created: {external_app.title}")
        
        # Test internal app  
        print("   Creating internal app...")
        internal_app = FastAPI(title="Test Internal", docs_url="/docs")
        print(f"   ✅ Internal app created: {internal_app.title}")
        
        # Test main app
        print("   Creating main app...")
        main_app = FastAPI(title="Test Main", docs_url=None)
        print(f"   ✅ Main app created: {main_app.title}")
        
        # Test mounting
        print("   Testing app mounting...")
        main_app.mount("/public", external_app, name="public")
        main_app.mount("/sandbox", internal_app, name="sandbox")
        print("   ✅ Apps mounted successfully")
        
        # Test routes
        print("   Checking mounted routes...")
        route_count = len(main_app.routes)
        print(f"   ✅ Main app has {route_count} routes")
        
        for route in main_app.routes:
            if hasattr(route, 'path'):
                print(f"      - {route.path}")
        
    except Exception as e:
        print(f"   ❌ App creation failed: {e}")
        import traceback
        traceback.print_exc()

def generate_fixed_main():
    """Generate a simplified main.py for testing."""
    print(f"\n📝 GENERATING SIMPLIFIED MAIN.PY")
    print("=" * 60)
    
    simplified_main = '''
# Simplified main.py for debugging
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create sub-apps
external_app = FastAPI(title="DC Lite Access", docs_url="/docs")
internal_app = FastAPI(title="DC Fullstack Access", docs_url="/docs")

# Add CORS
for app in [external_app, internal_app]:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Try to import routers (with error handling)
try:
    from app.api.api_v1.endpoints.external_router import external_router
    external_app.include_router(external_router)
    logger.info("✅ External router loaded")
except Exception as e:
    logger.error(f"❌ External router failed: {e}")

try:
    from app.api.api_v1.endpoints.internal_router import internal_router
    internal_app.include_router(internal_router)
    logger.info("✅ Internal router loaded")
except Exception as e:
    logger.error(f"❌ Internal router failed: {e}")

# Create main app
main_app = FastAPI(title="DC Chat Platform", docs_url=None)
main_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Mount apps
main_app.mount("/public", external_app, name="public")
main_app.mount("/sandbox", internal_app, name="sandbox")

@main_app.get("/")
def root():
    return {
        "message": "DC Chat Platform",
        "docs": {
            "public": "/public/docs",
            "internal": "/sandbox/docs"
        }
    }

@main_app.get("/debug")
def debug():
    return {
        "routes": [{"path": r.path, "name": getattr(r, 'name', 'unknown')} for r in main_app.routes],
        "mounts": [{"path": "/public", "app": "external"}, {"path": "/sandbox", "app": "internal"}]
    }

app = main_app

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting simplified app...")
    logger.info("📚 Public docs: http://127.0.0.1:8000/public/docs")
    logger.info("🔧 Internal docs: http://127.0.0.1:8000/sandbox/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)
'''
    
    with open("main_debug.py", "w") as f:
        f.write(simplified_main)
    
    print("   ✅ Created main_debug.py")
    print("   📋 To test: python main_debug.py")

if __name__ == "__main__":
    debug_environment()
    test_imports()
    test_app_creation()
    generate_fixed_main()
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Run: python main_debug.py")
    print("2. Check: http://127.0.0.1:8000/debug")
    print("3. Verify: http://127.0.0.1:8000/public/docs")
    print("4. Verify: http://127.0.0.1:8000/sandbox/docs")
    print("5. Compare behavior between dev and ECS")