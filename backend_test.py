#!/usr/bin/env python3
"""
Test script to verify FastAPI server can start without errors.
Run with: python backend_test.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required imports work"""
    print("\n🔍 Testing imports...")
    
    try:
        print("  ✓ Importing FastAPI modules...")
        from fastapi import FastAPI
        from pydantic import BaseModel
        print("    ✓ FastAPI OK")
        
        print("  ✓ Importing backend server...")
        from backend.main import app
        print("    ✓ backend.main OK")
        
        print("  ✓ Checking registered endpoints...")
        endpoints = [
            (route.path, route.methods if hasattr(route, 'methods') else 'N/A')
            for route in app.routes
        ]
        for path, methods in sorted(endpoints):
            if not path.startswith('/openapi') and not path.startswith('/docs'):
                print(f"    ✓ {path}")
        
        print("\n✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """Test that required configuration is available"""
    print("\n🔍 Testing configuration...")
    
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API key',
        'GEMINI_API_KEY': 'Gemini API key'
    }
    
    missing = []
    for var, desc in required_vars.items():
        if os.getenv(var):
            print(f"  ✓ {desc} ({var})")
        else:
            print(f"  ✗ {desc} ({var}) - NOT SET")
            missing.append(var)
    
    if missing:
        print(f"\n⚠️  Warning: Missing configuration: {', '.join(missing)}")
        print("  Add these to .env file before running the server")
        return False
    
    print("\n✅ All configuration OK!")
    return True


def main():
    print("=" * 60)
    print("FastAPI Backend Server Test")
    print("=" * 60)
    
    test1 = test_imports()
    test2 = test_configuration()
    
    print("\n" + "=" * 60)
    if test1 and test2:
        print("✅ Server is ready to run!")
        print("\nTo start the server, run:")
        print("  python -m uvicorn backend.main:app --reload")
        print("\nOr with the configured Python:")
        print("  C:/Users/User/.virtualenvs/packages-VbbqGVa_/Scripts/python.exe -m uvicorn backend.main:app --reload")
        return 0
    else:
        print("❌ Please fix the errors above before running the server")
        return 1


if __name__ == "__main__":
    sys.exit(main())
