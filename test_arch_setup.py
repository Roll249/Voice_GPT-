#!/usr/bin/env python3
"""
Test script to verify the Arch Linux setup is working properly
"""

import os
import sys
import subprocess
from pathlib import Path

def test_virtual_env():
    """Test if virtual environment was created and activated"""
    venv_path = Path("venv")
    if not venv_path.exists():
        print("❌ Virtual environment not found!")
        return False
    
    print("✅ Virtual environment exists")
    
    # Test if we can run pip list in the virtual environment
    venv_pip = venv_path / "bin" / "pip" if os.name != "nt" else venv_path / "Scripts" / "pip.exe"
    if venv_pip.exists():
        try:
            result = subprocess.run([str(venv_pip), "list"], capture_output=True, text=True, check=True)
            print("✅ Virtual environment pip is accessible")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error accessing virtual environment pip: {e}")
            return False
    else:
        print(f"❌ Pip executable not found at {venv_pip}")
        return False

def test_imports():
    """Test if key packages can be imported from the virtual environment"""
    venv_python = "venv/bin/python" if os.name != "nt" else "venv\\Scripts\\python.exe"
    
    # Test imports that should be in the virtual environment
    test_imports_script = '''
try:
    import torch
    print("✅ Torch imported successfully")
except ImportError as e:
    print(f"❌ Torch import failed: {e}")

try:
    import fastapi
    print("✅ FastAPI imported successfully")
except ImportError as e:
    print(f"❌ FastAPI import failed: {e}")

try:
    import uvicorn
    print("✅ Uvicorn imported successfully")
except ImportError as e:
    print(f"❌ Uvicorn import failed: {e}")

try:
    import ddddocr
    print("✅ Ddddocr imported successfully")
except ImportError as e:
    print(f"❌ Ddddocr import failed: {e}")
'''
    
    try:
        result = subprocess.run([venv_python, "-c", test_imports_script], capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running import tests: {e}")
        print(f"stderr: {e.stderr}")
        return False

def main():
    print("Testing Arch Linux setup...")
    print("="*50)
    
    success = True
    
    # Test virtual environment
    if not test_virtual_env():
        success = False
    
    # Test imports
    if not test_imports():
        success = False
    
    print("="*50)
    if success:
        print("🎉 All tests passed! Arch Linux setup is working correctly.")
    else:
        print("❌ Some tests failed. Please check your setup.")
    
    return success

if __name__ == "__main__":
    main()