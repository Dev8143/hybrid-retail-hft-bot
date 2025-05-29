#!/usr/bin/env python3
"""
Setup script for HFT Bot
Installs dependencies and sets up the environment
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, check=True):
    """Run a shell command"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, check=check)
    return result.returncode == 0

def install_python_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    
    # Upgrade pip first
    run_command(f"{sys.executable} -m pip install --upgrade pip")
    
    # Install core dependencies that are available
    core_deps = [
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "scipy>=1.10.0",
        "pyzmq>=25.0.0",
        "dash>=2.14.0",
        "plotly>=5.15.0",
        "flask>=2.3.0",
        "pyyaml>=6.0",
        "click>=8.1.0",
        "joblib>=1.3.0",
        "pytest>=7.4.0",
        "psutil>=5.9.0",
    ]
    
    for dep in core_deps:
        try:
            run_command(f"{sys.executable} -m pip install '{dep}'", check=False)
        except:
            print(f"Warning: Could not install {dep}")
    
    # Try to install optional dependencies
    optional_deps = [
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0",
        "dash-bootstrap-components>=1.4.0",
        "redis>=4.6.0",
    ]
    
    for dep in optional_deps:
        try:
            run_command(f"{sys.executable} -m pip install '{dep}'", check=False)
        except:
            print(f"Warning: Could not install optional dependency {dep}")

def create_directories():
    """Create necessary directories"""
    print("Creating directories...")
    
    directories = [
        "logs",
        "data",
        "models",
        "config",
        "monitoring"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created directory: {directory}")

def setup_environment():
    """Setup environment variables and configuration"""
    print("Setting up environment...")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write("# HFT Bot Environment Variables\n")
            f.write("PYTHONPATH=.\n")
            f.write("LOG_LEVEL=INFO\n")
            f.write("# Add your broker credentials here\n")
            f.write("# BROKER_LOGIN=\n")
            f.write("# BROKER_PASSWORD=\n")
            f.write("# BROKER_SERVER=\n")
        print("Created .env file")

def check_system_requirements():
    """Check system requirements"""
    print("Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print("Error: Python 3.8 or higher is required")
        return False
    
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Available memory: {memory.total / (1024**3):.1f} GB")
        
        if memory.total < 4 * (1024**3):  # 4GB
            print("Warning: Less than 4GB RAM available. Performance may be limited.")
    except ImportError:
        print("Could not check memory (psutil not available)")
    
    # Check CPU cores
    cpu_count = os.cpu_count()
    print(f"CPU cores: {cpu_count}")
    
    if cpu_count < 4:
        print("Warning: Less than 4 CPU cores. Performance may be limited.")
    
    return True

def run_tests():
    """Run basic tests to verify installation"""
    print("Running basic tests...")
    
    try:
        # Test imports
        import numpy as np
        import pandas as pd
        import yaml
        print("✓ Core dependencies imported successfully")
        
        # Test basic functionality
        data = np.random.randn(100, 5)
        df = pd.DataFrame(data)
        assert len(df) == 100
        print("✓ Basic functionality test passed")
        
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("="*50)
    print("HFT BOT SETUP")
    print("="*50)
    
    # Check system requirements
    if not check_system_requirements():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    install_python_dependencies()
    
    # Setup environment
    setup_environment()
    
    # Run tests
    if run_tests():
        print("\n" + "="*50)
        print("SETUP COMPLETE!")
        print("="*50)
        print("\nTo start the HFT Bot:")
        print("1. Configure your broker settings in config/hft_config.yaml")
        print("2. Run: python main.py --mode dashboard")
        print("3. Open http://localhost:12000 in your browser")
        print("\nFor full trading mode:")
        print("python main.py --mode all")
        print("\nFor training AI models:")
        print("python main.py --mode train")
    else:
        print("\n" + "="*50)
        print("SETUP FAILED!")
        print("="*50)
        print("Please check the error messages above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()