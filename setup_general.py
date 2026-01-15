#!/usr/bin/env python3
"""
General setup script for Vietnamese Text-to-Speech Application
Works on various Linux distributions including non-Arch systems
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_os():
    """Check the operating system"""
    system = platform.system().lower()
    distro = "unknown"
    
    if system == "linux":
        try:
            # Try to get distribution info
            import distro
            distro = distro.id().lower()
        except ImportError:
            # If distro module is not available, try reading /etc/os-release
            try:
                with open('/etc/os-release', 'r') as f:
                    for line in f:
                        if line.startswith('ID='):
                            distro = line.split('=')[1].strip().strip('"').lower()
                            break
            except FileNotFoundError:
                pass
    
    logger.info(f"Detected OS: {system}, Distribution: {distro}")
    return system, distro

def create_directories():
    """Create necessary directories for the application"""
    directories = ["models", "uploads", "outputs", "references", "temp", "data", "scripts"]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        else:
            logger.info(f"Directory already exists: {directory}")

def install_system_dependencies():
    """Install system-level dependencies"""
    system, distro = check_os()
    
    if system != "linux":
        logger.warning(f"This script is designed for Linux systems. Detected: {system}")
        logger.info("Continuing anyway...")
        return
    
    # Determine the package manager based on distribution
    package_manager = None
    update_cmd = None
    
    if distro in ['ubuntu', 'debian', 'raspbian']:
        package_manager = "apt-get"
        update_cmd = ["sudo", "apt-get", "update"]
    elif distro in ['centos', 'rhel', 'fedora', 'opensuse', 'sles']:
        if distro in ['fedora']:
            package_manager = "dnf"
        else:
            package_manager = "yum"  # Could also be zypper for opensuse
        update_cmd = None  # These systems often auto-update
    elif distro in ['arch', 'manjaro']:
        package_manager = "pacman"
        update_cmd = ["sudo", "pacman", "-Sy"]
    else:
        # For unknown distributions, try to detect the package manager
        if shutil.which("apt-get"):
            package_manager = "apt-get"
            update_cmd = ["sudo", "apt-get", "update"]
        elif shutil.which("yum"):
            package_manager = "yum"
            update_cmd = None
        elif shutil.which("dnf"):
            package_manager = "dnf"
            update_cmd = None
        elif shutil.which("pacman"):
            package_manager = "pacman"
            update_cmd = ["sudo", "pacman", "-Sy"]
        else:
            logger.warning("Could not identify package manager. Skipping system dependencies.")
            logger.info("Please manually install: python3-pip, python3-setuptools, python3-wheel, ffmpeg, gcc, make, cmake, python3-numpy, python3-scipy, python3-pyaudio")
            return

    # Install dependencies based on package manager
    if package_manager == "apt-get":
        packages = [
            "python3-pip", "python3-setuptools", "python3-wheel", 
            "ffmpeg", "gcc", "make", "cmake", "libasound2-dev", 
            "portaudio19-dev", "libsndfile1-dev", "wget", "git", 
            "unzip", "python3-numpy", "python3-scipy", "python3-pyaudio"
        ]
        cmd = ["sudo", "apt-get", "install", "-y"] + packages
    elif package_manager in ["yum", "dnf"]:
        packages = [
            "python3-pip", "python3-devel", "python3-setuptools", 
            "gcc", "gcc-c++", "make", "cmake", "ffmpeg", 
            "alsa-lib-devel", "portaudio-devel", "libsndfile-devel", 
            "wget", "git", "unzip", "numpy", "scipy", "pyaudio"
        ]
        cmd = ["sudo", package_manager, "install", "-y"] + packages
    elif package_manager == "pacman":
        packages = [
            "python-pip", "python-setuptools", "python-wheel", 
            "ffmpeg", "gcc", "make", "cmake", "alsa-lib", 
            "portaudio", "libsndfile", "wget", "git", 
            "unzip", "python-numpy", "python-scipy", "python-pyaudio"
        ]
        cmd = ["sudo", "pacman", "-S", "--noconfirm"] + packages
    else:
        logger.warning(f"Unknown package manager: {package_manager}")
        return

    try:
        if update_cmd:
            logger.info("Updating package database...")
            subprocess.run(update_cmd, check=True)
            logger.success("Updated package database")
        
        logger.info(f"Installing system dependencies using {package_manager}...")
        subprocess.run(cmd, check=True)
        logger.info("Successfully installed system dependencies")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install system dependencies: {e}")
        logger.warning("Continuing with Python dependencies...")

def create_requirements_txt():
    """Create requirements.txt file with all necessary dependencies"""
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.21.0",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "gradio>=3.44.0",
        "pydub>=0.25.1",
        "librosa>=0.10.0",
        "soundfile>=0.12.1",
        "PyPDF2>=3.0.0",
        "requests>=2.31.0",
        "python-multipart>=0.0.6",
        "huggingface-hub>=0.16.0",
        "sentencepiece>=0.1.99",
        "ddddocr>=1.4.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "TTS>=0.18.0",
        "chatterbox-tts>=0.1.0",
        "setuptools-rust>=1.7.0"
    ]
    
    with open("requirements.txt", "w") as f:
        for req in requirements:
            f.write(req + "\n")
    
    logger.info("Created requirements.txt file")

def install_python_dependencies():
    """Install Python dependencies in a virtual environment"""
    logger.info("Creating virtual environment for Python dependencies...")
    
    # Create virtual environment
    venv_path = Path("venv")
    if not venv_path.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        logger.info("Virtual environment created")
    else:
        logger.info("Virtual environment already exists")
    
    # Get the pip executable in the virtual environment
    if platform.system().lower() == "windows":
        venv_pip = venv_path / "Scripts" / "pip.exe"
    else:
        venv_pip = venv_path / "bin" / "pip"
    
    try:
        # Upgrade pip first
        logger.info("Upgrading pip in virtual environment...")
        subprocess.run([str(venv_pip), "install", "--upgrade", "pip"], check=True)
        logger.info("Pip upgraded successfully")
        
        # Install setuptools and wheel
        logger.info("Installing setuptools and wheel...")
        subprocess.run([str(venv_pip), "install", "setuptools", "wheel"], check=True)
        logger.info("Setuptools and wheel installed")
        
        # Install Rust (needed for some packages)
        logger.info("Installing Rust toolchain (needed for some packages)...")
        try:
            subprocess.run(["curl", "--proto", "'https://'", "--tlsv1.2", "-sSf", "https://sh.rustup.rs"], 
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # If rust is not installed, we'll install setuptools-rust which will handle it
        except:
            pass
        
        # Install requirements
        logger.info("Installing Python dependencies from requirements.txt...")
        subprocess.run([str(venv_pip), "install", "-r", "requirements.txt", "--break-system-packages"], check=True)
        logger.info("Python dependencies installed successfully")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install Python dependencies: {e}")
        return False

def create_arch_specific_configs():
    """Create architecture-specific configuration files"""
    # Create install_aur_helper.sh script
    aur_script_content = '''#!/bin/bash
# Script to install AUR helper (yay) on Arch Linux

echo "Checking for yay AUR helper..."

if ! command -v yay &> /dev/null; then
    echo "yay not found, installing..."
    
    # Install git if not present
    sudo pacman -S --noconfirm git
    
    # Clone and install yay
    cd /tmp
    git clone https://aur.archlinux.org/yay.git
    cd yay
    makepkg -si --noconfirm
    cd ..
    rm -rf yay
    
    echo "yay installed successfully!"
else
    echo "yay is already installed."
fi
'''
    
    scripts_dir = Path("scripts")
    if not scripts_dir.exists():
        scripts_dir.mkdir(exist_ok=True)
    
    with open("scripts/install_aur_helper.sh", "w") as f:
        f.write(aur_script_content)
    
    # Make the script executable
    os.chmod("scripts/install_aur_helper.sh", 0o755)
    
    logger.info("Created AUR helper installation script")

def main():
    print("Vietnamese Text-to-Speech Application Setup")
    print("=" * 50)
    
    # Check OS
    system, distro = check_os()
    
    # Create necessary directories
    create_directories()
    
    # Create requirements.txt
    create_requirements_txt()
    
    # Install Python dependencies in virtual environment
    success = install_python_dependencies()
    
    if success:
        logger.info("\nSetup completed successfully!")
        logger.info("\nTo run the application:")
        logger.info("1. Activate the virtual environment: source venv/bin/activate")
        logger.info("2. Run the API server: python -c \"import sys; sys.path.insert(0, '.'); from src.backend.api_server import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000, reload=False)\"")
        logger.info("\nTo run the frontend:")
        logger.info("1. cd src/frontend")
        logger.info("2. npm install")
        logger.info("3. npm run dev")
    else:
        logger.error("\nSetup failed. Please check the errors above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()