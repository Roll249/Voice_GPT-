# Setup Guide for Vietnamese TTS Application

This guide explains the different setup options available for the Vietnamese Text-to-Speech application.

## Overview

The application provides two different setup scripts optimized for different operating systems:

1. **General Setup** (`src/cli/setup.py`): Designed for Ubuntu, Debian, and other Debian-based distributions
2. **Arch Linux Setup** (`src/cli/arch_setup.py` or `./arch_setup.sh`): Specifically optimized for Arch Linux systems

## System-Specific Optimizations

### Arch Linux Setup (`arch_setup.py`)

The Arch Linux setup script includes optimizations specific to Arch:

- Uses `pacman` for system package management
- Installs Arch-specific packages and dependencies
- Includes AUR helper setup (yay/paru) for additional packages
- Optimized package installation order for Arch Linux systems
- Checks for Arch Linux compatibility before proceeding

### General Setup (`setup.py`)

The general setup script targets Debian-based systems:

- Uses `apt` for system package management
- Compatible with Ubuntu, Debian, Mint, and similar distributions
- Standard package installation approach

## How to Choose

- **Arch Linux or Arch-based systems (Manjaro, EndeavourOS, etc.)**: Use `./arch_setup.sh` or `python src/cli/arch_setup.py`
- **Debian-based systems (Ubuntu, Debian, Linux Mint, etc.)**: Use `python src/cli/setup.py`

## Usage Examples

### For Arch Linux:

```bash
# Using the convenience script
./arch_setup.sh --all

# Or directly calling the Python script
python src/cli/arch_setup.py --all
```

### For Debian-based systems:

```bash
python src/cli/setup.py --all
```

## Available Options

Both setup scripts support these command-line arguments:

- `--all`: Run complete setup (dependencies + backend + GPU check)
- `--install-deps`: Install system and Python dependencies only
- `--setup-backend`: Setup backend components only
- `--check-gpu`: Check GPU availability only

Example:
```bash
# For Arch Linux
python src/cli/arch_setup.py --install-deps --check-gpu

# For Debian-based systems  
python src/cli/setup.py --install-deps --check-gpu
```

## What Each Script Does

Both scripts perform these common tasks:
1. Creates necessary project directories
2. Installs system dependencies via native package manager
3. Installs Python dependencies via pip
4. Sets up the backend API server
5. Creates requirements.txt file
6. Performs GPU availability checks

The key difference is in how they handle system package installation, which is optimized per distribution.