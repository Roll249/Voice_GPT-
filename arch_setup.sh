#!/bin/bash
# Arch Linux Setup Script for Vietnamese TTS Application

echo "Vietnamese Text-to-Speech Application Setup for Arch Linux"
echo "=========================================================="

# Check if running on Arch Linux
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$ID" != "arch" && "$ID_LIKE" != *"arch"* ]]; then
        echo "Warning: This script is designed for Arch Linux. Current system: $NAME"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo "Cannot determine OS. This script is designed for Arch Linux."
    exit 1
fi

# Make sure we're in the workspace directory
cd /workspace || exit 1

# Run the Arch setup script
python3 src/cli/arch_setup.py "$@"