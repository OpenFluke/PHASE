#!/bin/bash

# Check and install python3-venv if missing
if ! dpkg -l | grep -q python3-venv; then
    echo "Installing python3-venv..."
    sudo apt update
    sudo apt install -y python3-venv
fi

# Set the virtual environment directory
VENV_DIR="myenv"

# Remove any existing incomplete virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "Removing incomplete virtual environment..."
    rm -rf "$VENV_DIR"
fi

# Create a new virtual environment
echo "Creating a virtual environment..."
python3 -m venv $VENV_DIR || { echo "Failed to create a virtual environment."; exit 1; }

# Activate the virtual environment
source $VENV_DIR/bin/activate || { echo "Failed to activate virtual environment."; exit 1; }

# Upgrade pip and install required packages
pip install --upgrade pip || { echo "Failed to upgrade pip."; exit 1; }
pip install python-docx || { echo "Failed to install required packages."; exit 1; }

# Check if the Python script exists
SCRIPT_NAME="generate_doc.py"
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "$SCRIPT_NAME does not exist. Please ensure it is in the same directory as this script."
    deactivate
    exit 1
fi

# Run the Python script
python3 $SCRIPT_NAME || { echo "Failed to execute Python script."; deactivate; exit 1; }

# Deactivate the virtual environment
deactivate || true
