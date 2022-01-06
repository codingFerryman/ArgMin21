#!/usr/bin/bash

# Specify the project path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Pull submodules
git submodule update --init --recursive

# Upgrade the building tools
pip install --upgrade pip setuptools wheel

# Install the requirements
pip install -r $SCRIPT_DIR/requirements.txt
