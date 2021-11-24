
# Specify the project path
$SCRIPT_DIR=$ExecutionContext.SessionState.Path.CurrentFileSystemLocation

# Set up and activate virtual environment
if (Test-Path -Path $SCRIPT_DIR\venv) {
    Write-Host "Virtual environment already exists."
} else {
    python -m venv $SCRIPT_DIR\venv
}
$ACTIVATE_VENV=Join-Path $SCRIPT_DIR \venv\Scripts\Activate.ps1
. $ACTIVATE_VENV

# Sync submodules (incl. data)
git submodule update --init --recursive

# Upgrade and install packages
python -m pip install --upgrade pip setuptools wheel
pip install -r $SCRIPT_DIR\requirements.txt
