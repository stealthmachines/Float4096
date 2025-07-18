# project_root/setup_project.ps1
$ErrorActionPreference = "Stop"

# Determine project root
$PROJECT_ROOT = Get-Location

# Check for Git
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "Git not found. Please install Git."
    exit 1
}

# Check for Python 3.8 or higher
$PYTHON = "python"
if (-not (Get-Command $PYTHON -ErrorAction SilentlyContinue)) {
    Write-Error "Python not found. Please install Python 3.8 or higher."
    exit 1
}

$PYTHON_VERSION = & $PYTHON --version | Select-String -Pattern "[0-9]\.[0-9]" | ForEach-Object { $_.Matches.Value }
if ([version]$PYTHON_VERSION -lt [version]"3.8") {
    Write-Error "Python 3.8 or higher required. Found version: $PYTHON_VERSION"
    exit 1
}

# Check for pip
if (-not (Get-Command pip -ErrorAction SilentlyContinue)) {
    Write-Error "pip not found. Please install pip for Python 3."
    exit 1
}

# Create and activate virtual environment
$VENV_DIR = Join-Path $PROJECT_ROOT "venv"
if (-not (Test-Path $VENV_DIR)) {
    Write-Host "Creating virtual environment in $VENV_DIR..."
    & $PYTHON -m venv $VENV_DIR
}

# Activate virtual environment
$ACTIVATE_SCRIPT = Join-Path $VENV_DIR "Scripts\Activate.ps1"
if (Test-Path $ACTIVATE_SCRIPT) {
    . $ACTIVATE_SCRIPT
} else {
    Write-Error "Failed to activate virtual environment."
    exit 1
}

# Set PYTHONPATH
$env:PYTHONPATH = "$env:PYTHONPATH;$PROJECT_ROOT"
Write-Host "Set PYTHONPATH to include $PROJECT_ROOT"

# Install dependencies
Write-Host "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# Install float4096 package
Write-Host "Installing float4096 package..."
pip install .

# Function to run tests
function Run-Tests {
    Write-Host "Running tests..."
    if (-not (Test-Path "tests/test_float4096.py")) {
        Write-Error "tests/test_float4096.py not found."
        exit 1
    }
    pytest tests/test_float4096.py -v
}

# Function to run cosmo_fit.py
function Run-CosmoFit {
    Write-Host "Running cosmo_fit.py..."
    if (-not (Test-Path "cosmo_fit/cosmo_fit.py")) {
        Write-Error "cosmo_fit/cosmo_fit.py not found."
        exit 1
    }
    python cosmo_fit/cosmo_fit.py
}

# Parse command-line arguments
switch ($args[0]) {
    "test" { Run-Tests }
    "cosmo" { Run-CosmoFit }
    "both" { Run-Tests; Run-CosmoFit }
    default {
        Write-Host "Usage: .\setup_project.ps1 {test|cosmo|both}"
        Write-Host "  test: Run tests only"
        Write-Host "  cosmo: Run cosmo_fit.py only"
        Write-Host "  both: Run tests and cosmo_fit.py"
        exit 1
    }
}
