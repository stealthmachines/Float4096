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

# Parse version
$PYTHON_VERSION_OUTPUT = & $PYTHON --version
if ($PYTHON_VERSION_OUTPUT -match "Python (\d+)\.(\d+)\.(\d+)") {
    $MAJOR = [int]$matches[1]
    $MINOR = [int]$matches[2]
    if ($MAJOR -lt 3 -or ($MAJOR -eq 3 -and $MINOR -lt 8)) {
        Write-Error "Python 3.8 or higher required. Found version: $PYTHON_VERSION_OUTPUT"
        exit 1
    }
} else {
    Write-Error "Unable to parse Python version."
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
$env:PYTHONPATH = "$PROJECT_ROOT;$env:PYTHONPATH"
Write-Host "Set PYTHONPATH to include $PROJECT_ROOT"

# Install dependencies
Write-Host "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Install float4096 in editable mode
Write-Host "Installing float4096 package..."
pip install -e .

# Function to run tests
function Run-Tests {
    Write-Host "Running tests..."
    if (-not (Test-Path "tests")) {
        Write-Error "tests/ directory not found."
        exit 1
    }
    pytest tests -v
}

# Function to run cosmo_fit.py
function Run-CosmoFit {
    Write-Host "Running cosmo_fit.py..."
    $COSMO_PATH = "cosmo_fit/cosmo_fit.py"
    if (-not (Test-Path $COSMO_PATH)) {
        Write-Error "$COSMO_PATH not found."
        exit 1
    }
    python $COSMO_PATH
}

# Parse command-line arguments
switch ($args[0]) {
    "test" { Run-Tests }
    "cosmo" { Run-CosmoFit }
    "both"  { Run-Tests; Run-CosmoFit }
    default {
        Write-Host "Usage: .\setup_project.ps1 {test|cosmo|both}"
        Write-Host "  test:   Run tests only"
        Write-Host "  cosmo:  Run cosmo_fit.py only"
        Write-Host "  both:   Run tests and cosmo_fit.py"
        exit 1
    }
}
