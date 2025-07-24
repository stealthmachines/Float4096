#!/bin/bash
set -e

# Determine project root
PROJECT_ROOT=$(pwd)

# Check for Git
if ! command -v git &> /dev/null; then
    echo "Git not found. Please install Git."
    exit 1
fi

# Check for Python 3.8 or higher
PYTHON="python3"
if ! command -v "$PYTHON" &> /dev/null; then
    PYTHON="python"
    if ! command -v "$PYTHON" &> /dev/null; then
        echo "Python not found. Please install Python 3.8 or higher."
        exit 1
    fi
fi

# Parse Python version
PYTHON_VERSION=$("$PYTHON" --version 2>&1)
if [[ $PYTHON_VERSION =~ Python\ ([0-9]+)\.([0-9]+)\.([0-9]+) ]]; then
    MAJOR=${BASH_REMATCH[1]}
    MINOR=${BASH_REMATCH[2]}
    if (( MAJOR < 3 || (MAJOR == 3 && MINOR < 8) )); then
        echo "Python 3.8 or higher required. Found version: $PYTHON_VERSION"
        exit 1
    fi
else
    echo "Unable to parse Python version."
    exit 1
fi

# Check for pip
if ! command -v pip &> /dev/null; then
    echo "pip not found. Please install pip for Python 3."
    exit 1
fi

# Create and activate virtual environment
VENV_DIR="$PROJECT_ROOT/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# Activate virtual environment (handle Linux/macOS and Windows Git Bash/WSL)
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"
else
    echo "Failed to find virtual environment activation script."
    exit 1
fi

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
echo "Set PYTHONPATH to include $PROJECT_ROOT"

# Install dependencies
echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Install float4096 in editable mode
echo "Installing float4096 package..."
pip install -e .

# Function to run tests
run_tests() {
    echo "Running tests..."
    if [ ! -d "tests" ]; then
        echo "tests/ directory not found."
        exit 1
    fi
    pytest tests -v
}

# Function to run cosmo_fit.py
run_cosmo_fit() {
    echo "Running cosmo_fit.py..."
    COSMO_PATH="cosmo_fit/cosmo_fit.py"
    if [ ! -f "$COSMO_PATH" ]; then
        echo "$COSMO_PATH not found."
        exit 1
    fi
    python "$COSMO_PATH"
}

# Parse command-line arguments
case "$1" in
    test)
        run_tests
        ;;
    cosmo)
        run_cosmo_fit
        ;;
    both)
        run_tests
        run_cosmo_fit
        ;;
    *)
        echo "Usage: ./setup_project.sh {test|cosmo|both}"
        echo "  test:   Run tests only"
        echo "  cosmo:  Run cosmo_fit.py only"
        echo "  both:   Run tests and cosmo_fit.py"
        exit 1
        ;;
esac
