#!/bin/sh
# project_root/setup_project.sh

# Exit on any error
set -e

# Determine project root (works across shells and OS)
PROJECT_ROOT=$(pwd)

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for Python 3.8 or higher
PYTHON="python3"
if ! command_exists "$PYTHON"; then
    PYTHON="python"
    if ! command_exists "$PYTHON"; then
        echo "Error: Python not found. Please install Python 3.8 or higher."
        exit 1
    fi
fi

PYTHON_VERSION=$($PYTHON --version | grep -o '[0-9]\.[0-9]' | head -1)
if [ "$(echo "$PYTHON_VERSION < 3.8" | bc)" -eq 1 ]; then
    echo "Error: Python 3.8 or higher required. Found version: $PYTHON_VERSION"
    exit 1
fi

# Create and activate virtual environment
VENV_DIR="$PROJECT_ROOT/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    $PYTHON -m venv "$VENV_DIR"
fi

# Activate virtual environment (cross-platform)
if [ "$(uname)" = "Darwin" ] || [ "$(uname)" = "Linux" ]; then
    . "$VENV_DIR/bin/activate"
elif [ "$(expr substr $(uname -s) 1 5)" = "MINGW" ] || [ "$(expr substr $(uname -s) 1 10)" = "MSYS" ]; then
    . "$VENV_DIR/Scripts/activate"
else
    echo "Error: Unsupported operating system."
    exit 1
fi

# Set PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$PROJECT_ROOT"
echo "Set PYTHONPATH to include $PROJECT_ROOT"

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# Install float4096 package
echo "Installing float4096 package..."
pip install .

# Function to run tests
run_tests() {
    echo "Running tests..."
    pytest tests/test_float4096.py -v
}

# Function to run cosmo_fit.py
run_cosmo_fit() {
    echo "Running cosmo_fit.py..."
    python cosmo_fit/cosmo_fit.py
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
        echo "Usage: $0 {test|cosmo|both}"
        echo "  test: Run tests only"
        echo "  cosmo: Run cosmo_fit.py only"
        echo "  both: Run tests and cosmo_fit.py"
        exit 1
        ;;
esac
