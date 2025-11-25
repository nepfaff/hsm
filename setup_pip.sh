#!/bin/bash
# HSM Setup Script (Pip/Venv Version)
# This script sets up the HSM environment using Python virtual environments and pip
#
# Usage:
#   ./setup_pip.sh                    # Run with GPU (CUDA 12.1) support
#   ./setup_pip.sh --cpu             # Run with CPU-only PyTorch
#   ./setup_pip.sh --force           # Force recreation of virtual environment

set -e  # Exit on any error

REPO_NAME="${REPO_NAME:-hsm}"
VENV_NAME=".venv"
USE_CPU=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1 \n"
}

# Progress bar function
show_progress() {
    local current=$1
    local total=$2
    local width=50
    local percentage=$((current * 100 / total))
    local completed=$((current * width / total))

    printf "\r${BLUE}[PROGRESS]${NC} ["
    for ((i=1; i<=completed; i++)); do printf "="; done
    for ((i=completed+1; i<=width; i++)); do printf " "; done
    printf "] %d%% (%d/%d)" $percentage $current $total
}

# Setup step tracker
TOTAL_STEPS=8
CURRENT_STEP=0
DOWNLOAD_FAILED=false

next_step() {
    CURRENT_STEP=$((CURRENT_STEP + 1))
    show_progress $CURRENT_STEP $TOTAL_STEPS
    echo -e "\n${BLUE}[STEP $CURRENT_STEP/$TOTAL_STEPS]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Validate .env file configuration
validate_env_file() {
    ensure_project_root
    log_info "Validating .env file configuration..."

    # Check if .env file exists
    if [ ! -f ".env" ]; then
        log_error ".env file not found!"
        log_info "Please create a .env file based on .env.example:"
        log_info "  cp .env.example .env"
        log_info "  # Then edit .env with your API keys"
        return 1
    fi

    # Check OPENAI_API_KEY
    OPENAI_KEY=$(grep "^OPENAI_API_KEY=" .env | cut -d'=' -f2 | sed 's/^"//' | sed 's/"$//')
    if [ -z "$OPENAI_KEY" ] || [ "$OPENAI_KEY" = "your_openai_api_key_here" ]; then
        log_error "OPENAI_API_KEY is not set or using default placeholder!"
        log_info "Please set your actual OpenAI API key in .env file"
        log_info "Get your key from: https://platform.openai.com/api-keys"
        return 1
    fi

    # Check HF_TOKEN (required for downloads)
    HF_TOKEN_VALUE=$(grep "^HF_TOKEN=" .env | cut -d'=' -f2 | sed 's/^"//' | sed 's/"$//')
    if [ -z "$HF_TOKEN_VALUE" ] || [ "$HF_TOKEN_VALUE" = "your_huggingface_token_here" ]; then
        log_error "HF_TOKEN is not set or using default placeholder!"
        log_info "Please set your actual Hugging Face token in .env file"
        log_info "Get your token from: https://huggingface.co/settings/tokens"
        return 1
    fi

    log_success "Environment configuration validated successfully \n"
    return 0
}

# Check system requirements
check_requirements() {
    ensure_project_root
    log_info "Checking system requirements..."

    local missing_tools=()

    # Check for curl or wget
    if ! command_exists curl && ! command_exists wget; then
        missing_tools+=("curl or wget")
    fi

    # Check for git
    if ! command_exists git; then
        missing_tools+=("git")
    fi

    # Check for unzip
    if ! command_exists unzip; then
        missing_tools+=("unzip")
    fi

    # Check for Python 3.10 or 3.11
    PYTHON_CMD=""
    if command_exists python3.11; then
        PYTHON_CMD="python3.11"
    elif command_exists python3.10; then
        PYTHON_CMD="python3.10"
    elif command_exists python3; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
        if [ "$PYTHON_VERSION" = "3.10" ] || [ "$PYTHON_VERSION" = "3.11" ]; then
            PYTHON_CMD="python3"
        else
            log_error "Python 3.10 or 3.11 is required, but found Python $PYTHON_VERSION"
            missing_tools+=("python3.10 or python3.11")
        fi
    else
        missing_tools+=("python3.10 or python3.11")
    fi

    # Check for pip
    if ! command_exists pip3 && ! command_exists pip; then
        missing_tools+=("pip")
    fi

    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please install them and run this script again."
        log_info "On Ubuntu/Debian:"
        log_info "  sudo apt-get update"
        log_info "  sudo apt-get install python3.10 python3.10-venv python3-pip curl git unzip"
        log_info "  # Or for Python 3.11: sudo apt-get install python3.11 python3.11-venv"
        log_info "  sudo apt-get install libspatialindex-dev  # Required for rtree package"
        log_info "  sudo apt-get install git-lfs  # Required for HSSD model downloads"
        exit 1
    fi

    # Check for libspatialindex (required for rtree)
    if ! ldconfig -p | grep -q libspatialindex; then
        log_warning "libspatialindex not found. This is required for the rtree package."
        log_info "Install with: sudo apt-get install libspatialindex-dev"
    fi

    log_success "All required tools are available \n"
}

# Setup virtual environment
setup_environment() {
    log_info "Setting up Python virtual environment..."

    ensure_project_root

    # Check if venv already exists
    if [ -d "$VENV_NAME" ]; then
        log_info "Virtual environment '$VENV_NAME' already exists"
        read -rp "Do you want to recreate it? (y/N): " REPLY
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Removing existing virtual environment..."
            rm -rf "$VENV_NAME"
        else
            log_info "Using existing environment"
            return 0
        fi
    fi

    # Create virtual environment
    log_info "Creating virtual environment '$VENV_NAME'..."
    if $PYTHON_CMD -m venv "$VENV_NAME"; then
        log_success "Virtual environment created successfully \n"
    else
        log_error "Failed to create virtual environment"
        log_info "Make sure python venv package is installed:"
        log_info "  sudo apt-get install python3.10-venv  # For Python 3.10"
        log_info "  sudo apt-get install python3.11-venv  # For Python 3.11"
        exit 1
    fi
}

# Install Python packages
install_packages() {
    log_info "Installing Python packages..."

    ensure_project_root

    # Activate virtual environment
    source "$VENV_NAME/bin/activate"

    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip

    # Install PyTorch based on CPU/GPU preference
    if [ "$USE_CPU" = true ]; then
        log_info "Installing PyTorch (CPU-only version)..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    else
        log_info "Installing PyTorch with CUDA 12.1 support..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    fi

    # Install requirements from requirements.txt
    log_info "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt

    # Install CLIP from GitHub
    log_info "Installing OpenAI CLIP from GitHub..."
    pip install git+https://github.com/openai/CLIP.git

    log_success "Python packages installed successfully \n"
}

# Download file with progress
download_file() {
    local url="$1"
    local output="$2"

    log_info "Downloading: $url"

    if command_exists wget; then
        wget --no-check-certificate -O "$output" "$url"
    elif command_exists curl; then
        curl -L -k -o "$output" "$url"
    else
        log_error "Neither curl nor wget available"
        exit 1
    fi
}

# Download HSSD models from Hugging Face
download_hssd_models() {
    log_info "Downloading HSSD models from Hugging Face..."

    mkdir -p data
    cd data

    # Store original directory
    ORIGINAL_DIR="$(pwd)"

    # Download HSSD models (this is a large download ~72GB)
    # Check if HSSD models are already downloaded (has .git directory)
    if [ ! -d "hssd-models/.git" ]; then
        log_info "Downloading HSSD models (~72GB)... This may take a while!"

        # Check if git-lfs is available
        if ! command_exists git-lfs; then
            log_error "git-lfs not found. Please install it:"
            log_info "  sudo apt-get install git-lfs"
            log_info "  git lfs install"
            return 1
        fi

        # Ensure git-lfs is initialized
        git lfs install

        # Configure git credential helper with HF token for HTTPS cloning
        if [ -f "../.env" ] && grep -q "HF_TOKEN=" ../.env; then
            HF_TOKEN=$(grep "HF_TOKEN=" ../.env | cut -d'=' -f2 | sed 's/^"//' | sed 's/"$//')
            if [ -n "$HF_TOKEN" ] && [ "$HF_TOKEN" != "your_huggingface_token_here" ]; then
                log_info "Using Hugging Face token from .env file for git authentication"
                # Configure git to use token for this operation
                export GIT_ASKPASS_TOKEN="$HF_TOKEN"
            else
                log_warning "HF_TOKEN not set properly in .env file"
                log_info "You may need to enter credentials manually or set up git credentials"
            fi
        else
            log_warning ".env file not found or HF_TOKEN not set"
            log_info "You may need to enter credentials manually"
        fi

        # Clone the dataset using git with authentication
        # Try HTTPS clone with token in URL if available
        if [ -n "$HF_TOKEN" ] && [ "$HF_TOKEN" != "your_huggingface_token_here" ]; then
            log_info "Attempting authenticated git clone..."
            if ! git clone https://user:${HF_TOKEN}@huggingface.co/datasets/hssd/hssd-models; then
                log_warning "Authenticated git clone failed, trying Python download..."
                CLONE_FAILED=true
            fi
        else
            log_info "Attempting git clone (may require authentication)..."
            if ! git clone https://huggingface.co/datasets/hssd/hssd-models; then
                log_warning "Git clone failed, trying Python download..."
                CLONE_FAILED=true
            fi
        fi

        # Fallback to Python-based download if git clone failed
        if [ "$CLONE_FAILED" = true ]; then
            log_info "Using Python huggingface_hub library for download..."
            # Activate venv to use huggingface_hub
            source "../$VENV_NAME/bin/activate"

            if [ ! -d "hssd-models/objects/0" ]; then
                python3 -c "
from huggingface_hub import snapshot_download
import os

token = os.getenv('HF_TOKEN') or '$HF_TOKEN' or None
print(f'Downloading with token: {\"Yes\" if token else \"No\"}')
snapshot_download(
    repo_id='hssd/hssd-models',
    repo_type='dataset',
    local_dir='hssd-models',
    token=token
)
" || {
                    log_error "Failed to obtain HSSD models via git and Python download"
                    log_info "Please check:"
                    log_info "  1. You accepted the license at https://huggingface.co/datasets/hssd/hssd-models"
                    log_info "  2. Your HF_TOKEN is valid and has access"
                    log_info "  3. You have enough disk space (~72GB)"
                    return 1
                }
            fi
        fi

        log_success "HSSD models downloaded \n"
    else
        log_success "HSSD models repository already cloned \n"
    fi

    cd ..
}

# Helper function to ensure we're back in project root
ensure_project_root() {
    # Change to the directory where setup_pip.sh is located
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR"
}

# Download decomposed models from Hugging Face
download_decomposed_models() {
    # Check if decomposed models already exist
    if [ -d "data/hssd-models/objects/decomposed" ]; then
        log_success "Decomposed models already downloaded \n"
        return 0
    fi

    log_info "Downloading decomposed models from Hugging Face..."
    mkdir -p data/hssd-models/objects

    # Activate venv to use huggingface-hub
    source "$VENV_NAME/bin/activate"

    # Prepare token argument if available
    HF_TOKEN_ARG=""
    if [ -f ".env" ] && grep -q "HF_TOKEN=" .env; then
        HF_TOKEN=$(grep "HF_TOKEN=" .env | cut -d'=' -f2 | sed 's/^"//' | sed 's/"$//')
        if [ -n "$HF_TOKEN" ] && [ "$HF_TOKEN" != "your_huggingface_token_here" ]; then
            HF_TOKEN_ARG="--token $HF_TOKEN"
        fi
    fi

    # Download decomposed models using Python API (more reliable than CLI)
    python3 -c "
from huggingface_hub import snapshot_download
import os

token = os.getenv('HF_TOKEN') or '$HF_TOKEN' or None
snapshot_download(
    repo_id='hssd/hssd-hab',
    repo_type='dataset',
    allow_patterns='objects/decomposed/**/*_part_*.glb',
    ignore_patterns='objects/decomposed/**/*_part.*.glb',
    local_dir='data/hssd-models',
    token=token
)
" || {
        log_error "Failed to download decomposed models"
        return 1
    }

    # Verify download
    if [ -d "data/hssd-models/objects/decomposed" ]; then
        log_success "Decomposed models downloaded \n"
    else
        log_warning "Decomposed models not found in expected location"
    fi
}

# Download data from GitHub releases
download_github_data() {
    # Check if data directory already has required contents
    if [ -d "data/motif_library" ] && [ -d "data/preprocessed" ]; then
        log_success "Core data already exists locally \n"
        return 0
    fi

    log_info "Processing data from GitHub releases..."
    mkdir -p data

    # Check if zip file already exists (look in multiple locations)
    local data_zip=""
    local found_local=false

    # First, check in project root
    if [ -f "data.zip" ]; then
        data_zip="data.zip"
        found_local=true
        log_success "Using local data.zip file from project root \n"
    # Then check inside data/ directory
    elif [ -f "data/data.zip" ]; then
        data_zip="data/data.zip"
        found_local=true
        log_success "Using local data.zip file from data/ directory \n"
    fi

    if [ "$found_local" = true ]; then
        # Check if the file is actually a valid zip
        if file "$data_zip" | grep -q "Zip archive"; then
            # Extract data
            log_info "Extracting data..."
            unzip -q "$data_zip"
            rm "$data_zip"
            log_success "GitHub data extracted successfully \n"
        else
            file_type=$(file "$data_zip" | head -1)
            log_error "Local data.zip is not a valid zip archive: $file_type"
            log_info "Please download a valid data.zip file from:"
            log_info "https://github.com/3dlg-hcvc/${REPO_NAME}/releases"
            return 1
        fi
    else
        log_info "No local data.zip found, downloading from GitHub..."
        local data_zip="data.zip"
        local data_url="https://github.com/3dlg-hcvc/${REPO_NAME}/releases/latest/download/data.zip"
        if ! download_file "$data_url" "$data_zip"; then
            log_warning "Failed to download data.zip. Please download manually:"
            log_info "wget --no-check-certificate -O data.zip '$data_url'"
            log_info "Then extract: unzip data.zip && rm data.zip"
            return 1
        fi

        # Check and extract the downloaded file
        if file "$data_zip" | grep -q "Zip archive"; then
            log_info "Extracting data..."
            unzip -q "$data_zip"
            rm "$data_zip"
            log_success "GitHub data extracted successfully \n"
        else
            file_type=$(file "$data_zip" | head -1)
            log_error "Downloaded data.zip is not a valid zip archive: $file_type"
            rm "$data_zip"
            return 1
        fi
    fi
}

# Download support surface data
download_support_surfaces() {
    # Check if support surface data already exists
    if [ -d "data/hssd-models/support-surfaces" ]; then
        log_success "Support surface data already exists locally \n"
        return 0
    fi

    log_info "Processing support surface data..."
    mkdir -p data/hssd-models

    # Check if zip file already exists (look in multiple locations)
    local support_zip=""
    local found_local=false

    # First, check in project root
    if [ -f "support-surfaces.zip" ]; then
        support_zip="support-surfaces.zip"
        found_local=true
        log_success "Using local support-surfaces.zip file from project root \n"
    # Then check inside data/ directory
    elif [ -f "data/support-surfaces.zip" ]; then
        support_zip="data/support-surfaces.zip"
        found_local=true
        log_success "Using local support-surfaces.zip file from data/ directory \n"
    fi

    if [ "$found_local" = true ]; then
        # Check if the file is actually a valid zip
        if file "$support_zip" | grep -q "Zip archive"; then
            # Extract to hssd-models
            log_info "Extracting support surfaces..."
            unzip -q "$support_zip" -d data/hssd-models
            rm "$support_zip"
            log_success "Support surface data extracted successfully \n"
        else
            file_type=$(file "$support_zip" | head -1)
            log_error "Local support-surfaces.zip is not a valid zip archive: $file_type"
            log_info "Please download a valid support-surfaces.zip file from:"
            log_info "https://github.com/3dlg-hcvc/${REPO_NAME}/releases"
            return 1
        fi
    else
        log_info "No local support-surfaces.zip found, downloading from GitHub..."
        local support_zip="support-surfaces.zip"
        local support_url="https://github.com/3dlg-hcvc/${REPO_NAME}/releases/latest/download/support-surfaces.zip"
        if ! download_file "$support_url" "$support_zip"; then
            log_warning "Failed to download support-surfaces.zip. Please download manually:"
            log_info "wget --no-check-certificate -O support-surfaces.zip '$support_url'"
            log_info "Then extract: unzip support-surfaces.zip -d data/hssd-models && rm support-surfaces.zip"
            return 1
        fi

        # Check and extract the downloaded file
        if file "$support_zip" | grep -q "Zip archive"; then
            log_info "Extracting support surfaces..."
            unzip -q "$support_zip" -d data/hssd-models
            rm "$support_zip"
            log_success "Support surface data extracted successfully \n"
        else
            file_type=$(file "$support_zip" | head -1)
            log_error "Downloaded support-surfaces.zip is not a valid zip archive: $file_type"
            rm "$support_zip"
            return 1
        fi
    fi
}

# Verify setup
verify_setup() {
    ensure_project_root
    log_info "Verifying setup..."

    local issues=0

    # Check virtual environment
    if [ ! -d "$VENV_NAME" ]; then
        log_error "Virtual environment '$VENV_NAME' not found"
        issues=$((issues + 1))
    else
        log_info "Virtual environment '$VENV_NAME' exists"

        # Check if key packages are installed
        source "$VENV_NAME/bin/activate"
        if ! python -c "import torch; import trimesh; import openai" 2>/dev/null; then
            log_error "Some key Python packages are missing"
            issues=$((issues + 1))
        else
            log_info "Key Python packages are installed"
        fi
    fi

    # Check data directory structure
    if [ ! -d "data" ]; then
        log_error "data directory not found"
        issues=$((issues + 1))
    else
        local required_paths=(
            "data/hssd-models/objects/9"
            "data/hssd-models/objects/x"
            "data/hssd-models/objects/decomposed"
            "data/hssd-models/support-surfaces"
            "data/motif_library/meta_programs"
            "data/preprocessed"
        )

        for path in "${required_paths[@]}"; do
            if [ ! -d "$path" ]; then
                log_error "Required path not found: $path"
                issues=$((issues + 1))
            fi
        done

        if [ $issues -eq 0 ]; then
            log_info "Data directory structure is correct"
        fi
    fi

    if [ $issues -gt 0 ]; then
        log_info "Setup verification found $issues issue(s)"
        return 1
    else
        log_success "Setup verification passed \n"
        return 0
    fi
}

# Main setup function
main() {
    echo -e "\n${BLUE}=======================================${NC}"
    log_info "Starting HSM setup (Pip/Venv version)..."
    log_info "========================================"
    echo ""

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --cpu)
                USE_CPU=true
                log_info "CPU-only mode enabled"
                shift
                ;;
            --force)
                # Force will be handled in setup_environment
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --cpu        Install CPU-only version of PyTorch"
                echo "  --force      Force recreation of virtual environment"
                echo "  --help       Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    # Run setup steps
    next_step "Checking system requirements..."
    check_requirements

    next_step "Validating environment configuration..."
    ensure_project_root
    if ! validate_env_file; then
        log_error "Environment configuration validation failed!"
        log_info "Please fix the issues above and run the setup script again."
        exit 1
    fi

    next_step "Setting up Python virtual environment..."
    ensure_project_root
    setup_environment

    next_step "Installing Python packages..."
    ensure_project_root
    install_packages

    next_step "Downloading HSSD models (~72GB)..."
    log_info "HSM by default uses HSSD models for 3D model retrieval."
    ensure_project_root
    if ! download_hssd_models; then
        DOWNLOAD_FAILED=true
    fi
    ensure_project_root
    if ! download_decomposed_models; then
        DOWNLOAD_FAILED=true
    fi

    next_step "Downloading preprocessed data from GitHub..."
    ensure_project_root
    if ! download_github_data; then
        log_warning "Preprocessed data download failed. Please download it manually to the root directory then run setup_pip.sh again."
        DOWNLOAD_FAILED=true
    fi

    next_step "Downloading support surface data..."
    ensure_project_root
    if ! download_support_surfaces; then
        log_warning "Support surface download failed. Please download it manually to the root directory then run setup_pip.sh again."
        DOWNLOAD_FAILED=true
    fi

    next_step "Verifying setup..."
    ensure_project_root
    if verify_setup; then
        show_progress $TOTAL_STEPS $TOTAL_STEPS
        echo -e "\n"

        if [ "$DOWNLOAD_FAILED" = true ]; then
            log_warning "HSM setup failed with download failures!"
            log_info ""
            log_info "Some downloads failed. Please download the missing data manually as above."
            log_info "Then run the setup script again."
            log_info "If problem persists, follow the manual setup instructions in INSTALL_PIP.md."
            log_info ""
            exit 1
        else
            log_success "HSM setup completed successfully!"
            log_info ""
            log_info "To generate a scene with a description using HSM, run the following commands:"
            log_info "1. source $VENV_NAME/bin/activate"
            log_info "2. python main.py -d 'your description'"
            log_info "The default output directory is results/single_run"
            log_info "For more details, please refer to INSTALL_PIP.md."
            log_info ""
        fi
    else
        echo -e "\n"
        log_warning "Setup failed with issues. Please review the errors above."
        exit 1
    fi
}

# Run main function
main "$@"
