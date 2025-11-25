# HSM Installation Guide (Pip/Venv)

This guide provides instructions for installing HSM (Hierarchical Scene Motifs) using Python virtual environments and pip instead of conda/mamba.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Installation](#detailed-installation)
- [Manual Installation](#manual-installation)
- [Troubleshooting](#troubleshooting)
- [Usage](#usage)

## Prerequisites

### System Requirements

- **Operating System**: Ubuntu 22.04 LTS (tested) or similar Linux distribution
- **Python**: Python 3.10 or 3.11
- **Disk Space**: ~75GB for HSSD models and data
- **Memory**: 16GB RAM recommended
- **GPU** (optional): NVIDIA GPU with CUDA 12.1 support for faster processing

### Required System Packages

Install the following system packages:

```bash
sudo apt-get update
sudo apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    git-lfs \
    curl \
    unzip \
    libspatialindex-dev

# Or for Python 3.11:
# sudo apt-get install python3.11 python3.11-venv
```

**Package Explanations:**
- `python3.10` and `python3.10-venv`: Python 3.10 with virtual environment support (or use 3.11)
- `git` and `git-lfs`: Git with Large File Storage for downloading HSSD models
- `curl` or `wget`: For downloading data files
- `unzip`: For extracting downloaded archives
- `libspatialindex-dev`: System library required for the rtree package (spatial indexing)

### Initialize Git LFS

```bash
git lfs install
```

### Required API Keys

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Then edit `.env` and add your API keys:

```
OPENAI_API_KEY=your_actual_openai_api_key
HF_TOKEN=your_actual_huggingface_token
```

**Where to get API keys:**
- **OpenAI API Key**: https://platform.openai.com/api-keys
- **Hugging Face Token**: https://huggingface.co/settings/tokens

**Important**: You must also accept the license for HSSD models on Hugging Face before downloading:
- Visit: https://huggingface.co/datasets/hssd/hssd-models
- Click "Agree and access repository"

## Quick Start

### Automated Setup (Recommended)

Run the automated setup script:

```bash
# Make the script executable
chmod +x setup_pip.sh

# For GPU (CUDA 12.1) support
./setup_pip.sh

# For CPU-only installation
./setup_pip.sh --cpu

# Force recreation of virtual environment
./setup_pip.sh --force
```

The script will:
1. Check system requirements
2. Validate your `.env` file
3. Create a Python virtual environment (`.venv`)
4. Install PyTorch with CUDA 12.1 (or CPU-only)
5. Install all Python dependencies
6. Download HSSD models (~72GB)
7. Download preprocessed data
8. Download support surface data
9. Verify the installation

After successful installation:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Generate a scene
python main.py -d "cozy living room with a sofa and coffee table"
```

## Detailed Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/3dlg-hcvc/hsm.git
cd hsm
```

### Step 2: Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env and add your API keys
nano .env  # or use your preferred editor
```

### Step 3: Run Setup Script

```bash
chmod +x setup_pip.sh
./setup_pip.sh
```

The installation process has 8 steps:
1. **System requirements check**: Validates all required tools are installed
2. **Environment validation**: Checks `.env` file for valid API keys
3. **Virtual environment setup**: Creates `.venv` directory
4. **Package installation**: Installs PyTorch and all dependencies
5. **HSSD models download**: Downloads ~72GB of 3D models (may take hours)
6. **Preprocessed data download**: Downloads CLIP embeddings and object mappings
7. **Support surfaces download**: Downloads geometric constraint data
8. **Verification**: Validates complete installation

### Step 4: Activate and Test

```bash
source .venv/bin/activate
python main.py -d "modern bedroom" --output results/test_bedroom
```

## Manual Installation

If you prefer to install manually or the automated script fails:

### 1. Create Virtual Environment

```bash
# Using Python 3.10
python3.10 -m venv .venv

# Or using Python 3.11
# python3.11 -m venv .venv

source .venv/bin/activate
```

### 2. Upgrade pip

```bash
pip install --upgrade pip
```

### 3. Install PyTorch

**For GPU (CUDA 12.1):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

### 5. Download Data

#### HSSD Models (~72GB)

```bash
# Make sure git-lfs is installed and initialized
git lfs install

# Login to Hugging Face (use token from .env)
huggingface-cli login --token YOUR_HF_TOKEN

# Clone HSSD models
cd data
git clone https://huggingface.co/datasets/hssd/hssd-models
cd ..
```

#### Decomposed Models

```bash
huggingface-cli download hssd/hssd-hab \
    --repo-type=dataset \
    --include "objects/decomposed/**/*_part_*.glb" \
    --exclude "objects/decomposed/**/*_part.*.glb" \
    --local-dir "data/hssd-models"
```

Or use Python:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='hssd/hssd-hab',
    repo_type='dataset',
    allow_patterns='objects/decomposed/**/*_part_*.glb',
    ignore_patterns='objects/decomposed/**/*_part.*.glb',
    local_dir='data/hssd-models',
    token='YOUR_HF_TOKEN'
)
```

#### Preprocessed Data

Download from GitHub releases:

```bash
wget --no-check-certificate -O data.zip \
    https://github.com/3dlg-hcvc/hsm/releases/latest/download/data.zip
unzip data.zip
rm data.zip
```

#### Support Surfaces

```bash
wget --no-check-certificate -O support-surfaces.zip \
    https://github.com/3dlg-hcvc/hsm/releases/latest/download/support-surfaces.zip
unzip support-surfaces.zip -d data/hssd-models
rm support-surfaces.zip
```

### 6. Verify Installation

Check that all required directories exist:

```bash
ls -la data/
# Should show:
# - hssd-models/objects/9
# - hssd-models/objects/x
# - hssd-models/objects/decomposed
# - hssd-models/support-surfaces
# - motif_library/meta_programs
# - preprocessed
```

Test Python imports:

```python
python -c "import torch; import trimesh; import openai; import clip; print('All imports successful')"
```

## Troubleshooting

### Missing libspatialindex

**Error:** `OSError: Could not find libspatialindex_c library`

**Solution:**
```bash
sudo apt-get install libspatialindex-dev
pip uninstall rtree
pip install rtree
```

### Python 3.10/3.11 Not Found

**Error:** `python3.10: command not found` or `python3.11: command not found`

**Solution for Ubuntu 22.04:**
```bash
sudo apt-get update
# For Python 3.10 (usually pre-installed on Ubuntu 22.04)
sudo apt-get install python3.10 python3.10-venv

# Or for Python 3.11
sudo apt-get install python3.11 python3.11-venv
```

**Solution for Ubuntu 20.04:**
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
# Install either version
sudo apt-get install python3.10 python3.10-venv python3.10-dev
# or
sudo apt-get install python3.11 python3.11-venv python3.11-dev
```

### Git LFS Download Issues

**Error:** Large files not downloading or showing as text pointers

**Solution:**
```bash
# Install and initialize git-lfs
sudo apt-get install git-lfs
git lfs install

# For existing clone, fetch LFS files
cd data/hssd-models
git lfs pull
```

### CUDA Not Available

**Issue:** PyTorch not detecting CUDA

**Check CUDA version:**
```bash
nvidia-smi
```

**Reinstall PyTorch with correct CUDA version:**
```bash
# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Verify in Python:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

### Hugging Face Authentication Issues

**Error:** `401 Unauthorized` when downloading HSSD models

**Solution:**
```bash
# Login interactively
huggingface-cli login

# Or use token directly
huggingface-cli login --token YOUR_HF_TOKEN

# Verify login
huggingface-cli whoami
```

Make sure you've accepted the HSSD dataset license on Hugging Face.

### Out of Disk Space

**Issue:** Not enough space for 72GB HSSD models

**Solutions:**
1. Use a different location for data:
   ```bash
   # Create symlink to larger drive
   mkdir /path/to/larger/drive/hsm-data
   ln -s /path/to/larger/drive/hsm-data data
   ```

2. Download only required objects (advanced - may affect functionality)

### Package Installation Failures

**Error:** `Failed building wheel for python-fcl` or similar

**Solutions:**

For `python-fcl`:
```bash
sudo apt-get install libfcl-dev liboctomap-dev
pip install python-fcl
```

For `manifold3d`:
```bash
pip install --upgrade pip setuptools wheel
pip install manifold3d
```

For `embreex`:
```bash
sudo apt-get install libembree-dev
pip install embreex
```

## Usage

### Activate Environment

Every time you want to use HSM:

```bash
cd /path/to/hsm
source .venv/bin/activate
```

### Basic Scene Generation

```bash
python main.py -d "modern living room with grey sofa and wooden coffee table"
```

### Advanced Usage

```bash
# Specify object types
python main.py -d "bedroom" -t large wall small --output results/bedroom

# Skip certain processing steps
python main.py -d "kitchen" --skip-scene-motifs --skip-solver

# Custom output directory
python main.py -d "office" --output custom/output/path
```

### Deactivate Environment

```bash
deactivate
```

## Differences from Conda Installation

| Aspect | Conda/Mamba | Pip/Venv |
|--------|-------------|----------|
| Environment manager | conda/mamba | venv |
| Activation command | `conda activate hsm` | `source .venv/bin/activate` |
| Deactivation | `conda deactivate` | `deactivate` |
| Package manager | conda/pip hybrid | pip only |
| CUDA handling | Through conda channels | Through PyTorch pip index |
| System libraries | Some bundled | Must install separately |

## Additional Resources

- **Main README**: [README.md](README.md)
- **Project website**: https://3dlg-hcvc.github.io/hsm/
- **Issues**: https://github.com/3dlg-hcvc/hsm/issues
- **PyTorch Installation**: https://pytorch.org/get-started/locally/
- **Hugging Face CLI**: https://huggingface.co/docs/huggingface_hub/guides/cli

## Performance Notes

### GPU vs CPU

- **GPU (CUDA)**: Significantly faster for CLIP embeddings and PyTorch operations (recommended)
- **CPU**: Works but slower, suitable for testing or when GPU not available

### Memory Requirements

- **Minimum**: 8GB RAM
- **Recommended**: 16GB RAM
- **Heavy scenes**: 32GB RAM for complex scenes with many objects

### Disk I/O

The 72GB HSSD model download can take several hours depending on your internet connection:
- **100 Mbps**: ~2-3 hours
- **50 Mbps**: ~4-6 hours
- **25 Mbps**: ~8-12 hours

Use the `--cpu` flag if you encounter GPU memory issues or don't have a CUDA-capable GPU.
