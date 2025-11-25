# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HSM (Hierarchical Scene Motifs) is a research project for generating realistic 3D indoor scenes from natural language descriptions. The system uses a hierarchical framework that organizes objects into meaningful motifs across multiple scales, leveraging Large Language Models (LLMs) and Visual Language Models (VLMs) for scene understanding and generation.

## Environment Setup

The project uses conda/mamba for environment management with Python 3.11:

```bash
# Automated setup (recommended)
./setup.sh

# Manual environment creation  
mamba env create -f environment.yml
conda activate hsm
```

**Required Environment Variables** (create from `.env.example`):
- `OPENAI_API_KEY`: OpenAI API key for LLM integration
- `HF_TOKEN`: Hugging Face token for HSSD model downloads

## Core Commands

### Scene Generation
```bash
# Basic scene generation
python main.py -d "description of room" --output results/output_dir

# With specific object types
python main.py -d "living room" -t large wall small --output results/living_room

# Ablation studies
python main.py -d "bedroom" --skip-scene-motifs --skip-solver --skip-spatial-optimization
```

### Data Setup
The automated setup script handles data downloads, but manual commands:
```bash
# Download HSSD models (requires HF token)
hf download hssd/hssd-hab --repo-type=dataset --include "objects/decomposed/**/*_part_*.glb" --local-dir "data/hssd-models"
```

## Architecture Overview

### Core Components

**`main.py`**: Entry point that orchestrates the hierarchical scene generation pipeline:
- Room analysis and decomposition
- Large objects (main furniture) processing  
- Wall objects (wall-mounted items) processing
- Ceiling objects (lights, fans) processing
- Small objects (decorative items) processing

**`hsm_core/`**: Core library organized by functionality:

- **`scene/`**: Scene management and 3D object processing
  - `setup.py`: Scene initialization and configuration loading
  - `large.py`, `wall.py`, `ceiling.py`, `small.py`: Object type-specific processors
  - `scene_3d.py`: 3D scene representation and manipulation
  - `placer.py`: Object placement logic and collision detection

- **`scene_motif/`**: Hierarchical scene motif system
  - `core/`: Object hierarchy and arrangement logic
  - `generation/`: LLM-based scene decomposition and motif generation
  - `spatial/`: Spatial optimization and constraint solving
  - `programs/`: Scene motif program interpretation

- **`retrieval/`**: 3D model retrieval and matching system
  - `core/`: Main retrieval logic using CLIP embeddings
  - `model/`: CLIP model management and embedding computation
  - `data/`: HSSD dataset integration and WordNet mapping

- **`solvers/`**: Spatial constraint solving
  - `unified_optimizer.py`: Multi-objective spatial optimization
  - `solver_dfs.py`: Depth-first search solver

- **`vlm/`**: Vision Language Model integration
  - `gpt.py`: OpenAI GPT integration for scene understanding
  - `vlm.py`: Room layout generation from descriptions

### Configuration System

**`configs/scene/scene_config.yaml`**: Main configuration file controlling:
- Room description and geometry settings
- Object generation parameters (iterations, occupancy targets)
- Feature toggles (scene motifs, solver, spatial optimization)
- Object type selection (large, wall, ceiling, small)

**`configs/prompts/`**: LLM prompt templates for different generation stages

### Data Dependencies

- **HSSD Models**: ~72GB of 3D models from Habitat Synthetic Scenes Dataset
- **Preprocessed Data**: CLIP embeddings and object category mappings
- **Support Surfaces**: Geometric data for object placement validation

### Key Workflows

1. **Scene Initialization**: Parse config, create output directories, initialize 3D scene
2. **Room Analysis**: Generate room geometry using VLM if not specified
3. **Hierarchical Object Placement**: Process objects by scale (large → wall → ceiling → small)
4. **Scene Motif Generation**: Use LLMs to create semantically meaningful object arrangements
5. **Spatial Optimization**: Solve collision detection and support constraints
6. **Output Generation**: Save 3D models (.glb), visualizations (.png), and scene state (.json)

## Development Notes

- The system processes objects hierarchically by scale and semantic importance
- Scene motifs are generated using LLM reasoning about spatial relationships
- 3D model retrieval uses CLIP embeddings for semantic matching
- Spatial optimization ensures physical plausibility (collision avoidance, support validation)
- All LLM sessions are logged and saved for debugging and analysis