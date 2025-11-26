#!/usr/bin/env python3
"""
Batch scene generation script for SceneEval evaluation.

Reads scene descriptions from SceneEval annotations.csv and generates scenes
using HSM, outputting them in SceneEval-compatible format.

Usage:
    python scripts/batch_sceneeval.py --start 0 --end 5
    python scripts/batch_sceneeval.py --ids 0,1,2,3,4
"""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path


def read_annotations(csv_path: Path) -> list[dict]:
    """Read annotations from CSV file."""
    annotations = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotations.append({
                'id': int(row['ID']),
                'description': row['Description'],
                'difficulty': row.get('Difficulty', 'unknown')
            })
    return annotations


def generate_scene(scene_id: int, description: str, hsm_dir: Path, temp_output_dir: Path) -> Path | None:
    """
    Run HSM to generate a single scene.

    Returns the path to stk_scene_state.json if successful, None otherwise.
    """
    print(f"\n{'='*60}")
    print(f"Generating scene {scene_id}")
    print(f"Description: {description}")
    print(f"{'='*60}")

    # Create temp output directory for this scene
    scene_temp_dir = temp_output_dir / f"scene_{scene_id}"
    scene_temp_dir.mkdir(parents=True, exist_ok=True)

    # Run HSM main.py
    cmd = [
        sys.executable,
        str(hsm_dir / "main.py"),
        "-d", description,
        "--output", str(scene_temp_dir)
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(hsm_dir),
            capture_output=False,  # Let output stream to console
            text=True,
            timeout=1800  # 30 minute timeout per scene
        )

        if result.returncode != 0:
            print(f"ERROR: HSM failed for scene {scene_id} with return code {result.returncode}")
            return None

    except subprocess.TimeoutExpired:
        print(f"ERROR: Timeout generating scene {scene_id}")
        return None
    except Exception as e:
        print(f"ERROR: Exception generating scene {scene_id}: {e}")
        return None

    # Find the stk_scene_state.json file (it may be in a timestamped subdirectory)
    stk_files = list(scene_temp_dir.rglob("stk_scene_state.json"))

    if not stk_files:
        print(f"ERROR: No stk_scene_state.json found for scene {scene_id}")
        return None

    # Use the most recently modified one if multiple exist
    stk_file = max(stk_files, key=lambda p: p.stat().st_mtime)
    print(f"Found scene state: {stk_file}")

    return stk_file


def main():
    parser = argparse.ArgumentParser(
        description="Batch generate scenes for SceneEval evaluation"
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=str(Path.home() / "SceneEval/input/annotations.csv"),
        help="Path to SceneEval annotations.csv"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(Path.home() / "SceneEval/input/HSM"),
        help="Output directory for SceneEval-compatible scene files"
    )
    parser.add_argument(
        '--temp-dir',
        type=str,
        default=str(Path.home() / "hsm/results/sceneeval_batch"),
        help="Temporary directory for HSM outputs"
    )
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help="Start scene ID (inclusive)"
    )
    parser.add_argument(
        '--end',
        type=int,
        default=5,
        help="End scene ID (exclusive)"
    )
    parser.add_argument(
        '--ids',
        type=str,
        default=None,
        help="Comma-separated list of scene IDs to generate (overrides --start/--end)"
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help="Skip scenes that already have output files"
    )

    args = parser.parse_args()

    # Resolve paths
    csv_path = Path(args.csv).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    temp_dir = Path(args.temp_dir).expanduser()
    hsm_dir = Path(__file__).parent.parent.resolve()

    # Validate CSV exists
    if not csv_path.exists():
        print(f"ERROR: Annotations CSV not found: {csv_path}")
        sys.exit(1)

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Read annotations
    annotations = read_annotations(csv_path)
    print(f"Loaded {len(annotations)} scene annotations from {csv_path}")

    # Determine which scene IDs to process
    if args.ids:
        scene_ids = [int(x.strip()) for x in args.ids.split(',')]
    else:
        scene_ids = list(range(args.start, args.end))

    print(f"Will generate scenes: {scene_ids}")

    # Track results
    successful = []
    failed = []
    skipped = []

    # Generate each scene
    for scene_id in scene_ids:
        # Check if scene ID is valid
        if scene_id < 0 or scene_id >= len(annotations):
            print(f"WARNING: Scene ID {scene_id} out of range, skipping")
            failed.append(scene_id)
            continue

        # Check if output already exists
        output_file = output_dir / f"scene_{scene_id}.json"
        if args.skip_existing and output_file.exists():
            print(f"Skipping scene {scene_id} (output already exists)")
            skipped.append(scene_id)
            continue

        # Get scene info
        scene_info = annotations[scene_id]
        description = scene_info['description']

        # Generate scene
        stk_file = generate_scene(scene_id, description, hsm_dir, temp_dir)

        if stk_file:
            # Copy to output directory with SceneEval naming convention
            shutil.copy2(stk_file, output_file)
            print(f"SUCCESS: Saved {output_file}")
            successful.append(scene_id)
        else:
            failed.append(scene_id)

    # Print summary
    print(f"\n{'='*60}")
    print("BATCH GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Successful: {len(successful)} scenes - {successful}")
    print(f"Failed:     {len(failed)} scenes - {failed}")
    print(f"Skipped:    {len(skipped)} scenes - {skipped}")
    print(f"\nOutput directory: {output_dir}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
