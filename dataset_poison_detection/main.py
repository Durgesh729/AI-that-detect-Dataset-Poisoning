import os
import subprocess
import sys
import numpy as np
import pandas as pd

def run_script(script_name):
    # Get the directory of the current file (main.py)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(base_dir, "scripts", script_name)
    
    print(f"\n" + "="*50)
    print(f"RUNNING: {script_name}")
    print("="*50)
    
    # Run from the base_dir so that relative paths in scripts work correctly
    result = subprocess.run([sys.executable, script_path], capture_output=False, cwd=base_dir)
    if result.returncode != 0:
        print(f"Error running {script_name}")
    return result.returncode == 0

def main():
    print("Dataset Poisoning Detection Pipeline Orchestrator")
    print("================================================")
    
    # Ensure data directories exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    pipeline_stages = [
        "poison_attack.py",
        "feature_profiling.py",
        "isolation_forest_detector.py",
        "autoencoder_detector.py",
        "influence_functions.py",
        "trust_score.py",
        "train_eval.py"
    ]

    for stage in pipeline_stages:
        success = run_script(stage)
        if not success:
            print(f"Pipeline failed at {stage}")
            return

    print("\n" + "="*50)
    print("FINAL PIPELINE SUMMARY")
    print("="*50)
    
    # Consolidate results
    # In a production version, we would parse the output or load npy files
    # Here we'll print a reminder to check the console outputs above for specific precision/recall
    
    print("\nDetection methods completed. Results generated in data/processed/")
    print("Check the individual stage outputs above for Precision and Recall metrics.")
    print("\nVisualizations saved:")
    print("- data/processed/profile_label_flip_dataset.png")
    print("- data/processed/profile_backdoor_dataset.png")
    
    print("\nDetection flags saved as .npy files in data/processed/")
    print("Pipeline Execution Complete.")

if __name__ == "__main__":
    main()
