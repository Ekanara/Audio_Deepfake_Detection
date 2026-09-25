import json
import os
import librosa
import numpy as np
from pathlib import Path
import traceback

def check_audio_file_integrity(audio_path, max_duration=30):
    """
    Check if an audio file can be loaded and has valid properties.
    
    Args:
        audio_path: Path to audio file
        max_duration: Maximum expected duration in seconds
    
    Returns:
        dict with status and details
    """
    issues = []
    
    try:
        # Check file exists
        if not os.path.exists(audio_path):
            return {"status": "error", "issues": ["File does not exist"]}
        
        # Check file size
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            issues.append("File is empty (0 bytes)")
        elif file_size < 100:
            issues.append(f"File is suspiciously small ({file_size} bytes)")
        
        # Try to load audio
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Check audio properties
            duration = len(audio) / sr
            
            if len(audio) == 0:
                issues.append("Audio has no samples")
            
            if duration > max_duration:
                issues.append(f"Audio duration too long: {duration:.2f}s (max: {max_duration}s)")
            
            if duration < 0.1:
                issues.append(f"Audio duration too short: {duration:.2f}s")
            
            # Check for NaN or Inf values
            if np.isnan(audio).any():
                issues.append("Audio contains NaN values")
            
            if np.isinf(audio).any():
                issues.append("Audio contains Inf values")
            
            # Check audio range
            if audio.max() == audio.min():
                issues.append("Audio is constant (no variation)")
            
            return {
                "status": "ok" if not issues else "warning",
                "issues": issues,
                "duration": duration,
                "sample_rate": sr,
                "samples": len(audio),
                "file_size": file_size
            }
            
        except Exception as e:
            issues.append(f"Failed to load audio: {str(e)}")
            return {"status": "error", "issues": issues}
            
    except Exception as e:
        return {"status": "error", "issues": [f"Unexpected error: {str(e)}"]}

def diagnose_training_data(json_path, start_idx=0, end_idx=None, verbose=False):
    """
    Diagnose issues in training data JSON file.
    
    Args:
        json_path: Path to JSON file
        start_idx: Start checking from this index (useful for narrowing down)
        end_idx: Stop at this index (None = check all)
        verbose: Print details for every file
    """
    print(f"Loading JSON from: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    total = len(data)
    if end_idx is None:
        end_idx = total
    
    print(f"Total entries: {total}")
    print(f"Checking range: {start_idx} to {end_idx}")
    print("="*80)
    
    problematic_files = []
    error_files = []
    warning_files = []
    
    for idx in range(start_idx, min(end_idx, total)):
        item = data[idx]
        audio_path = item.get('audio', '')
        label = item.get('label', '')
        
        if verbose:
            print(f"\n[{idx}/{total}] Checking: {audio_path}")
        
        result = check_audio_file_integrity(audio_path)
        
        if result['status'] == 'error':
            error_files.append((idx, audio_path, result['issues']))
            print(f"\n❌ ERROR at index {idx}:")
            print(f"   Path: {audio_path}")
            print(f"   Label: {label}")
            for issue in result['issues']:
                print(f"   - {issue}")
        
        elif result['status'] == 'warning':
            warning_files.append((idx, audio_path, result['issues']))
            print(f"\n⚠️  WARNING at index {idx}:")
            print(f"   Path: {audio_path}")
            print(f"   Label: {label}")
            for issue in result['issues']:
                print(f"   - {issue}")
            if verbose:
                print(f"   Duration: {result.get('duration', 'N/A'):.2f}s")
                print(f"   Sample rate: {result.get('sample_rate', 'N/A')} Hz")
        
        elif verbose:
            print(f"   ✅ OK - Duration: {result.get('duration', 0):.2f}s, SR: {result.get('sample_rate', 0)} Hz")
        
        # Progress indicator
        if (idx + 1) % 50 == 0 and not verbose:
            print(f"Progress: {idx + 1}/{end_idx} checked...")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total checked: {min(end_idx, total) - start_idx}")
    print(f"Errors: {len(error_files)}")
    print(f"Warnings: {len(warning_files)}")
    print(f"Clean: {min(end_idx, total) - start_idx - len(error_files) - len(warning_files)}")
    
    # Save problematic indices
    if error_files or warning_files:
        with open('problematic_indices.txt', 'w') as f:
            f.write("ERROR FILES:\n")
            for idx, path, issues in error_files:
                f.write(f"\nIndex: {idx}\n")
                f.write(f"Path: {path}\n")
                f.write(f"Issues: {', '.join(issues)}\n")
            
            f.write("\n\nWARNING FILES:\n")
            for idx, path, issues in warning_files:
                f.write(f"\nIndex: {idx}\n")
                f.write(f"Path: {path}\n")
                f.write(f"Issues: {', '.join(issues)}\n")
        
        print(f"\nProblematic files saved to: problematic_indices.txt")
        
        if error_files:
            print(f"\n🎯 First error at index: {error_files[0][0]}")
            print(f"   This is likely causing your training to stop at step {error_files[0][0] + 1}")
    
    return error_files, warning_files

if __name__ == "__main__":
    # Replace with your JSON file path
    json_file = "data/label/beats/audio_labels_beats_event_stage3.json"
    
    # Training parameters
    batch_size = 160
    failed_step = 97
    
    # Calculate the range where the problem occurs
    # Step 97 processes samples from index (97 * 160) to (98 * 160 - 1)
    start_sample = failed_step * batch_size
    end_sample = (failed_step + 1) * batch_size
    
    print(f"Training stops at step {failed_step} with batch size {batch_size}")
    print(f"This means the problem is in batch {failed_step}, which contains samples {start_sample} to {end_sample - 1}")
    print(f"Checking this range first...\n")
    print("="*80)
    
    # Check the problematic batch
    diagnose_training_data(json_file, start_idx=start_sample, end_idx=end_sample, verbose=True)
    
    print("\n\n" + "="*80)
    response = input("Press Enter to check ALL files (this may take a while), or 'q' to quit: ")
    
    if response.lower() != 'q':
        print("="*80 + "\n")
        # Then check everything
        diagnose_training_data(json_file, verbose=False)