#!/usr/bin/env python3
"""
Monitor the training progress and show what's happening.
"""

import os
import time
import subprocess

def monitor_training():
    """Monitor the training progress."""
    
    print("🔍 Monitoring Training Progress...")
    print("=" * 50)
    
    while True:
        # Check if process is still running
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            if 'train_and_analyze_layer8.py' in result.stdout:
                print(f"✅ Training is running - {time.strftime('%H:%M:%S')}")
                
                # Check if analysis_results directory exists
                if os.path.exists('analysis_results'):
                    print("📁 Analysis results directory created")
                    
                    # List contents
                    for root, dirs, files in os.walk('analysis_results'):
                        for file in files:
                            if file.endswith('.json') or file.endswith('.md'):
                                print(f"   📄 {file}")
                
            else:
                print("❌ Training process not found - may have completed or failed")
                break
                
        except Exception as e:
            print(f"Error checking process: {e}")
            break
        
        time.sleep(10)  # Check every 10 seconds

if __name__ == "__main__":
    monitor_training()
