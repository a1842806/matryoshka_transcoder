#!/bin/bash

# GPU Usage Monitor - Shows only GPUs currently in use with their owners
# Usage: ./gpu_usage.sh

echo "GPU Usage Monitor"
echo "================="
echo

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Make sure NVIDIA drivers are installed."
    exit 1
fi

# Get GPU count
gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "$gpu_count" -eq 0 ]; then
    echo "No NVIDIA GPUs detected."
    exit 1
fi

# Check if any processes are running on GPUs
processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v "^$" | wc -l)
if [ "$processes" -eq 0 ]; then
    echo "No processes currently using GPUs."
    exit 0
fi

echo "Active GPU Processes:"
echo "--------------------"

# Get GPU information and process details
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader 2>/dev/null | while IFS=',' read -r gpu_uuid pid process_name; do
    # Skip empty lines
    if [ -z "$gpu_uuid" ] || [ -z "$pid" ]; then
        continue
    fi
    
    # Get GPU index from UUID
    gpu_index=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | grep "$gpu_uuid" | cut -d',' -f1 | tr -d ' ')
    
    # Get process owner
    user=$(ps -o user= -p "$pid" 2>/dev/null || echo 'unknown')
    
    # Get GPU memory usage for this specific GPU
    gpu_mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | sed -n "$((gpu_index + 1))p" | tr -d ' ')
    if [ -n "$gpu_mem" ]; then
        mem_used=$(echo "$gpu_mem" | cut -d',' -f1)
        mem_total=$(echo "$gpu_mem" | cut -d',' -f2)
        if [ "$mem_total" -gt 0 ]; then
            mem_percent=$((mem_used * 100 / mem_total))
        else
            mem_percent=0
        fi
    else
        mem_used=0
        mem_total=0
        mem_percent=0
    fi
    
    # Get GPU utilization
    gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sed -n "$((gpu_index + 1))p" | tr -d ' ')
    
    # Format process name (truncate if too long)
    if [ ${#process_name} -gt 20 ]; then
        process_name="${process_name:0:17}..."
    fi
    
    printf "GPU %s | %-8s | %-6s | %-20s | %3s%% util | %s/%s MB (%2d%% mem)\n" \
        "$gpu_index" "$user" "$pid" "$process_name" "$gpu_util" "$mem_used" "$mem_total" "$mem_percent"
done

echo
echo "Summary:"
echo "--------"
echo "Total GPUs: $gpu_count"
echo "GPUs in use: $(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -v "^$" | sort -u | wc -l)"
echo "Active processes: $processes"
