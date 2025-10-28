#!/bin/bash

# GPU Usage Monitor - Notifies when other users start using GPUs
# Usage: ./gpu_monitor.sh [check_interval_seconds]

# Default check interval (seconds)
CHECK_INTERVAL=${1:-10}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Sound notification (optional)
NOTIFY_SOUND=true

# Log file for tracking
LOG_FILE="/tmp/gpu_monitor.log"

# Function to play notification sound
play_sound() {
    if [ "$NOTIFY_SOUND" = true ]; then
        # Try different sound commands
        if command -v paplay &> /dev/null; then
            paplay /usr/share/sounds/alsa/Front_Left.wav 2>/dev/null || true
        elif command -v aplay &> /dev/null; then
            aplay /usr/share/sounds/alsa/Front_Left.wav 2>/dev/null || true
        elif command -v beep &> /dev/null; then
            beep 2>/dev/null || true
        fi
    fi
}

# Function to send desktop notification
send_notification() {
    local title="$1"
    local message="$2"
    
    if command -v notify-send &> /dev/null; then
        notify-send "$title" "$message" -i video-display
    fi
}

# Function to log activity
log_activity() {
    local message="$1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $message" >> "$LOG_FILE"
}

# Function to get current GPU users
get_gpu_users() {
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader 2>/dev/null | while IFS=',' read -r gpu_uuid pid process_name; do
        # Clean up whitespace
        gpu_uuid=$(echo "$gpu_uuid" | tr -d ' ')
        pid=$(echo "$pid" | tr -d ' ')
        process_name=$(echo "$process_name" | tr -d ' ')
        
        if [ -n "$gpu_uuid" ] && [ -n "$pid" ]; then
            user=$(ps -o user= -p "$pid" 2>/dev/null || echo 'unknown')
            gpu_index=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | grep "$gpu_uuid" | cut -d',' -f1 | tr -d ' ')
            echo "$gpu_index,$user,$pid,$process_name"
        fi
    done
}

# Function to check for new users
check_gpu_usage() {
    local current_users=$(get_gpu_users)
    local current_user_count=$(echo "$current_users" | grep -v "^$" | wc -l)
    
    # If no processes, clear the state
    if [ "$current_user_count" -eq 0 ]; then
        echo "" > /tmp/gpu_monitor_state
        return
    fi
    
    # Check if state file exists, if not initialize it
    if [ ! -f /tmp/gpu_monitor_state ]; then
        echo "$current_users" > /tmp/gpu_monitor_state
        echo -e "${BLUE}📊 Initialized GPU monitoring state${NC}"
        return
    fi
    
    # Compare with previous state
    local previous_users=$(cat /tmp/gpu_monitor_state)
    local previous_user_count=$(echo "$previous_users" | grep -v "^$" | wc -l)
    
    # Check for new users
    echo "$current_users" | while IFS=',' read -r gpu_index user pid process_name; do
        if [ -n "$gpu_index" ] && [ -n "$user" ]; then
            # Check if this user/process combination is new
            if ! echo "$previous_users" | grep -q "$gpu_index,$user,$pid,$process_name"; then
                # Check if it's a different user than current user
                current_user=$(whoami)
                if [ "$user" != "$current_user" ]; then
                    # Get GPU details for the alert
                    gpu_details=$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | sed -n "$((gpu_index + 1))p")
                    gpu_name=$(echo "$gpu_details" | cut -d',' -f1 | tr -d ' ')
                    mem_used=$(echo "$gpu_details" | cut -d',' -f2 | tr -d ' ')
                    mem_total=$(echo "$gpu_details" | cut -d',' -f3 | tr -d ' ')
                    gpu_util=$(echo "$gpu_details" | cut -d',' -f4 | tr -d ' ')
                    
                    echo -e "${RED}🚨 ALERT: User '$user' started using GPU $gpu_index!${NC}"
                    echo -e "${YELLOW}   GPU: $gpu_name${NC}"
                    echo -e "${YELLOW}   Process: $process_name (PID: $pid)${NC}"
                    echo -e "${YELLOW}   Memory: ${mem_used}MB / ${mem_total}MB${NC}"
                    echo -e "${YELLOW}   Utilization: ${gpu_util}%${NC}"
                    
                    # Send notifications
                    send_notification "GPU Alert" "User '$user' started using GPU $gpu_index ($gpu_name)"
                    play_sound
                    
                    # Log the activity
                    log_activity "ALERT: User '$user' started using GPU $gpu_index ($gpu_name) - Process: $process_name (PID: $pid) - Memory: ${mem_used}MB/${mem_total}MB - Util: ${gpu_util}%"
                else
                    # Only log if this is truly a new process (not just the first run)
                    if [ -n "$previous_users" ]; then
                        log_activity "INFO: Current user started using GPU $gpu_index - Process: $process_name (PID: $pid)"
                    fi
                fi
            fi
        fi
    done
    
    # Update state
    echo "$current_users" > /tmp/gpu_monitor_state
}

# Function to show current status
show_status() {
    echo -e "${BLUE}=== GPU Monitor Status ===${NC}"
    echo "Monitoring interval: ${CHECK_INTERVAL}s"
    echo "Log file: $LOG_FILE"
    echo "Current user: $(whoami)"
    echo
    
    # Get GPU count
    local gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1 | tr -d ' ')
    
    echo -e "${YELLOW}GPU Status Overview:${NC}"
    echo "Total GPUs: $gpu_count"
    echo
    
    # Show detailed GPU information
    for ((i=0; i<gpu_count; i++)); do
        local gpu_info=$(nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | sed -n "$((i+1))p")
        local gpu_index=$(echo "$gpu_info" | cut -d',' -f1 | tr -d ' ')
        local gpu_name=$(echo "$gpu_info" | cut -d',' -f2 | tr -d ' ')
        local mem_used=$(echo "$gpu_info" | cut -d',' -f3 | tr -d ' ')
        local mem_total=$(echo "$gpu_info" | cut -d',' -f4 | tr -d ' ')
        local gpu_util=$(echo "$gpu_info" | cut -d',' -f5 | tr -d ' ')
        local temp=$(echo "$gpu_info" | cut -d',' -f6 | tr -d ' ')
        
        # Calculate memory percentage
        local mem_percent=0
        if [ "$mem_total" -gt 0 ]; then
            mem_percent=$((mem_used * 100 / mem_total))
        fi
        
        # Format GPU name (truncate if too long)
        if [ ${#gpu_name} -gt 25 ]; then
            gpu_name="${gpu_name:0:22}..."
        fi
        
        echo -e "${BLUE}GPU $gpu_index:${NC} $gpu_name"
        echo -e "  Memory: ${mem_used}MB / ${mem_total}MB (${mem_percent}%)"
        echo -e "  Utilization: ${gpu_util}%"
        echo -e "  Temperature: ${temp}°C"
        
        # Show processes on this GPU
        local gpu_processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader 2>/dev/null | while IFS=',' read -r gpu_uuid pid process_name; do
            gpu_uuid=$(echo "$gpu_uuid" | tr -d ' ')
            pid=$(echo "$pid" | tr -d ' ')
            process_name=$(echo "$process_name" | tr -d ' ')
            if [ -n "$gpu_uuid" ] && [ -n "$pid" ]; then
                gpu_idx=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | grep "$gpu_uuid" | cut -d',' -f1 | tr -d ' ')
                if [ "$gpu_idx" = "$gpu_index" ]; then
                    user=$(ps -o user= -p "$pid" 2>/dev/null || echo 'unknown')
                    echo "    $user ($process_name - PID: $pid)"
                fi
            fi
        done)
        
        if [ -n "$gpu_processes" ]; then
            echo -e "  ${YELLOW}Active processes:${NC}"
            echo "$gpu_processes"
        else
            echo -e "  ${GREEN}No active processes${NC}"
        fi
        echo
    done
}

# Main monitoring loop
main() {
    echo -e "${BLUE}🔍 GPU Usage Monitor Started${NC}"
    echo "Press Ctrl+C to stop"
    echo
    
    show_status
    
    while true; do
        check_gpu_usage
        sleep "$CHECK_INTERVAL"
    done
}

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Stopping GPU monitor...${NC}"
    rm -f /tmp/gpu_monitor_state
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. Make sure NVIDIA drivers are installed.${NC}"
    exit 1
fi

# Run main function
main
