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
    
    # Check if state file exists
    if [ ! -f /tmp/gpu_monitor_state ]; then
        echo "$current_users" > /tmp/gpu_monitor_state
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
                    echo -e "${RED}🚨 ALERT: User '$user' started using GPU $gpu_index!${NC}"
                    echo -e "${YELLOW}   Process: $process_name (PID: $pid)${NC}"
                    
                    # Send notifications
                    send_notification "GPU Alert" "User '$user' started using GPU $gpu_index"
                    play_sound
                    
                    # Log the activity
                    log_activity "ALERT: User '$user' started using GPU $gpu_index - Process: $process_name (PID: $pid)"
                else
                    echo -e "${GREEN}ℹ️  You started using GPU $gpu_index${NC}"
                    log_activity "INFO: Current user started using GPU $gpu_index - Process: $process_name (PID: $pid)"
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
    
    local current_users=$(get_gpu_users)
    if [ -n "$current_users" ] && [ "$(echo "$current_users" | grep -v "^$" | wc -l)" -gt 0 ]; then
        echo -e "${YELLOW}Currently active GPU users:${NC}"
        echo "$current_users" | while IFS=',' read -r gpu_index user pid process_name; do
            if [ -n "$gpu_index" ]; then
                echo "  GPU $gpu_index: $user ($process_name)"
            fi
        done
    else
        echo -e "${GREEN}No GPU processes currently running${NC}"
    fi
    echo
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
