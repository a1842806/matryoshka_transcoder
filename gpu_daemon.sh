#!/bin/bash

# GPU Monitor Daemon - Runs in background and sends notifications
# Usage: ./gpu_daemon.sh start|stop|status

DAEMON_PID_FILE="/tmp/gpu_monitor_daemon.pid"
DAEMON_LOG_FILE="/tmp/gpu_monitor_daemon.log"
CHECK_INTERVAL=5

# Function to check for GPU usage changes
check_gpu_changes() {
    local current_users=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader 2>/dev/null | while IFS=',' read -r gpu_uuid pid process_name; do
        if [ -n "$gpu_uuid" ] && [ -n "$pid" ]; then
            user=$(ps -o user= -p "$pid" 2>/dev/null || echo 'unknown')
            gpu_index=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | grep "$gpu_uuid" | cut -d',' -f1 | tr -d ' ')
            echo "$gpu_index,$user,$pid,$process_name"
        fi
    done)
    
    # If no processes, clear state and return
    if [ -z "$current_users" ] || [ "$(echo "$current_users" | grep -v "^$" | wc -l)" -eq 0 ]; then
        echo "" > /tmp/gpu_daemon_state
        return
    fi
    
    # Check against previous state
    if [ ! -f /tmp/gpu_daemon_state ]; then
        echo "$current_users" > /tmp/gpu_daemon_state
        return
    fi
    
    local previous_users=$(cat /tmp/gpu_daemon_state)
    local current_user=$(whoami)
    
    # Check for new users
    echo "$current_users" | while IFS=',' read -r gpu_index user pid process_name; do
        if [ -n "$gpu_index" ] && [ -n "$user" ]; then
            if ! echo "$previous_users" | grep -q "$gpu_index,$user,$pid,$process_name"; then
                if [ "$user" != "$current_user" ]; then
                    # Send notification
                    if command -v notify-send &> /dev/null; then
                        notify-send "GPU Alert" "User '$user' started using GPU $gpu_index" -i video-display
                    fi
                    
                    # Log to file
                    echo "$(date '+%Y-%m-%d %H:%M:%S') - ALERT: User '$user' started using GPU $gpu_index - Process: $process_name (PID: $pid)" >> "$DAEMON_LOG_FILE"
                    
                    # Optional: Send email notification (uncomment and configure)
                    # echo "User '$user' started using GPU $gpu_index at $(date)" | mail -s "GPU Alert" your-email@domain.com
                fi
            fi
        fi
    done
    
    # Update state
    echo "$current_users" > /tmp/gpu_daemon_state
}

# Daemon main loop
daemon_loop() {
    echo "$(date) - GPU Monitor Daemon started" >> "$DAEMON_LOG_FILE"
    
    while true; do
        check_gpu_changes
        sleep "$CHECK_INTERVAL"
    done
}

# Start daemon
start_daemon() {
    if [ -f "$DAEMON_PID_FILE" ]; then
        local pid=$(cat "$DAEMON_PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "GPU Monitor Daemon is already running (PID: $pid)"
            return 1
        else
            rm -f "$DAEMON_PID_FILE"
        fi
    fi
    
    # Start daemon in background
    daemon_loop &
    local daemon_pid=$!
    echo "$daemon_pid" > "$DAEMON_PID_FILE"
    
    echo "GPU Monitor Daemon started (PID: $daemon_pid)"
    echo "Log file: $DAEMON_LOG_FILE"
    echo "Check interval: ${CHECK_INTERVAL}s"
}

# Stop daemon
stop_daemon() {
    if [ -f "$DAEMON_PID_FILE" ]; then
        local pid=$(cat "$DAEMON_PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            kill "$pid"
            rm -f "$DAEMON_PID_FILE"
            rm -f /tmp/gpu_daemon_state
            echo "GPU Monitor Daemon stopped"
        else
            echo "GPU Monitor Daemon is not running"
            rm -f "$DAEMON_PID_FILE"
        fi
    else
        echo "GPU Monitor Daemon is not running"
    fi
}

# Show daemon status
show_status() {
    if [ -f "$DAEMON_PID_FILE" ]; then
        local pid=$(cat "$DAEMON_PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "GPU Monitor Daemon is running (PID: $pid)"
            echo "Log file: $DAEMON_LOG_FILE"
            echo "Check interval: ${CHECK_INTERVAL}s"
            
            # Show recent log entries
            if [ -f "$DAEMON_LOG_FILE" ]; then
                echo
                echo "Recent activity:"
                tail -5 "$DAEMON_LOG_FILE"
            fi
        else
            echo "GPU Monitor Daemon is not running"
            rm -f "$DAEMON_PID_FILE"
        fi
    else
        echo "GPU Monitor Daemon is not running"
    fi
}

# Main script logic
case "$1" in
    start)
        start_daemon
        ;;
    stop)
        stop_daemon
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 {start|stop|status}"
        echo
        echo "Commands:"
        echo "  start  - Start the GPU monitor daemon"
        echo "  stop   - Stop the GPU monitor daemon"
        echo "  status - Show daemon status and recent activity"
        exit 1
        ;;
esac
