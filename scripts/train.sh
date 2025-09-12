#!/bin/bash

set -e

JOBS_FILE="${1:-jobs.txt}"
SESSION_NAME="svebm_sequential"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

session_exists() {
    tmux has-session -t "$1" 2>/dev/null
}

# Check periodically and wait for tmux session to complete
wait_for_training_completion() {
    print_status "Waiting for current training session to complete..."
    
    while session_exists "svebm_train"; do
        sleep 10
    done
    
    print_success "Training session completed"
}

run_training_job() {
    local job_args="$1"
    local job_num="$2"
    local total_jobs="$3"
    
    echo
    echo "============================================================"
    print_status "Starting Job $job_num/$total_jobs"
    print_status "Arguments: $job_args"
    echo "============================================================"
    
    # Kill existing sessions
    if session_exists "svebm_train"; then
        print_warning "Killing existing training session."
        tmux kill-session -t svebm_train 2>/dev/null || true
        sleep 2
    fi
    
    print_status "Running: python main.py fit --config=$CONFIG_FILE $job_args"
    
    mkdir -p logs

    tmux new-session -d -s svebm_train -n training
    tmux send-keys -t svebm_train:training "conda activate SV-EBM && python main.py fit --config=$CONFIG_FILE $job_args 2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log" Enter
    
    wait_for_training_completion
    
    print_success "Job $job_num/$total_jobs completed"
}

    
extract_config_file() {
    local config_file="$1"
    
    if [[ ! -f "$config_file" ]]; then
        print_error "Configuration file '$config_file' not found"
        exit 1
    fi
    
    while IFS= read -r line || [[ -n "$line" ]]; do
        if [[ "$line" =~ ^[[:space:]]*# ]] || [[ -z "${line// }" ]]; then
            continue
        fi
        
        if [[ "$line" =~ ^CONFIG_FILE=(.+)$ ]]; then
            echo "${BASH_REMATCH[1]}"
            return
        fi
    done < "$config_file"
    
    echo "config/test_conf.yml"
}

load_jobs() {
    local config_file="$1"
    
    if [[ ! -f "$config_file" ]]; then
        print_error "Configuration file '$config_file' not found"
        exit 1
    fi
    
    local jobs=()
    
    while IFS= read -r line || [[ -n "$line" ]]; do
        if [[ "$line" =~ ^[[:space:]]*# ]] || [[ -z "${line// }" ]]; then
            continue
        fi
        
        if [[ "$line" =~ ^CONFIG_FILE= ]]; then
            continue
        fi
        
        line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        
        if [[ -n "$line" ]]; then
            jobs+=("$line")
        fi
    done < "$config_file"
    
    for job in "${jobs[@]}"; do
        echo "$job"
    done
}

main() {
    print_status "SV-EBM Sequential Training Runner"
    print_status "Jobs file: $JOBS_FILE"
    
    CONFIG_FILE=$(extract_config_file "$JOBS_FILE")
    print_status "Base config: $CONFIG_FILE"
    
    local jobs
    local jobs_output
    jobs_output=$(load_jobs "$JOBS_FILE")
    mapfile -t jobs <<< "$jobs_output"
    
    local total_jobs=${#jobs[@]}
    
    if [[ $total_jobs -eq 0 ]]; then
        print_error "No valid jobs found in configuration file"
        exit 1
    fi
    
    print_status "Found $total_jobs training jobs to run sequentially"
    
    # Signal handler
    trap 'print_warning "Interrupted by user. Stopping training sequence."; tmux kill-session -t svebm_train 2>/dev/null || true; exit 0' INT TERM
    
    for i in "${!jobs[@]}"; do
        local job_num=$((i + 1))
        local job_args="${jobs[i]}"
        
        print_status "Preparing job $job_num/$total_jobs"
        
        run_training_job "$job_args" "$job_num" "$total_jobs"
        
        sleep 5
    done
    
    echo
    echo "============================================================"
    print_success "All training jobs completed."
    print_status "$total_jobs jobs finished"
    echo "============================================================"
}

if [[ ! -f "Makefile" ]]; then
    print_error "Makefile not found. Please run this script from the project root directory."
    exit 1
fi

if ! command -v tmux &> /dev/null; then
    print_error "tmux is not installed. Please install tmux to use this script."
    exit 1
fi

main "$@"
