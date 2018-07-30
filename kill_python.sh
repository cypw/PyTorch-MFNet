#!/bin/bash
# This scropt is for distributed training
# ---------------------------------------
# You can use this script to kill all python and the corresponding tmux sessions.

# Kill all python threads on remote nodes
HOST_FILE='Hosts'
cat ${HOST_FILE} | xargs -I{} ssh {} 'hostname && pkill python'

# Exit all tmux sessions
SESSION_NAME='\"pytorch_dist\"'
read -n1 -p "Exit all ${SESSION_NAME} sessions? [y,n]" input
if [[ $input == "Y" || $input == "y" ]]; then
    cat ${HOST_FILE} | xargs -I{} ssh {} "hostname && tmux send -t \"${SESSION_NAME}\" ENTER \"exit\" ENTER"
fi
echo "\n"