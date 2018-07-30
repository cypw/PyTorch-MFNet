PWD_PATH=$(cd ./ && pwd)

HOST_FILE=./Hosts

SESSION_NAME='\"pytorch_dist\"'
CMD_NEW_SESSION="tmux new -s ${SESSION_NAME} -d"
CMD_RUN='export PATH=$PATH; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH; '
CMD_RUN+="cd ${PWD_PATH}; "
CMD_RUN+="bash dist.sh; "


cat ${HOST_FILE} | xargs -I{} ssh {} "hostname && ${CMD_NEW_SESSION} || true && tmux send -t \"${SESSION_NAME}\" \"${CMD_RUN}\" ENTER"
