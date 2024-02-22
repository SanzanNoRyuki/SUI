#!/bin/bash
source ./build.sh || { echo "Build script failed" ; exit 1;}
# a_star
SOLVER="bfs"
if [ $# -ne 0 ]
  then
    SOLVER=$1
fi
# mem_limit 2GB
if [ "${SOLVER}" == "bfs" ]; then
  "${SUI_BUILD_DIR}"/fc-sui --solver "${SOLVER}" --easy-mode 20 --mem-limit 2147483648 10 156231
elif [ "${SOLVER}" == "dfs" ]; then
  "${SUI_BUILD_DIR}"/fc-sui --solver "${SOLVER}" --dls-limit 10 --easy-mode 15 --mem-limit 2147483648 10 156231
else
  "${SUI_BUILD_DIR}"/fc-sui --solver "${SOLVER}" --heuristic student --easy-mode 15 --mem-limit 2147483648 10 156231
# add run options for dfs and A*
fi
