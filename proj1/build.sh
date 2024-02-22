#!/bin/bash
#git submodule update --init --recursive #in case of problem run this instead of:
git pull --recurse-submodules --jobs=10

#Uncomment next line and comment line after if you want to use make
#GENERATOR="Unix Makefiles"
GENERATOR=Ninja
export SUI_BUILD_TYPE="debug"

if [ $# -ne 0 ]
  then
    SUI_BUILD_TYPE=$1
fi

if [ "${SUI_BUILD_TYPE}" == "dfs" ] || [ "${SUI_BUILD_TYPE}" == "bfs" ] || [ "${SUI_BUILD_TYPE}" == "a_star" ]; then
    SUI_BUILD_TYPE="debug"
fi

export SUI_BUILD_DIR="cmake-build-${SUI_BUILD_TYPE}"

mkdir -p "${SUI_BUILD_DIR}"
# rm -rf "${SUI_BUILD_DIR:?}/*" # Probably unnecessary
cd "${SUI_BUILD_DIR}" || { echo "Error creating dir: ${SUI_BUILD_DIR}"; exit 1;}

if [ "${SUI_BUILD_TYPE}" == "debug" ]; then
  cmake -G ${GENERATOR} -DCMAKE_BUILD_TYPE=Debug .. || { echo "Error loading cmake."; exit 1;}
else
  cmake -G ${GENERATOR} -DCMAKE_BUILD_TYPE=Release .. || { echo "Error loading cmake."; exit 1;}
fi

time cmake --build . --target fc-sui -j 12 || { echo "Build failed!"; exit 1;}
echo "Build succeeded!"
cd -
