#!/bin/bash

#sudo apt-get install gcc g++ cmake ninja-build git

git init
git submodule add https://github.com/ibenes/freecell.git
git submodule update --init --recursive

cp freecell/sui-solution.cc .

chmod u+x build.sh run.sh

./build.sh
./build.sh release
