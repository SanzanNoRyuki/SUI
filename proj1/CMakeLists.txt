cmake_minimum_required(VERSION 3.12)

project(SUI-proj1)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG_INIT  "-std=c++17 -Wall -Wextra -pedantic -DNDebug")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-std=c++17 -Wall -Wextra -pedantic -O2 -DNDebug")


find_package(Threads REQUIRED)

set(SOURCES freecell/card.cc freecell/fc-sui.cc freecell/card-storage.cc freecell/move.cc freecell/game.cc freecell/strategies-provided.cc freecell/search-interface.cc freecell/memusage.cc freecell/mem_watch.cc freecell/evaluation-type.cc sui-solution.cc)
add_executable(fc-sui ${SOURCES})
target_include_directories(fc-sui PRIVATE freecell)
target_link_libraries(fc-sui Threads::Threads)
