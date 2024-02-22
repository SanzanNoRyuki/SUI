#!/usr/bin/python3
import subprocess
import sys
import statistics

# TEST INPUTS:
# test01_version # python3 cmake-build-release/test.py 0 1 2 4 8
# test02_version # python3 cmake-build-release/test.py 13 69 255 420 1337
# test03_version # python3 cmake-build-release/test.py 6969 156231 964618 1999999 6421616
# test04_version # python3 cmake-build-release/test.py 9999999 55555555 77777777 15151515 21321132

# bfs, dfs, a_star
SOLVER = "a_star"

# student, nb_not_home
HEURISITC = "student"

# 0, 10, 20, 1000, 1000000
DLS_LIMIT =  10

# 10, 20, 50, 140, 400
EZ_MODE = 10

# 1, 10, 20, 50, 140, 400
GAMES_COUNT = 10

# 2147483648
MEM_LIMIT = 2 * 1024 * 1024 * 1024

def get_stats(string):
    words = string.split();

    success = int(words[5])
    time = int(words[16])
    states = int(words[-1])

    return success, time, states

running = []
done = []
for seed in sys.argv[1:]:
    running.append(
        (
            subprocess.Popen(['cmake-build-release/fc-sui', '--solver', f'{SOLVER}', '--heuristic', f'{HEURISITC}', '--easy-mode', f'{EZ_MODE}', f'{GAMES_COUNT}', f'{seed}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE),
            f"cmake-build-release/fc-sui --solver {SOLVER} --heuristic {HEURISITC} --dls-limit {DLS_LIMIT} --easy-mode {EZ_MODE} {GAMES_COUNT} {seed}"
        )
    )

for process, command in running:
    done.append((process.communicate(), command))

succeses = 0
times = 0
avg_states = 0
median = []
for p, a in done:
    print(f"Params: {a}", file=sys.stderr)
    print(f"Out: {p[0].decode()}", end="", file=sys.stderr)
    print(get_stats(p[0].decode()), end="\n\n", file=sys.stderr)

    success, time, states = get_stats(p[0].decode())
    median.append((success, time, states))

    succeses += success
    times += time
    avg_states += states

succeses /= len(done)
times /= len(done)
avg_states /= len(done)

print(f"{succeses} {times} {avg_states}")
print(f"{statistics.median([i[0] for i in median])} {statistics.median([i[1] for i in median])} {statistics.median([i[2] for i in median])}")
