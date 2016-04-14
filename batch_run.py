import sys
import os
import subprocess

NUMBER_OF_RUNS = 100
NUMBER_PER_BATCH = 20

NUMBER_OF_DEFENDERS = 5
NUMBER_OF_ATTACKERS = 2
ATTACK_TYPE = "TTA"  # "TTA"
MAX_INTERCEPT_DISTANCE = 20.

path = os.path.dirname(os.path.realpath(__file__))

procs = list()
n = 0
while n < NUMBER_OF_RUNS:
    for i in range(min(NUMBER_PER_BATCH, NUMBER_OF_RUNS - n)):
        proc = subprocess.Popen([sys.executable, '{}/simulation_main.py'.format(path), str(NUMBER_OF_DEFENDERS),
                                 str(NUMBER_OF_ATTACKERS), str(MAX_INTERCEPT_DISTANCE), ATTACK_TYPE,
                                 "./results/def{}_atk{}_maxIntDist{}_{}_{}".format(NUMBER_OF_DEFENDERS,
                                                                                   NUMBER_OF_ATTACKERS,
                                                                                   MAX_INTERCEPT_DISTANCE,
                                                                                   ATTACK_TYPE, n)])
        procs.append(proc)
        n += 1
    for proc in procs:
        proc.wait()
    procs = list()


