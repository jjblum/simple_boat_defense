import sys
import os
import subprocess

NUMBER_OF_RUNS = 50
NUMBER_PER_BATCH = NUMBER_OF_RUNS

path = os.path.dirname(os.path.realpath(__file__))

procs = list()
n = 0
while n < NUMBER_OF_RUNS:
    for i in range(min(NUMBER_PER_BATCH, NUMBER_OF_RUNS - n)):
        proc = subprocess.Popen([sys.executable, '{}/simulation_main.py'.format(path), "3", "2", "50.", "dynamic"])
        procs.append(proc)
    for proc in procs:
        proc.wait()
    procs = list()
    n += NUMBER_PER_BATCH


