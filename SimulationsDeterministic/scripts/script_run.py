import numpy as np
import matplotlib.pyplot as plt
import math
import sys


eta = sys.argv[1]

layers = [i for i in range(2,13)]

lines = ["python3 qRAM_twostep.py " + " " + str(l) + " " + eta  for l in layers]

with open('scripts/scripts_run_' + eta + '.txt', 'w') as f:
	for line in lines:
		print(line)
		f.write(line)
		f.write('\n')
		f.write("wait")
		f.write('\n')