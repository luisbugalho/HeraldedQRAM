import numpy as np
import matplotlib.pyplot as plt
import math
import sys


layers = int(sys.argv[1])
P1 = float(sys.argv[2])
P2 = int(sys.argv[3])

lines = ["python3 qRAM_repeater.py " + str(i) + " " + str(P1) + " " + str(P2)  for i in range(2,layers+1)]

with open('scripts/scripts_run_' + str(P1) + "_" + 	str(P2) + '.txt', 'w') as f:
	for line in lines:
		print(line)
		f.write(line)
		f.write('\n')
		f.write("wait")
		f.write('\n')