import numpy as np
import matplotlib.pyplot as plt
import math
import sys


N1 = [3 , 2]
P2 = [0 , 1 , 2 , 3 , 4]

lines = ["source scripts/scripts_run_" + str(n1) + "_" + str(p2) + ".txt" for n1 in N1 for p2 in P2]

with open('scripts/scripts_run_N1_all.txt', 'w') as f:
	for line in lines:
		print(line)
		f.write(line)
		f.write('\n')
		f.write("wait")
		f.write('\n')
