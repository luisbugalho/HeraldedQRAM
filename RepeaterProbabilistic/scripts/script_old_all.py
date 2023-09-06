import numpy as np
import matplotlib.pyplot as plt
import math
import sys


P1 = [  0.8 , 0.7 , 0.6 , 0.5]
P2 = [0 , 1 , 2 , 3 , 4]

lines = ["source scripts/scripts_run_" + str(p1) + "_" + str(p2) + ".txt" for p1 in P1 for p2 in P2]

with open('scripts/scripts_run_P1_all2.txt', 'w') as f:
	for line in lines:
		print(line)
		f.write(line)
		f.write('\n')
		f.write("wait")
		f.write('\n')
