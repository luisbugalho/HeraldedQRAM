import numpy as np
import matplotlib.pyplot as plt
import math
import sys


eta = sys.argv[1]

T1 = [0,1,2,3,4]
T2 = [0,1,2,3,4]
CNOT = [0,1,2,3,4,5]

lines = ["python3 readData/treatdatafile.py " + " " + eta + " " + str(t1) + " " + str(t2) + " " + str(cnot)  for t1 in T1 for t2 in T2 for cnot in CNOT]

with open('scripts/scripts_treat_' + eta + '.txt', 'w') as f:
	for line in lines:
		print(line)
		f.write(line)
		f.write('\n')
		f.write("wait")
		f.write('\n')