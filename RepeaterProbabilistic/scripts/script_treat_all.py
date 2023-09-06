import numpy as np
import matplotlib.pyplot as plt
import math
import sys




N1 = [2,3,4,5]
P1 = [1.0 , 0.9 , 0.8 , 0.7, 0.6, 0.5]
P2 = [0,1,2,3,4]

if sys.argv[1] == "N1":

	lines = ["source scripts/scripts_treat_N1_" + str(t1) + "_P2_" + str(t2) + ".txt"  for t1 in N1 for t2 in P2]

	with open('scripts/scripts_treat_all_N1.txt', 'w') as f:
		for line in lines:
			print(line)
			f.write(line)
			f.write('\n')
			f.write("wait")
			f.write('\n')

else:

	lines = ["source scripts/scripts_treat_P1_" + str(p1) + "_P2_" + str(t2) + ".txt"  for p1 in P1 for t2 in P2]

	with open('scripts/scripts_treat_all_P1.txt', 'w') as f:
		for line in lines:
			print(line)
			f.write(line)
			f.write('\n')
			f.write("wait")
			f.write('\n')
