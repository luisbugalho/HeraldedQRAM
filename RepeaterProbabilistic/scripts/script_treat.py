import numpy as np
import matplotlib.pyplot as plt
import math
import sys

flag = True
if sys.argv[1] == "P1":
	flag = False


N1 = [2,3,4,5]
P1 = [1.0 , 0.9 , 0.8 , 0.7, 0.6, 0.5]
P2 = [0,1,2,3,4,5]


T1 = [0,1,2,3,4]
T2 = [0,1,2,3,4]
CNOT = [0,1,2,3,4,5]

if flag:
	for n1 in N1:
		for p2 in P2:
			pathname = 'N1_' + str(n1) + "_P2_" + str(p2) 

			lines = ["python3 readData/treatdatafile.py " + " " + pathname + " " + str(t1) + " " + str(t2) + " " + str(cnot)  for t1 in T1 for t2 in T2 for cnot in CNOT]

			with open('scripts/scripts_treat_' + pathname + '.txt', 'w') as f:
				for line in lines:
					print(line)
					f.write(line)
					f.write('\n')
					f.write("wait")
					f.write('\n')

else:
	for p1 in P1:
		for p2 in P2:
			pathname = 'P1_' + str(p1) + "_P2_" + str(p2) 

			lines = ["python3 readData/treatdatafile.py " + " " + pathname + " " + str(t1) + " " + str(t2) + " " + str(cnot)  for t1 in T1 for t2 in T2 for cnot in CNOT]

			with open('scripts/scripts_treat_' + pathname + '.txt', 'w') as f:
				for line in lines:
					print(line)
					f.write(line)
					f.write('\n')
					f.write("wait")
					f.write('\n')