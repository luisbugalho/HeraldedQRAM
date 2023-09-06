import numpy as np
import matplotlib.pyplot as plt


N1 = [2,3,4,5]
P1 = [1.0 , 0.9 , 0.8 , 0.7, 0.6, 0.5]
P2 = [0,1,2,3,4]


lines01 = ["mkdir All_Data_Times/N1_" + str(t1) + "_P2_" + str(t2) + "/"  for t1 in N1 for t2 in P2 ]
lines02 = ["mkdir All_Data_Times/P1_" + str(t1) + "_P2_" + str(t2) + "/"  for t1 in P1 for t2 in P2 ]


lines1 = ["cp All_Data_Raw_New/N1_" + str(t1) + "_P2_" + str(t2) + "/qRAM_teleportation_times_" + str(layer) + ".npy All_Data_Times/N1_" + str(t1) + "_P2_" + str(t2) + "/"  for t1 in N1 for t2 in P2 for layer in range(2,11)]
lines2 =["cp All_Data_Raw_New/P1_" + str(t1) + "_P2_" + str(t2) + "/qRAM_teleportation_times_" + str(layer) + ".npy All_Data_Times/P1_" + str(t1) + "_P2_" + str(t2) + "/"  for t1 in P1 for t2 in P2 for layer in range(2,11)]
lines = lines01 + lines02 + lines1 + lines2

with open('scripts/copy_times.txt', 'w') as f:
	for line in lines:
		print(line)
		f.write(line)
		f.write('\n')
		f.write("wait")
		f.write('\n')
