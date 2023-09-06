import numpy as np
import matplotlib.pyplot as plt
import math
import sys

eta = 0

layers = 5

plot_guides = ["0_0_0","3_1_0","3_1_1","3_1_2","3_1_3","3_1_4"]

lines = ["python3 qRAM_twostep_test.py " + str(layers) + " " + str(eta) + " " + plot_guide  for plot_guide in plot_guides]

n_simul = 20

for i,line in enumerate(lines):
	with open('scripts/scripts_test_' + str(eta) + "_" + plot_guides[i] + '.txt', 'w') as f:
		for i in range(n_simul):
			print(line)
			f.write(line)
			f.write('\n')
			f.write("wait")
			f.write('\n')