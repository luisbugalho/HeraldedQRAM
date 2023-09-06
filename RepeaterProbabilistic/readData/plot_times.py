import numpy as np
import matplotlib.pyplot as plt
import math
import sys


### argv1 - which simulations we want
### argv2 - scaling on the y axis
### argv3 - fillin betwwen
### argv4 - order of the plots - true means reversed
### argv5 - Savename for file
### argv6 - print P_d
### argv7 - print efficiency


file_directory = list(map(str, sys.argv[1].split(',')))
layers = 10
efficiency = [0.9, 0.5, 0.6, 0.7, 0.8, 1.0]


data_tree = []
data_tree_std = []

data_points = [2**i for i in range(2,layers+1)]

for file in file_directory:
	data_mean = []
	data_std = []

	for i in range(2,layers+1):
		data_array = np.load('All_Data_Times/' + file + '/qRAM_teleportation_times_'+str(i) + '.npy')
		new_data = [data_array[i+1]-data_array[i] for i in range(0,len(data_array)-1)]
		#print(new_data)
		data_mean.append(np.mean(new_data))
		data_std.append(np.std(new_data)/100**0.5)
		

	data_tree.append(data_mean)
	print(data_mean)
	data_tree_std.append(data_std)
	print(data_std)

label_ini = []
for file in file_directory:
	labelplot = ""
	if sys.argv[6] == "True" :
		if file[0:2] == "N1":
			labelplot += "$P_{d}$ = " + str("{:.1%}".format(1/2**(int(file[3])-1)))  

		elif file[0:2] == "P1":
			labelplot += "$P_{d}$ = " + str("{:.0%}".format(1-float(file[3:6])))  

	if sys.argv[7] == "True":
		if file[0:2] == "N1":
			if sys.argv[7] == "True" :
				labelplot += " | "
			labelplot += "$\eta$ = " + str("{:.1f}".format(efficiency[int(file[8])])) 

		elif file[0:2] == "P1":
			if sys.argv[7] == "True" :
				labelplot += " | "
			labelplot += "$\eta$ = " + str("{:.1f}".format(efficiency[int(file[10])])) 

		else: 
			labelplot += "Two-step Scheme : $\eta$ = " + str(efficiency[int(file[0])])

	label_ini.append(labelplot)

print(label_ini)
cmap = plt.get_cmap('twilight', 25)
colors = cmap(np.linspace(0,1,25))
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12

plt.figure(figsize=(8,6))
if sys.argv[4] == "True":
	for i in range(len(file_directory)-1,-1,-1):
		plt.errorbar(data_points,data_tree[i],data_tree_std[i],color=colors[4*(i+1)],linewidth=2,label=label_ini[i])
		if sys.argv[3]:
			if sys.argv[3] == "True":
				plt.fill_between(data_points, np.array(data_tree[i]) - np.array(data_tree_std[i]), np.array(data_tree[i]) + np.array(data_tree_std[i]),color=colors[4*(i+1)], alpha=0.2)
else:
	for i in range(0,len(file_directory)):
		plt.errorbar(data_points,data_tree[i],data_tree_std[i],color=colors[4*(i+1)],linewidth=2,label=label_ini[i])
		if sys.argv[3]:
			if sys.argv[3] == "True":
				plt.fill_between(data_points, np.array(data_tree[i]) - np.array(data_tree_std[i]), np.array(data_tree[i]) + np.array(data_tree_std[i]),color=colors[4*(i+1)], alpha=0.2)

plt.yscale(sys.argv[2])
#plt.ylim([0.9825, 1.001])

plt.legend(loc=3)
# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='-' , alpha=0.7)

# Show the minor grid lines with very faint and almost transparent grey lines
# plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# plt.minorticks_on()
# plt.errorbar(data_points,data_mean,data_std)
# plt.fill_between(data_points,data_min, data_max,alpha=0.2, edgecolor='#CC4F1B', facecolor='#FF9848')
#plt.yscale('log')
#plt.title('Query time Scaling')
plt.xscale('log')
plt.ylabel('Query time (ns)')
plt.xlabel('Number of Qubits')
plt.legend()
plt.subplots_adjust(left=0.1, bottom=None, right=0.98, top=0.98, wspace=None, hspace=None)


plt.savefig( sys.path[0] + "/../Plots/" + sys.argv[5] + '.pdf',dpi=300)


plt.show()

#print("\nData summary:\n", data_array)