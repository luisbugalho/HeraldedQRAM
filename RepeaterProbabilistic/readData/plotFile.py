import numpy as np
import matplotlib.pyplot as plt
import math
import sys

def multiplyList(myList) :
     
    # Multiply elements one by one
    result = 1
    for x in myList:
         result = result * x
    return result

def errorList(myList,errorList) :
     
    # Multiply elements one by one
    prodList = multiplyList(myList)
    result = 0
    for i in range(0,len(myList)):
         result = result + prodList/myList[i]*errorList[i]
    return result

def Transpose(myList) :

	numpy_array = np.array(myList)
	#print(len(myList[0]))
	transpose = numpy_array.T
	numpy_array = transpose.tolist()
	#print(len(numpy_array))

	return numpy_array

def std_dict(d,key):
	new_list = d[key]
	return np.std(np.array(new_list))

def mean_dict(d,key):
	new_list = d[key]
	return np.mean(np.array(new_list))


def max_func(a,b):
	if a>b:
		return a
	else:
		return b

T1 = [math.inf, 2*10**7, 2*10**8, 2*10**9, 2*10**10]
T2 = [math.inf, 10**7, 10**8, 10**9, 10**10]
CNOT_ERRORS = [0, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
efficiency = [0.9, 0.5, 0.6, 0.7, 0.8, 1.0]
cnot_fidelity =  [1-err for err in CNOT_ERRORS]
CNOT_ERROR = [4 / 3 * (1 - err ** 2) for err in cnot_fidelity]

data_tree = []
data_tree_std = []
data_points = []


### argv1 - which simulations we want
### argv2 - which combinations of T1 we want
### argv3 - which combinations of T2 we want
### argv4 - which combinations of CNOT errors we want
### argv5 - scaling on the y axis
### argv6 - name of the plot
### argv7 - flag print Pd
### argv8 - flag print eta
### argv9 - Flag print T1
### argv10 - Flag Print T2
### argv11 -  Flag Print CNOT
### argv12 - Savename for file
### argv13 - compare flag
### argv14 - compare simulation -directory

print(sys.argv)

file_directory = list(map(str, sys.argv[1].split(',')))
FLAG_T1 = list(map(int, sys.argv[2].split(',')))
FLAG_T2 = list(map(int, sys.argv[3].split(',')))
FLAG_CNOT = list(map(int, sys.argv[4].split(',')))

PLOT_GUIDES = [(flag_file, flag_t1,flag_t2,flag_cnot) for flag_file in file_directory for flag_t1 in FLAG_T1 for flag_t2 in FLAG_T2 for flag_cnot in FLAG_CNOT]
print(PLOT_GUIDES)

for plot_guides in PLOT_GUIDES:
	data_array = np.load(sys.path[0] + "/../All_Data_Processed_New/" + str(plot_guides[0]) + "/" + str(plot_guides[0]) + "_" + str(plot_guides[1])  + "_" + str(plot_guides[2]) + "_" + str(plot_guides[3]) + ".npy",allow_pickle=True)
	
	data_tree.append([ multiplyList(data_array[i][0]) for i in range(0,len(data_array)) ])
	data_tree_std.append([ errorList(data_array[i][0],data_array[i][1]) for i in range(0,len(data_array)) ])
	data_points = [2**i for i in range(2,len(data_array)+2)]

mini = 1000
for data in data_tree:
	if len(data) < mini:
		mini = len(data)

for i in range(0,len(data_tree)):
	data_tree[i] = data_tree[i][0:mini]
	data_tree_std[i] = data_tree_std[i][0:mini]
data_points = data_points[0:mini]

n_colors = len(PLOT_GUIDES)*4+4
cmap = plt.get_cmap('twilight', n_colors)
colors = cmap(np.linspace(0,1,n_colors))
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12

plt.figure(figsize=(8,6))

#for i in range(layers-2,layers-1):
#	plt.errorbar(data_points_layers[0:i+2],data_array[i][0],data_array[i][1],label='Last Layer GHZ Fidelity')# + str(i+2) + ' layers')
for i in range(0,len(PLOT_GUIDES)):
	print(data_tree[i])

	#LABELING

	labelplot = ""
	if sys.argv[7] == "True" :
		if PLOT_GUIDES[i][0][0:2] == "N1":
			labelplot += "$P_{d}$ = " + str("{:.1%}".format(1/2**(int(PLOT_GUIDES[i][0][3])-1)))  

		elif PLOT_GUIDES[i][0][0:2] == "P1":
			labelplot += "$P_{d}$ = " + str("{:.0%}".format(1-float(PLOT_GUIDES[i][0][3:6])))  

	if sys.argv[8] == "True":
		if PLOT_GUIDES[i][0][0:2] == "N1":
			if sys.argv[7] == "True" :
				labelplot += " | "
			labelplot += "$\eta$ = " + str("{:.1f}".format(efficiency[int(PLOT_GUIDES[i][0][8])])) 

		elif PLOT_GUIDES[i][0][0:2] == "P1":
			if sys.argv[7] == "True" :
				labelplot += " | "
			labelplot += "$\eta$ = " + str("{:.1f}".format(efficiency[int(PLOT_GUIDES[i][0][10])])) 

		else: 
			labelplot += "Two-step Scheme : $\eta$ = " + str(efficiency[int(PLOT_GUIDES[i][0])])

	if sys.argv[9] == "True":
		if sys.argv[8] == "True" or sys.argv[7] == "True":
			labelplot += " | "
		if PLOT_GUIDES[i][1] > 2:
			labelplot += "$T_1$ = " + str("{:.0f}".format(T1[PLOT_GUIDES[i][1]]/10**9)) + " s"
		else:
			labelplot += "$T_1$ = " + str("{:.0f}".format(T2[PLOT_GUIDES[i][2]]/10**6)) + ' ms'

	if sys.argv[10] == "True":
		if sys.argv[9] == "True" or sys.argv[8] == "True" or sys.argv[7] == "True":
			labelplot += " | "
		if PLOT_GUIDES[i][2] > 2:
			labelplot += "$T_2$ = " + str("{:.0f}".format(T2[PLOT_GUIDES[i][2]]/10**9)) + ' s'
		else:
			labelplot += "$T_2$ = " + str("{:.0f}".format(T2[PLOT_GUIDES[i][2]]/10**6)) + ' ms'

	if sys.argv[11] == "True":
		if sys.argv[10] == "True" or sys.argv[9] == "True" or sys.argv[8] == "True" or sys.argv[7] == "True":
			labelplot += " | "
		if PLOT_GUIDES[i][3] == 0:
			labelplot +=  "$\epsilon_{CNOT}$ = 0"
		else:
			labelplot +=  "$\epsilon_{CNOT}$ = $10^{" + str(CNOT_ERRORSp[PLOT_GUIDES[i][3]]) + "}$"


	plt.errorbar(data_points,data_tree[i],data_tree_std[i],color=colors[4*(i+1)],linewidth=2,label=labelplot)
	plt.fill_between(data_points, np.array(data_tree[i]) - np.array(data_tree_std[i]), np.array(data_tree[i]) + np.array(data_tree_std[i]),color=colors[4*(i+1)], alpha=0.2)

if sys.argv[13] == "True":
	file_directory_compare = list(map(str, sys.argv[14].split(',')))
	n_colors2 = len(file_directory_compare)*4+4
	cmap2 = plt.get_cmap('gist_gray_r', n_colors2)
	colors2 = cmap(np.linspace(0,1,n_colors2))
	
	for j,file in enumerate(file_directory_compare):
		data_array = np.load(file + ".npy",allow_pickle=True)
	
		data_compare = [ multiplyList(data_array[i][0]) for i in range(0,len(data_array)) ]
		data_compare_std = [ errorList(data_array[i][0],data_array[i][1]) for i in range(0,len(data_array)) ]
		data_points = [2**i for i in range(2,len(data_array)+2)]
		plt.errorbar(data_points[0:12],data_compare[0:12],data_compare_std[0:12],color="black",linewidth=2,linestyle="--",label=sys.argv[15])
	#plt.fill_between(data_points, np.array(data_tree[i]) - np.array(data_tree_std[i]), np.array(data_tree[i]) + np.array(data_tree_std[i]),color=colors[4*(i+1)], alpha=0.2)
# plt.errorbar(data_points,data_binary_mean,data_binary_std,label='Binary Tree Fidelity')
# plt.fill_between(data_points,data_min, data_max,alpha=0.2, edgecolor='#CC4F1B', facecolor='#FF9848')
#plt.yscale('log')
#plt.title('Binary Tree Fidelity Scaling - '+sys.argv[6])
plt.xscale('log')
plt.xlabel('Number of Memory Qubits')
#plt.vlines(x=data_points[-1], ymin=0, ymax=data_tree[0][-1], linewidth=1, linestyle='-.',color=colors[6],alpha=0.7)
for i in range(0,len(PLOT_GUIDES)):
	plt.hlines(y=data_tree[i][-1], xmin=0, xmax=10000000, linewidth=1, linestyle='-.',color=colors[4*(i+1)],alpha=0.7)
	#plt.annotate(str(round(data_tree[i][-1],4)), xy = (float(data_points[-1])*1.1,data_tree[i][-1]*1.0001), xytext = (float(data_points[-1])*1.1, data_tree[i][-1]*1.0001), color=colors[4*(i+1)])
plt.xlim([3,8*2**10])
plt.ylim([-0.05,1.05])
plt.subplots_adjust(left=0.15, bottom=None, right=0.98, top=0.98, wspace=None, hspace=None)
#plt.yscale('log')

plt.yscale(sys.argv[5])
plt.ylabel('Fidelity')
#plt.ylim([0.9825, 1.001])

textstr = '$P_d = $' + str("{:.1%}".format(1/2**(int(PLOT_GUIDES[0][0][3])-1))) + "\n"

if PLOT_GUIDES[0][2] > 2:
	textstr += "$T_2$ = " + str("{:.0f}".format(T2[PLOT_GUIDES[0][2]]/10**9)) + ' s'
else:
	textstr += "$T_2$ = " + str("{:.0f}".format(T2[PLOT_GUIDES[0][2]]/10**6)) + ' ms'


props = dict(boxstyle='round', facecolor='orchid', alpha=0.3)

plt.text(6, 0.93, textstr, verticalalignment='top', bbox=props)


plt.legend(loc=3)
# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='-' , alpha=0.7)

# Show the minor grid lines with very faint and almost transparent grey lines
#plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#plt.minorticks_on()


plt.savefig( sys.path[0] + "/../Plots/" + sys.argv[12] + '.pdf',dpi=300,transparent=True)
plt.show()

#print("\nData summary:\n", data_array)