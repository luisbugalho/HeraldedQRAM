import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os



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

def ft(t1):
	return 1-math.exp(-t1)

def fp(p1):
	return (1 + (1-p1)**2)/2

def f1(t1,t2):
	return 1 - ft(t1) - ft(t2) + 2*ft(t1)*ft(t2)

def f2(t1,t2,p1):
	return (1 + (1-p1)*f1(t1,t2))/2

def f2n(pairen,p1):
	t1 = pairen[0]
	t2 = pairen[1]
	return (1 + (1-p1)*f1(t1,t2))/2

def f3(t1,t2,p1):
	return (1-f2(t1,t2,p1))/f2(t1,t2,p1)

def f3n(pairen,p1):
	t1 = pairen[0]
	t2 = pairen[1]
	return (1-f2(t1,t2,p1))/f2(t1,t2,p1)

def h(pn):
	return (1-pn/2)**2-pn*(1-pn/2)

def max_func(a,b):
	if a>b:
		return a
	else:
		return b

layers = 12
T1 = [math.inf, 2*10**7, 2*10**8, 2*10**9, 2*10**10]
T2 = [math.inf, 10**7, 10**8, 10**9, 10**10]
CNOT_ERRORS = [0, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
cnot_fidelity =  [1-err for err in CNOT_ERRORS]
CNOT_ERROR = [4 / 3 * (1 - err ** 0.5) for err in cnot_fidelity]


file_directory = sys.argv[1]


data_tree = []
data_tree_std = []
# data_layers = [] #all data for each layer
# data_layers_std = [] #all data for each layer
# data_tree = [] #data for each tree
# data_tree_std = [] #data for each tree
data_points = [2**i for i in range(2,layers+1)]

flag_T1 = int(sys.argv[2])
flag_T2 = int(sys.argv[3])
flag_CNOT = int(sys.argv[4])

PLOT_GUIDES = (T1[flag_T1],T2[flag_T2],CNOT_ERROR[flag_CNOT])

data_array = []
for i in range(2,layers+1):
	if not os.path.exists(sys.path[0] + '/../All_Data_Raw_New/' + file_directory + '/qRAM_teleportation_fidelities_'+str(i) + '.npy'):
		continue
	data_unstructured = np.load(sys.path[0] + '/../All_Data_Raw_New/' + file_directory + '/qRAM_teleportation_fidelities_'+str(i) + '.npy',allow_pickle=True)
	data_fidelity = {layer: [] for layer in range(1, i+1)}

	#print(data_unstructured)
	#print(len(corrected))

	for k in range(0,len(data_unstructured)):

		noisedict_eletron = data_unstructured[k]['time_electron']
		noisedict_nuclear = data_unstructured[k]['time_nuclear']
		dict_eletron = data_unstructured[k]['positions_electron']
		dict_nuclear = data_unstructured[k]['positions_nuclear']
		noisedict_CNOT_e = data_unstructured[k]['ncnots_e']
		noisedict_CNOT_n = data_unstructured[k]['ncnots_n']

		t1 = PLOT_GUIDES[0]
		t2 = PLOT_GUIDES[1]
		cnot_error = PLOT_GUIDES[2]
         
		treedict_a = {(j,k): [0.,0.] for j in range(1, i + 1) for k in range(0, 2 ** (int(j - 1)) )} # every pair used indexed by the starting qubit, hence 2**layer -1

		treedict_cnots = {(j,k): False for j in range(1, i + 1) for k in range(0, 2 ** (int(j - 1)) )} # every pair used indexed by the starting qubit, hence 2**layer -1

		treedict_b = {(j,k): 0. for j in range(1, i + 1) for k in range(0, 2 ** (int(j - 1)) + 1)} # every node of the tree

		#print(treedict_a)
		#print(treedict_b)


		for qubit in dict_eletron:
			if dict_eletron[qubit][3] == 1:
				treedict_b[(dict_eletron[qubit][1],dict_eletron[qubit][0])] += noisedict_eletron[qubit][1]/t1 # if adding after GHZ state is distributed continue 
				#print("Adding time no electron: " + str(noisedict_eletron[qubit][1]/t1) )
				continue

			if dict_eletron[qubit][2] == "right": # Right Pair Qubits 
				treedict_a[(dict_eletron[qubit][1],dict_eletron[qubit][0])][0] += noisedict_eletron[qubit][0]/t1
				#print("Adding time no electron: " + str(noisedict_eletron[qubit][0]/t1) )

			if dict_eletron[qubit][2] == "left": # Left Pair Qubits
				treedict_a[(dict_eletron[qubit][1],dict_eletron[qubit][0]-1)][1] += noisedict_eletron[qubit][0]/t1
				#print("Adding time no electron: " + str(noisedict_eletron[qubit][0]/t1) )	

			if qubit in noisedict_CNOT_e:
				treedict_cnots[(dict_eletron[qubit][1],dict_eletron[qubit][0]-1)] = True

		
		for qubit in dict_nuclear:
			#print(qubit)

			if dict_nuclear[qubit][3] == 1 and qubit in noisedict_nuclear and qubit is not None:
				treedict_b[(dict_nuclear[qubit][1],dict_nuclear[qubit][0])] += noisedict_nuclear[qubit][1]/t1/100
				#print("Adding time no electron: " + str(noisedict_nuclear[qubit][1]/t1) )

			#print("Something here")


		for j in range(1, i+1):
			total_time_electron= 0.
			total_time_nuclear = 0.
			term1 = 1
			term2 = 0

			num_cnots_e = 0
			num_cnots_n = 0

			for key in dict_eletron:
			    if(dict_eletron[key][1]==j):
			        total_time_electron += noisedict_eletron[key][0] + noisedict_eletron[key][1]
			        if(key in noisedict_CNOT_e):
			            num_cnots_e += noisedict_CNOT_e[key]

			for key in dict_nuclear:
			    if(dict_nuclear[key][1]==j) and key is not None:
			        total_time_nuclear += noisedict_nuclear[key][0] + noisedict_nuclear[key][1]
			        if(key in noisedict_CNOT_n):
			            num_cnots_n += noisedict_CNOT_n[key]

			for key in treedict_a:
				if key[0] == j:
					term1 = term1*f2n(treedict_a[key],treedict_cnots[key]*cnot_error)

			for key in treedict_b:
				if key[0] == j:
					term2_aux = term1*(1-ft(treedict_b[key]))

					if(key in treedict_a):
						term2_aux = term2_aux*f3n(treedict_a[key],cnot_error)
					if((key[0],key[1]-1) in treedict_a):
						term2_aux = term2_aux*f3n(treedict_a[(key[0],key[1]-1)],cnot_error)

					term2 += term2_aux

			num_cnots_tot = num_cnots_n + num_cnots_e

			term3 = ((1-cnot_error)**2 + cnot_error/2*(1-cnot_error/2))**num_cnots_e
			#print("term1: " + str(term1))
			#print("term2: " + str(term2))
			#print("term3: " + str(term3))


			fidelity00 = max_func(1/2*term3*(term1+term2),0) #term1 -> pairs fidelity including damping+ electronic cnots errors // term2 -> final damping errors // term3 -> first order approximation of nuclear cnots errors
			fidelity01 = 1/2*math.exp(-total_time_electron*(1/t2 + 1/(2*t1) + 1/(2*t1) )  - total_time_nuclear/100*(1/t2 + 1/(2*t1) +1/(2*t1)))*(1-cnot_error)**(num_cnots_e+num_cnots_n/2)*(1-cnot_error+cnot_error**2/2)**(num_cnots_n/2) # antidiagonal entries  = 1/t2 + 1/(2*t1) -> dephasing + sqrt(exp(-dt/t1)) -> damping + (1-p)**n -> depolarising CNOTs
			fidelity10 = fidelity01
			fidelity11 = 1/2*max_func(term1*term3,0)

			#print("fidelity00: " + str(fidelity00))
			#print("fidelity01: " + str(fidelity01))
			#print("fidelity10: " + str(fidelity10))
			#print("fidelity11: " + str(fidelity11))


			fidelity = 1/2*(fidelity00+fidelity01+fidelity10+fidelity11)
			data_fidelity[j] += [fidelity]
		#print(data_fidelity)


	print('--------------------------')
	data_std_aux = []
	data_mean_aux = []
	for j in range(1,i+1):
		data_std_aux.append(std_dict(data_fidelity,j))
		data_mean_aux.append(mean_dict(data_fidelity,j))
	#print(data_mean_aux)
	data_array.append([data_mean_aux,data_std_aux])

pathname = sys.path[0] + "/../All_Data_Processed_New/" + file_directory
if not os.path.exists(pathname):
    os.makedirs(pathname)

np.save(pathname + "/" + file_directory + "_" + str(flag_T1)  + "_" + str(flag_T2) + "_" + str(flag_CNOT) , data_array)
print("writing to " + "/../All_Data_Processed_New/" + file_directory + "_" + str(flag_T1)  + "_" + str(flag_T2) + "_" + str(flag_CNOT))
#file with 


