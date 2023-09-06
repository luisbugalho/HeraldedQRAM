# README.md

Simulations:

run qRAM_repeater.py

file nomenclature: P1_a_P2_b

We vary the P1 and P2 as they are what fundamentally changes the simulations themselves

	P1 - value of hybridization, a in float[0.0, 1.0]

		P1 -> 1 : fully probabilistic
		P1 -> 0 : fully deterministic

	P2 - value of efficiency, b in int[0,4]
		
		P2 = 0 :  0.966*0.966*0.936 
		P2 = 1 :  0.5 
		P2 = 2 :  0.6 
		P2 = 3 :  0.7  
		P2 = 4 :  0.8 

Plots and Treating Data:

file nomenclature: P1_a_P2_b_t1_t2_cnot

	t1 - Value for T1 damping time

		t1 = 0 : inf
		t1 = 1 : 2 . 10**7
		t1 = 2 : 2 . 10**8

	t2 - Value for T2 dephasing time

		t2 = 0 : inf
		t2 = 1 : 10**7
		t2 = 2 : 10**8

	cnot - Value for cnot error

		cnot = 0 : 0
		cnot = 1 : 0.01
		cnot = 2 : 0.001
		cnot = 3 : 0.0001
		cnot = 4 : 0.00001


Repeater Smart Scheme - above the N1th distribution layer its deterministic:

run qRAM_repeater_smart.py

file nomenclature: N1_a_P2_b

We vary the N1 and P2 as they are what fundamentally changes the simulations themselves

	N1 - value of hybridization, a in int(2,N_layers)

		N1 -> 2 : above 2nd layer is all detemrinistic ~ 25% deterministic
		N1 -> 3 : above 3rd layer is all detemrinistic ~ 12.5% deterministic
		N1 -> 4 : above 4th layer is all detemrinistic ~ 6.125% deterministic
		...

	P2 - value of efficiency, b in int[0,4]
		
		P2 = 0 :  0.966*0.966*0.936 ~ 0.9
		P2 = 1 :  0.5 
		P2 = 2 :  0.6 
		P2 = 3 :  0.7  
		P2 = 4 :  0.8 

Plots and Treating Data:

file nomenclature: N1_a_P2_b_t1_t2_cnot

	t1 - Value for T1 damping time

		t1 = 0 : inf
		t1 = 1 : 2 . 10**7
		t1 = 2 : 2 . 10**8

	t2 - Value for T2 dephasing time

		t2 = 0 : inf
		t2 = 1 : 10**7
		t2 = 2 : 10**8

	cnot - Value for cnot error

		cnot = 0 : 0
		cnot = 1 : 0.01
		cnot = 2 : 0.001
		cnot = 3 : 0.0001
		cnot = 4 : 0.00001

