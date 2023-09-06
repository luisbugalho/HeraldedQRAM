# HeraldedQRAM

This repository simulates different types of architectures for a QRAM, based on initial works developed by Chen _et.al_ [1] ([10.1103/PRXQuantum.2.030319](https://www.doi.org/10.1103/PRXQuantum.2.030319)) and new ones introduced as a part of a recent work developed by Bugalho _et.al_ [2] ([arXiv:2210.13494](https://arxiv.org/abs/2210.13494))

The repository is organized in the following way:

Folders:
- SimulationsDeterministic: main directory for simulations based on the architecture of [1]
- RepeaterProbabilistic: main directory for simulations based on the architecture introduced in [2]

Within both folders, the structure is identical:
- main .py file with the code to execute the simulations of the architecture
- folder /scripts with all the scripts to run to produce the data, treat the data and plot the data, and also generate and run the scripts themselves
- folder /readData with the .py files to treat and plot the data
- folder /Plots with the plots presented in the paper
- folder /Data_Raw and /Data_Processed with the direct data from simulations and after doing the calculations of the fidelity respectively
- README document with additional information about the data structure



(Keep in mind this might not be the most efficient setup of data...)
