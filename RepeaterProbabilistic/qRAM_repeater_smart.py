# Adding duration of gates in the decoherence of the memory
# Adding duration of gates in the decoherence of the memory

import sys
import random as rd
import os

from numpy import linalg as LA
from scipy.stats import bernoulli
import math
import netsquid as ns
import numpy as np
import netsquid.qubits as nq
from netsquid.qubits import qubitapi as qapi
from netsquid.util.simtools import sim_time
import pydynaa as pydynaa
import scipy
import scipy.stats
from netsquid.components.models.qerrormodels import DepolarNoiseModel, FibreLossModel, T1T2NoiseModel, DephaseNoiseModel
from netsquid.components.component import Message, Port
from netsquid.components.qmemory import QuantumMemory
from netsquid.nodes import Node
from netsquid.protocols.nodeprotocols import NodeProtocol, LocalProtocol, TimedNodeProtocol
from netsquid.protocols.protocol import Signals, Protocol
from netsquid.components.instructions import INSTR_SWAP
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.qprocessor import QuantumProcessor, PhysicalInstruction, QuantumProgram
from netsquid.qubits import ketstates as ks, QFormalism
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel
from netsquid.components.qchannel import QuantumChannel
from netsquid.components.cchannel import ClassicalChannel
from netsquid.nodes.network import Network
from pydynaa import EventExpression
import netsquid.components.instructions as instr
from netsquid.components.models.qerrormodels import QuantumErrorModel
from netsquid.qubits import operators as ops
from netsquid.qubits.ketstates import  s0
from netsquid.qubits.sparsedmtools import SparseDMRepr

from operator import add


#ns.set_qstate_formalism(QFormalism.SPARSEDM)
#ns.set_qstate_formalism(QFormalism.DM)
ns.set_qstate_formalism(QFormalism.STAB)

def is_zero_state(state):
    qaux, = qapi.create_qubits(1,system_name = "QAUX")
    #print(state.qstate.qrepr)
    if(qaux.qstate.compare(state.qstate)):
        #print('skipping noise')
        return True

    return False

def counted(f):
    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        #print(args)
        #print(kwargs)
        return f(*args, **kwargs)
    wrapped.calls = 0
    wrapped.args = []
    return wrapped

def empty_null_entries(dicto1,dicto2):
    empty_keys = [k for k,v in dicto1.items() if v == [0.,0.]]
    for k in empty_keys:
        del dicto1[k]
        if k in dicto2:
            del dicto2[k]


def make_tree(layer,n_deterministic):
    tree = {i:{"child_left":None,"child_right":None,"parent":None,"deterministic":False} for i in range(1,2**layer+1)}

    #Main tree for swapping
    for i in range(layer,0,-1):
        for j in range(1+2**i,2**(layer),2**(i+1)):
            tree[j]["child_left"] = j-max(1,2**(i-1))
            tree[j]["child_right"] = j+max(1,2**(i-1))
            tree[j-max(1,2**(i-1))]["parent"] = j
            tree[j+max(1,2**(i-1))]['parent'] = j
            if i >= n_deterministic-1:
                tree[j]["deterministic"] = True

    #Base of the tree for generation
    for i in range(2,2**layer+1,2):
        tree[i]['child_left'] = i-1
        tree[i]['child_right'] = i+1
    
    #End nodes
    tree[1]['parent'] = 2
    tree[2**layer]['child_left'] = None
    tree[2**layer]['child_right'] = None

    return tree


class DepolarNoiseModel2(DepolarNoiseModel):
    @counted
    def error_operation(self, *args, **kwargs):
        #print('Applying CNOT noise1 to: ' + str(args) + ' + kwargs: ' +str(kwargs))
        return DepolarNoiseModel.error_operation(self, *args, **kwargs)

    def compute_model(self, *args, **kwargs):
        #print('Applying CNOT noise2 to: ' + str(args)+ ' + kwargs: ' +str(kwargs))
        return DepolarNoiseModel.compute_model(self, *args, **kwargs)

    def apply_noise(self, *args, **kwargs):
        #print('Applying CNOT noise3 to: ' + str(args))

        return DepolarNoiseModel.apply_noise(self, *args, **kwargs)

class T1T2NoiseModel2(T1T2NoiseModel):
    @counted
    def error_operation(self, *args, **kwargs):
        return T1T2NoiseModel.error_operation(self, *args, **kwargs)

    def compute_model(self, *args, **kwargs):
        return T1T2NoiseModel.compute_model(self, *args, **kwargs)

    def apply_noise(self, *args, **kwargs):
        #print('Applying noise to: ' + str(args[0]))
        #print(qapi.reduced_dm(args[0]))
        if(self == noise_e):
            list_add = [0,0]
            if args[0] in dict_eletron:
                list_add[dict_eletron[args[0]][3]] = int(not is_zero_state(args[0]))*args[1]
            noisedict_eletron[args[0]] = list( map(add, noisedict_eletron.get(args[0],[0.,0.]), list_add)) 

        if(self == noise_n):
            list_add = [0,0]
            if args[0] in dict_nuclear:
                list_add[dict_nuclear[args[0]][3]] = int(not is_zero_state(args[0]))*args[1]
            noisedict_nuclear[args[0]] = list( map(add, noisedict_nuclear.get(args[0],[0.,0.]), list_add))

        return T1T2NoiseModel.apply_noise(self, *args, **kwargs)

largest_layer = int(sys.argv[1])
n_value = int(sys.argv[2]) # Above the N layer of each layer nodes are deterministic
p_efficiency = int(sys.argv[3])


trees = {}
trees[0] = {}
trees[0][1] = {"child_left":None,"child_right":None,"parent":None,"deterministic":False}
for i in range(1,largest_layer+1):
    trees[i] = make_tree(i,n_value)
#print(trees)
reset_time = 5000  # 5000 nano second
gate_time = 32
interaction_time = 0.1
list_efficiency = [ 0.966*0.966*0.936 , 0.490517624657570 ,  0.519643623260618 ,  0.551480633221287  , 0.586244711212384 ]
efficiency = list_efficiency[p_efficiency]
efficiency_bob = 1
channel_loss = 1
n_nuclear_spins = 1

T1 = [math.inf, 10**7, 10**8]
T2 = [math.inf, 10**7, 10**8]
t1_n = T1 * 100
t1_e = T1 
t2_n = [t2 * 100 for t2 in T2]
t2_e = T2
noise_p = T1T2NoiseModel()
noise_e = T1T2NoiseModel2(T2=t2_e[1])
noise_n = T1T2NoiseModel2(T2=t2_n[1])
noisedict_eletron = {}
dict_eletron = {}
noisedict_nuclear = {}
dict_nuclear = {}
noisedict_CNOT_e = {}
noisedict_CNOT_n = {}
# T2=65.1 * 10 ** 6
cnotn_time = 16000  # 16000
cnote_time = 29
CNOT_ERRORS = [0, 0.01, 0.001, 0.0001, 0.00001]
cnot_fidelity =  [1-err for err in CNOT_ERRORS]
CNOT_ERROR = [4 / 3 * (1 - err ** 2) for err in cnot_fidelity]
CNOTNoise1 = DepolarNoiseModel2(depolar_rate=0, time_independent=True)
CNOTNoise2 = DepolarNoiseModel2(depolar_rate=0, time_independent=True)
distance = .00001
speed = 2e8
num_round = 100



class EmitProtocol(NodeProtocol):   
    def __init__(self, node, index, layer):
        super().__init__(node=node, name='emit' + str(index) + 'layer_' + str(layer))
        # self.resetting = False
        self.name_left = str(index) + 'layer_' + str(layer) + 'left'
        self.name_right = str(index) + 'layer_' + str(layer) + 'right'
        self.ready = False
        self.receive_emit = False
        self.layer = int(layer)
        self.index = int(index)
        self.add_signal("success0")
        for i in range(1, 2 ** (self.layer - 1) + 1):
            self.add_signal('emit' + str(i))
            self.add_signal("reset" + str(i))
            self.add_signal("move" + str(i))
            self.add_signal("success" + str(i))
        self.add_signal("restart")
        self.add_signal("ready")
        self.state = 0
        self.success = 0
        self.qubit_e_left = None
        self.qubit_e_right = None

    def run(self):
        if self.layer == 1:
            while True:
                if self.state == 0:
                    self.state = 1
                if self.state == 1:
                    if self.node.subcomponents["QuantumMemory" + self.name_right].busy:
                        yield self.await_program(self.node.subcomponents["QuantumMemory" + self.name_right])

                    self.send_signal("success" + str(self.index))
                    self.success = 1

                    yield self.await_signal(circuit_protocol, "restart")
                    self.success = 0
                    self.state = 0

        elif int(self.index) < 2**(self.layer-1): # does not receive a photon in cavity protocol, only entangling protocol
            while True:
                if self.state == 0:
                    reset_expr = self.await_signal(circuit_protocol.subprotocols['rcv' + str(self.index + 1) + 'layer_' + str(self.layer)],"reset" + str(self.index))
                    yield reset_expr
                    if self.node.subcomponents["QuantumMemory" + self.name_right].busy:
                        yield self.await_program(self.node.subcomponents["QuantumMemory" + self.name_right])
                    self.reset_cavity_electron(flag_delete=True)
                    self.state = 1


                if self.state == 1:
                    program_expr = self.await_program(self.node.subcomponents["QuantumMemory" + self.name_right])
                    emit_expr = self.await_signal(circuit_protocol.subprotocols['rcv' + str(self.index + 1) + 'layer_' + str(self.layer)],"emit" + str(self.index))
                    expr_pe = program_expr | emit_expr
                    yield expr_pe
                    if expr_pe.second_term.value:
                        self.receive_emit = True
                    if expr_pe.first_term.value:
                        self.ready = True
                    if self.receive_emit and self.ready:
                        self.state = 2
                        self.receive_emit = False
                        self.ready = False

                if self.state == 2:
                    # emit to the next node
                    self.emit()
                    
                    if self.node.subcomponents["QuantumMemory" + self.name_right].busy:
                        yield self.await_program(self.node.subcomponents["QuantumMemory" + self.name_right])

                    #creating a map of qubits for each cavity
                    self.qubit_e_right, = self.node.subcomponents["QuantumMemory" + str(self.name_right)].peek(positions=[1])
                    dict_eletron[self.qubit_e_right] = [self.index,self.layer,"right",0]

                    photon, = self.node.subcomponents["QuantumMemory"+ self.name_right].pop(0)
                    #print(self.name_right + ": poping photon " + str(photon) )
                    self.node.ports["port_" + str(self.index) + str(self.index + 1) + "layer_" + str(self.layer)].tx_output(photon)
                    #print(self.name_right + str(self.node.subcomponents["QuantumMemory" + self.name_right].used_positions))

                    self.state = 3
                    self.send_signal("success" + str(self.index))
                    self.success = 1

                if self.state == 3:
                    yield self.await_signal(circuit_protocol.subprotocols['rcv' + str(self.index + 1) + 'layer_' + str(self.layer)],"reset" + str(self.index))
                    if self.node.subcomponents["QuantumMemory" + self.name_right].busy:
                        yield self.await_program(self.node.subcomponents["QuantumMemory" + self.name_right])
                    self.reset_cavity_electron(flag_delete=True)
                    self.state = 1
                    self.success = 0

    def reset_cavity_electron(self,flag_delete=False,flag_no_add=False):  # prepare |0> + |1>, place holder
        if(flag_delete):
            if(self.qubit_e_right != None and self.qubit_e_right in dict_eletron): 
                noisedict_eletron.pop(self.qubit_e_right)
                dict_eletron.pop(self.qubit_e_right)

        self.node.subcomponents["QuantumMemory" + self.name_right].execute_program(self.reset_program_electron(),qubit_mapping=[1])       

    class reset_program_electron(QuantumProgram):
        default_num_qubits = 1

        def program(self):
            q0, = self.get_qubit_indices(1)
            self.apply(instr.INSTR_INIT, [q0])
            self.apply(instr.INSTR_H, [q0])
            yield self.run(parallel=True)

    def emit(self):
        #print("emiting")
        photon, = nq.create_qubits(1)
        self.node.subcomponents["QuantumMemory" + self.name_right].put(photon, [0])
        self.node.subcomponents["QuantumMemory" + self.name_right].execute_instruction(instr.INSTR_CNOT, [1, 0],parallel=True)


class ReceiveProtocol(NodeProtocol):    
    def __init__(self, node, index, layer):
        super().__init__(node=node, name='rcv' + str(index) + 'layer_' + str(layer))
        # self.resetting = False
        self.name_left = str(index) + 'layer_' + str(layer) + 'left'
        self.name_right = str(index) + 'layer_' + str(layer) + 'right'
        self.ready = False
        self.receive_emit = False
        self.layer = int(layer)
        self.index = int(index)
        self.add_signal("success0")
        for i in range(1, 2 ** (self.layer - 1) + 1):
            self.add_signal('emit' + str(i))
            self.add_signal("reset" + str(i))
            self.add_signal("move" + str(i))
            self.add_signal("success" + str(i))
        self.add_signal("restart")
        self.add_signal("ready")
        self.state = 0
        self.success = 0
        self.qubit_e_left = None
        self.qubit_e_right = None

    def run(self):
        if  self.index == 1:
            while True:
                if self.state == 0:
                    self.state = 1
                if self.state == 1:
                    if self.node.subcomponents["QuantumMemory" + self.name_left].busy:
                        yield self.await_program(self.node.subcomponents["QuantumMemory" + self.name_left])

                    self.success = 1
                    self.send_signal("success" + str(self.index))
                    yield self.await_signal(circuit_protocol, "restart")
                    self.state = 0

        if int(self.index) > 1 : # does not receive a photon in cavity protocol, only entangling protocol
            while True:
                if self.state == 0:
                    self.send_signal("reset" + str(self.index-1))
                    #print(str(self.name_left) + ": sending reset signal to emit protocol " )
                    if self.node.subcomponents["QuantumMemory" + self.name_left].busy:
                        yield self.await_program(self.node.subcomponents["QuantumMemory" + self.name_left]) 
                    self.reset_cavity_electron(flag_delete=True)
                    self.state = 1


                if self.state == 1:
                    if self.node.subcomponents["QuantumMemory" + self.name_left].busy:
                        yield self.await_program(self.node.subcomponents["QuantumMemory" + self.name_left])
                        
                    #creating a map of qubits for each cavity
                    self.qubit_e_left, = self.node.subcomponents["QuantumMemory" + str(self.name_left)].peek(positions=[1])
                    dict_eletron[self.qubit_e_left] = [self.index,self.layer,"left",0]    

                    self.send_signal("emit" + str(self.index - 1))
                   
                    yield self.await_port_input(self.node.ports["port_" + str(self.index) + str(self.index - 1) + "layer_" + str(self.layer)])
                    photon = self.node.ports["port_" + str(self.index) + str(self.index - 1) + "layer_" + str(self.layer)].rx_input().items[0]

                    m = bernoulli.rvs(efficiency)  # with probability detection_efficiency being 1, likely to be 1

                    if m == 0:
                        self.state = 0
                    else:
                        if self.node.subcomponents["QuantumMemory" + self.name_left].busy:
                            yield self.await_program(self.node.subcomponents["QuantumMemory" + self.name_left])
                        self.interact(photon)
                        if self.node.subcomponents["QuantumMemory" + self.name_left].busy:
                            yield self.await_program(self.node.subcomponents["QuantumMemory" + self.name_left])
                        self.send_signal("success" + str(self.index))

                        self.state = 2
                        self.success = 1

                if self.state == 2:
                    if self.index % 2 == 0:
                        yield self.await_signal(circuit_protocol.subprotocols['link' + str(self.index) + 'layer_' + str(self.layer)],"reset" + str(self.index))
                    else:
                        yield self.await_signal(circuit_protocol.subprotocols['link' + str(self.index-1) + 'layer_' + str(self.layer)],"reset" + str(self.index-1))
                    self.state = 0
                    self.success = 0

    def reset_cavity_electron(self,flag_delete=False,flag_no_add=False):  # prepare |0> + |1>, place holder
        if(flag_delete):
            if(self.qubit_e_left != None and self.qubit_e_left in dict_eletron): 
                noisedict_eletron.pop(self.qubit_e_left)
                dict_eletron.pop(self.qubit_e_left)

        self.node.subcomponents["QuantumMemory" + self.name_left].execute_program(self.reset_program_electron(),qubit_mapping=[1])       

    class reset_program_electron(QuantumProgram):
        default_num_qubits = 1

        def program(self):
            q0, = self.get_qubit_indices(1)
            self.apply(instr.INSTR_INIT, [q0])
            self.apply(instr.INSTR_H, [q0])
            yield self.run(parallel=True)


    def interact(self, photon):
        #print('interacting cavity')
        self.node.subcomponents["QuantumMemory" + self.name_left].put(photon, [0])
        self.node.subcomponents["QuantumMemory" + self.name_left].execute_program(self.interact_program(),qubit_mapping=[0, 1])

    class interact_program(QuantumProgram):
            default_num_qubits = 2

            def program(self):
                q2, q1 = self.get_qubit_indices(2)  # q1 electron, q2 photon
                self.apply(instr.INSTR_CNOT, [q1, q2])
                self.apply(instr.INSTR_MEASURE, q2, output_key="m1")
                yield self.run()
                if self.output["m1"][0] == 1:
                    self.apply(instr.INSTR_X, q1)
                yield self.run()


class LinkProtocol(NodeProtocol):    
    def __init__(self, node, index, layer):
        super().__init__(node=node, name= 'link' + str(index) + 'layer_' + str(layer))
        # self.resetting = False
        self.name_left = str(index) + 'layer_' + str(layer) + 'left'
        self.name_right = str(index) + 'layer_' + str(layer) + 'right'
        self.ready = False
        self.receive_emit = False
        self.layer = int(layer)
        self.index = int(index)
        self.add_signal("success0")
        for i in range(1, 2 ** (self.layer - 1) + 1):
            self.add_signal('emit' + str(i))
            self.add_signal("reset" + str(i))
            self.add_signal("move" + str(i))
            self.add_signal("success" + str(i))
        self.add_signal("restart")
        self.add_signal("ready")
        self.state = 0
        self.success = 0
        self.qubit_n = None
        self.qubit_e_left = None
        self.qubit_e_right = None
        self.left_ready = False
        self.right_ready = False
        self.flag_deterministic = trees[self.layer-1][self.index]["deterministic"]
        print(str(self.layer) + ' looking at index ' + str(self.index) + "and its deterministic: " + str(self.flag_deterministic))
        self.left_child = trees[self.layer-1][self.index]["child_left"] #-1 +1 due to the structure of the tree starting at 0-2^n
        self.right_child = trees[self.layer-1][self.index]["child_right"]
        self.parent = trees[self.layer-1][self.index]["parent"]

    def run(self):
        if self.layer == 1:
            while True:
                if self.state == 0:
                    self.state = 1
                if self.state == 1:
                    self.success = 1
                    self.send_signal("success" + str(self.index))
                    yield self.await_signal(circuit_protocol, "restart")
                    self.success = 0
                    self.state = 0
        
        elif int(self.index) == 1:
            while True:
                if self.state == 0:
                    self.state = 1
                if self.state == 1:
                    yield self.await_signal(circuit_protocol.subprotocols['rcv' + str(self.index+1) + 'layer_' + str(self.layer)],"success" + str(self.index+1))
                    self.success = 1
                    self.send_signal("success" + str(self.index))
                    yield self.await_signal(circuit_protocol.subprotocols["link" + str(self.index+1) + 'layer_' + str(self.layer)],"reset" + str(self.index+1))
                    self.send_signal("reset" + str(self.index))

                    self.success = 0
                    self.state = 0

        elif int(self.index) == 2**(self.layer-1):
            while True:
                if self.state == 0:
                    self.state = 1
                if self.state == 1:
                    yield self.await_signal(circuit_protocol.subprotocols['rcv' + str(self.index) + 'layer_' + str(self.layer)],"success" + str(self.index))
                    self.success = 1
                    self.send_signal("success" + str(self.index))
                    if self.layer == 2:
                        yield self.await_signal(circuit_protocol,"restart")
                    else:
                        yield self.await_signal(circuit_protocol.subprotocols["link" + str(self.index-1) + 'layer_' + str(self.layer)],"reset" + str(self.index-1))
                    
                    self.send_signal("reset" + str(self.index))

                    self.success = 0
                    self.state = 0

        else: # does not receive a photon in cavity protocol, only entangling protocol
            while True:
                if self.state == 0:

                    if self.index % 2 == 0: #at the first linking layer you listen for entanglement success from receive protocol
                        link_expr_left = self.await_signal(circuit_protocol.subprotocols["rcv" + str(self.index) + 'layer_' + str(self.layer)],"success" + str(self.index))
                        link_expr_right = self.await_signal(circuit_protocol.subprotocols["rcv" + str(self.right_child) + 'layer_' + str(self.layer)],"success" + str(self.right_child))
                        link_expr = link_expr_left | link_expr_right
                        yield link_expr

                        if link_expr.first_term.value:
                            self.left_ready = True
                        if link_expr.second_term.value:
                            self.right_ready = True

                    else: #at higher linking layers, you listen for previous linking successes
                        link_expr_left = self.await_signal(circuit_protocol.subprotocols["link" + str(self.left_child) + 'layer_' + str(self.layer)],"success" + str(self.left_child))
                        link_expr_right = self.await_signal(circuit_protocol.subprotocols["link" + str(self. ) + 'layer_' + str(self.layer)],"success" + str(self.right_child))
                        link_expr = link_expr_left | link_expr_right
                        yield link_expr

                        if link_expr.first_term.value:
                            self.left_ready = True
                        if link_expr.second_term.value:
                            self.right_ready = True

                    if self.left_ready and self.right_ready:
                        self.state = 1

                if self.state == 1:
                    
                    if not self.flag_deterministic :
                        m = bernoulli.rvs(efficiency)  # with probability detection_efficiency being 1, likely to be 1

                        if m == 0:
                            self.state = 0
                            self.left_ready = False
                            self.right_ready = False
                            self.send_signal("reset"+ str(self.index))
                        else:
                            if self.node.subcomponents["QuantumMemory" + self.name_left].busy:
                                yield self.await_program(self.node.subcomponents["QuantumMemory" + self.name_left])
                            if self.node.subcomponents["QuantumMemory" + self.name_right].busy:
                                yield self.await_program(self.node.subcomponents["QuantumMemory" + self.name_right])
                            self.link1p()
                            if self.node.subcomponents["QuantumMemory" + self.name_left].busy:
                                yield self.await_program(self.node.subcomponents["QuantumMemory" + self.name_left])
                            if self.node.subcomponents["QuantumMemory" + self.name_right].busy:
                                yield self.await_program(self.node.subcomponents["QuantumMemory" + self.name_right])

                            photon, = self.node.subcomponents["QuantumMemory" + self.name_left].pop(0)
                            self.link2p(photon)
                            if self.node.subcomponents["QuantumMemory" + self.name_left].busy:
                                yield self.await_program(self.node.subcomponents["QuantumMemory" + self.name_left])
                            if self.node.subcomponents["QuantumMemory" + self.name_right].busy:
                                yield self.await_program(self.node.subcomponents["QuantumMemory" + self.name_right])
                          
                            self.send_signal("success" + str(self.index))
                            self.success = 1
                            self.state = 2
                            
                    else:
                        if self.node.subcomponents["QuantumMemory" + self.name_left].busy:
                            yield self.await_program(self.node.subcomponents["QuantumMemory" + self.name_left])
                        if self.node.subcomponents["QuantumMemory" + self.name_right].busy:
                            yield self.await_program(self.node.subcomponents["QuantumMemory" + self.name_right])
                        self.qubit_e_left, = self.node.subcomponents["QuantumMemory" + str(self.name_left)].peek(positions=[1])
                        self.link1d()
                        noisedict_CNOT_e[self.qubit_e_left] = 1
                        if self.node.subcomponents["QuantumMemory" + self.name_left].busy:
                                yield self.await_program(self.node.subcomponents["QuantumMemory" + self.name_left])
                        if self.node.subcomponents["QuantumMemory" + self.name_right].busy:
                            yield self.await_program(self.node.subcomponents["QuantumMemory" + self.name_right])

                        self.qubit_n, = self.node.subcomponents["QuantumMemory" + str(self.name_right)].peek(positions=[2])
                        dict_nuclear[self.qubit_n] = [self.index,self.layer,None,1]

                        electron = self.node.subcomponents["QuantumMemory" + self.name_left].pop(1)
                        self.link2d(electron)
                        noisedict_CNOT_n[self.qubit_n] = 1

                        if self.node.subcomponents["QuantumMemory" + self.name_left].busy:
                            yield self.await_program(self.node.subcomponents["QuantumMemory" + self.name_left])
                        if self.node.subcomponents["QuantumMemory" + self.name_right].busy:
                            yield self.await_program(self.node.subcomponents["QuantumMemory" + self.name_right])

                        self.send_signal("success" + str(self.index))
                        self.success = 1
                        self.state = 2

                    
                if self.state == 2:
                        
                    self.left_ready = False
                    self.right_ready = False

                    #print(self.name + ": waiting for restart or link reset")
                    if self.index == 2**(self.layer -1)/2+1:
                        yield self.await_signal(circuit_protocol,"restart")
                        #print(self.name + ": restarting at link")
                        self.send_signal("reset"+ str(self.index))
                        self.state = 0
                        self.success = 0
                    else:
                        yield self.await_signal(circuit_protocol.subprotocols["link" + str(self.parent) + 'layer_' + str(self.layer)],"reset" + str(self.parent))
                        self.send_signal("reset"+ str(self.index))
                        self.state = 0
                        self.success = 0                            
    
    def link1p(self):
        photon, = nq.create_qubits(1)
        self.node.subcomponents["QuantumMemory" + self.name_left].put(photon, [0])
        self.node.subcomponents["QuantumMemory" + self.name_left].execute_instruction(instr.INSTR_CNOT, [1, 0],parallel=True)

    def link2p(self,photon):
        flag1 = False
        self.node.subcomponents["QuantumMemory" + self.name_right].put(photon, [0])
        self.node.subcomponents["QuantumMemory" + self.name_right].execute_program(self.interact_program(), [0,1])
        self.node.subcomponents["QuantumMemory" + self.name_left].execute_program(self.measure_program(), [1],flag_correction=flag1)
        if flag1:
            self.node.subcomponents["QuantumMemory" + self.name_right].execute_instruction(instr.X, [1],parallel=True)
      

    class interact_program(QuantumProgram):
        default_num_qubits = 2

        def program(self):
            q2, q1 = self.get_qubit_indices(2)  # q1 electron, q2 photon
            self.apply(instr.INSTR_CNOT, [q1, q2])
            self.apply(instr.INSTR_MEASURE, q2, output_key="m1")
            yield self.run(parallel=True)
            if self.output["m1"][0] == 1:
                self.apply(instr.INSTR_Z, q1)
            yield self.run(parallel=True)

    class measure_program(QuantumProgram):
        default_num_qubits = 1

        def program(self,flag_correction):
            q1 = self.get_qubit_indices(1)  # q1 electron, q2 photon
            self.apply(instr.INSTR_MEASURE_X, q1, output_key="m1")
            yield self.run(parallel=True)
            if self.output["m1"][0] == 1:
                flag_correction = True


    def link1d(self):
        nuclearq, = nq.create_qubits(1)
        self.node.subcomponents["QuantumMemory" + self.name_right].put(nuclearq, [2])
        self.node.subcomponents["QuantumMemory" + self.name_right].execute_program(self.transfer_program(), [1, 2])

            
    def link2d(self,electron):
        self.node.subcomponents["QuantumMemory" + self.name_right].put(electron, [1])
        self.node.subcomponents["QuantumMemory" + self.name_right].execute_program(self.link_program_deterministic(), [1, 2])


   
    
    class transfer_program(QuantumProgram):
        default_num_qubits = 2

        def program(self):
            q1, q2 = self.get_qubit_indices(2)  # q1 electron, q2 nuclear
            self.apply(CNOT_e, [q1, q2])
            self.apply(instr.INSTR_MEASURE_X, q1, output_key="m1")
            yield self.run()
            if self.output["m1"][0] == 1:
                self.apply(instr.INSTR_Z, q2)
            yield self.run()

    class link_program_deterministic(QuantumProgram):
        default_num_qubits = 2

        def program(self):
            q1, q2 = self.get_qubit_indices(2)  # q1 electron, q2 nuclear
            self.apply(CNOT_n, [q2, q1])
            self.apply(instr.INSTR_MEASURE, q1, output_key="m1")
            yield self.run()


class EntanglingProtocol(NodeProtocol):
    def __init__(self, node, name, role, layer):
        super().__init__(node=node, name=name)
        self.role = role
        self.layer = layer
        self.ready = False
        self.receive_emit = False
        for i in range(1, 2 ** (self.layer - 1) + 1):
            self.add_signal('emit' + str(i))
            self.add_signal("reset" + str(i))
            self.add_signal("move" + str(i))
            self.add_signal("success" + str(i))
        self.add_signal("restart")
        self.add_signal("entanglement success")
        self.add_signal("For Alice: entanglement success")
        self.add_signal("For Bob: entanglement success")
        self.add_signal("entanglement failure")
        self.add_signal("start entangling layer " + str(self.layer))

        self.state = 0
        self.success = 0
        self.qubit_e = None
        self.qubit_n = None

    def run(self):
        if self.role == "Alice":  # Alice is in qRAM, index is 1
            yield self.await_signal(circuit_protocol, "start entangling layer " + str(self.layer))

            while True:
                if self.state == 0:
                    self.reset_cavity(flag_delete=True)
                    self.state = 1

                if self.state == 1:
                    yield self.await_program(self.node.subcomponents["QuantumMemory1layer_" + str(self.layer) + "left"])

                    #creating a map of qubits for each cavity
                    self.qubit_e, self.qubit_n = self.node.subcomponents["QuantumMemory1layer_" + str(self.layer) + "left"].peek(positions=[1,2])
                    dict_eletron[self.qubit_e] = [1,self.layer,"left",0]
                    dict_nuclear[self.qubit_n] = [1,self.layer,"left",1]


                    self.emit()

                    if self.node.subcomponents["QuantumMemory1layer_" + str(self.layer) + "left"].busy:
                        yield self.await_program(self.node.subcomponents["QuantumMemory1layer_" + str(self.layer) + "left"])
                    photon, = self.node.subcomponents["QuantumMemory1layer_" + str(self.layer) + "left"].pop(0)
                    self.node.ports["port_10layer_" + str(self.layer)].tx_output(photon)
                    self.state = 2

                if self.state == 2:
                    success_exp = self.await_signal(circuit_protocol.subprotocols["Bob_layer_" + str(self.layer)],"For Alice: entanglement success")
                    failure_exp = self.await_signal(circuit_protocol.subprotocols["Bob_layer_" + str(self.layer)],"entanglement failure")
                    expr_sf = success_exp | failure_exp

                    yield expr_sf

                    if expr_sf.first_term.value:
                        self.transfer_Bell_state_from_electron_to_nuclear()

                        if self.node.subcomponents["QuantumMemory1layer_" + str(self.layer) + "left"].busy:
                            yield self.await_program(self.node.subcomponents["QuantumMemory1layer_" + str(self.layer) + "left"])
                        noisedict_CNOT_e[self.qubit_e] = 1
                        if self.layer > 1:
                            electron, = self.node.subcomponents["QuantumMemory1layer_" + str(self.layer) + "right"].pop(1)
                            self.node.subcomponents["QuantumMemory1layer_" + str(self.layer) + "left"].put(electron,[1])

                            self.link_Bell_state_to_create_GHZ()
                            
                            if self.node.subcomponents["QuantumMemory1layer_" + str(self.layer) + "left"].busy:
                                yield self.await_program(self.node.subcomponents["QuantumMemory1layer_" + str(self.layer) + "left"])

                            noisedict_CNOT_n[self.qubit_n] = 1

                        self.send_signal("For Bob: entanglement success")
                        yield self.await_signal(circuit_protocol, "restart")
                        self.qubit_e = None
                        self.qubit_n = None
                        self.state = 0
                        self.success = 0
                        yield self.await_signal(circuit_protocol, "start entangling layer " + str(self.layer))
                    if expr_sf.second_term.value:
                        self.state = 0

        if self.role == "Bob":  # Bob is in QC, node: node 0
            yield self.await_signal(circuit_protocol, "start entangling layer " + str(self.layer))

            while True:
                if self.state == 0:
                    self.state = 1
                    self.reset_cavity(flag_delete=True)

                if self.state == 1:
                    yield self.await_program(self.node.subcomponents["QuantumMemory0layer_" + str(self.layer) + "right"])  # Bob's memory is in node 1 of CavityProtocol

                    #creating a map of qubits for each cavity
                    self.qubit_e,self.qubit_n = self.node.subcomponents["QuantumMemory0layer_" + str(self.layer) + "right"].peek(positions=[1,2])
                    dict_eletron[self.qubit_e] = [0,self.layer,"right",0]
                    dict_nuclear[self.qubit_n] = [0,self.layer,"right",1]


                    timer_exp = self.await_timer(1e9 * distance / speed + 0.10000001)
                    photon_exp = self.await_port_input(self.node.ports["port_01layer_" + str(self.layer)])  # may need to add layer number
                    expr_tp = timer_exp | photon_exp
                    yield expr_tp

                    m = bernoulli.rvs(efficiency)  # with probability detection_efficiency being 1, likely to be 1

                    if expr_tp.first_term.value or m == 0:
                        yield self.await_timer(1e9 * distance / speed)
                        self.send_signal("entanglement failure")  # may need to add layer number
                        self.state = 0
                    if expr_tp.second_term.value and m == 1:
                        photon = self.node.ports["port_01layer_" + str(self.layer)].rx_input().items[0]

                        self.interact(photon)

                        if self.node.subcomponents["QuantumMemory0layer_" + str(self.layer) + "right"].busy:
                            yield self.await_program(self.node.subcomponents["QuantumMemory0layer_" + str(self.layer) + "right"])
                        yield self.await_timer(1e9 * distance / speed)
                        self.send_signal("For Alice: entanglement success")  # may need to add layer number
                        #self.transfer_Bell_state_from_electron_to_nuclear()
                        #noisedict_CNOT_e[self.qubit_e] = 1


                        yield self.await_signal(circuit_protocol.subprotocols["Alice_layer_" + str(self.layer)],"For Bob: entanglement success")
                        self.send_signal("entanglement success")  # may need to add layer number
                        self.success = 1


                        yield self.await_signal(circuit_protocol, "restart")
                        self.qubit_e = None
                        self.qubit_n = None
                        self.state = 0
                        self.success = 0
                        yield self.await_signal(circuit_protocol, "start entangling layer " + str(self.layer))

    def emit(self):
        photon, = nq.create_qubits(1)
        self.node.subcomponents["QuantumMemory1layer_" + str(self.layer) + "left"].put(photon, [0])
        self.node.subcomponents["QuantumMemory1layer_" + str(self.layer) + "left"].execute_instruction(instr.INSTR_CNOT, [1, 0])

    def reset_cavity_electron(self,flag_delete=False,flag_no_add=False):  # prepare |0> + |1>, place holder
        if(flag_delete):
            if(self.qubit_e != None): 
                noisedict_eletron.pop(self.qubit_e)
                dict_eletron.pop(self.qubit_e)
    
        self.node.subcomponents["QuantumMemory1layer_" + str(self.layer)].measure(positions=[1],skip_noise=True,discard=False)      
        self.node.subcomponents["QuantumMemory1layer_" + str(self.layer) + "left"].execute_program(self.reset_program_electron(),qubit_mapping=[2])

    class reset_program_electron(QuantumProgram):
        default_num_qubits = 1

        def program(self):
            q0, = self.get_qubit_indices(1)
            self.apply(instr.INSTR_INIT, [q0])
            self.apply(instr.INSTR_H, [q0])
            yield self.run()

    def reset_cavity(self,flag_delete=False):  # prepare |0> + |1>, place holder
        if self.role == "Bob":
            if(flag_delete):
                if(self.qubit_e != None): 
                    noisedict_eletron.pop(self.qubit_e)
                    dict_eletron.pop(self.qubit_e)
                if(self.qubit_n != None): 
                    noisedict_nuclear.pop(self.qubit_n)
                    dict_nuclear.pop(self.qubit_n)
            self.node.subcomponents["QuantumMemory0layer_" + str(self.layer) + "right"].execute_program(self.reset_program(),qubit_mapping=[1, 2])
        else:
            if(flag_delete):
                if(self.qubit_e != None): 
                    noisedict_eletron.pop(self.qubit_e)
                    dict_eletron.pop(self.qubit_e)
                if(self.qubit_n != None): 
                    noisedict_nuclear.pop(self.qubit_n)
                    dict_nuclear.pop(self.qubit_n)
            self.node.subcomponents["QuantumMemory1layer_" + str(self.layer) + "left"].execute_program(self.reset_program(),qubit_mapping=[1, 2])

    class reset_program(QuantumProgram):
        default_num_qubits = 2

        def program(self):
            q0, q1 = self.get_qubit_indices(2)
            self.apply(instr.INSTR_INIT, [q0, q1])
            self.apply(instr.INSTR_H, [q0])
            yield self.run()

    def transfer_Bell_state_from_electron_to_nuclear(self):
        #print('transfering entangling at layer ' + str(self.layer))
        if self.role == "Bob":
            self.node.subcomponents["QuantumMemory0layer_" + str(self.layer) + "right"].execute_program(self.transfer_program(),qubit_mapping=[1, 2])
        else:
            self.node.subcomponents["QuantumMemory1layer_" + str(self.layer) + "left"].execute_program(self.transfer_program(),qubit_mapping=[1,2])

    class transfer_program(QuantumProgram):
        default_num_qubits = 2

        def program(self):
            q1, q2 = self.get_qubit_indices(2)  # q1 electron, q2 nuclear
            self.apply(CNOT_e, [q1, q2])
            self.apply(instr.INSTR_MEASURE_X, q1, output_key="m1")
            yield self.run()
            if self.output["m1"][0] == 1:
                self.apply(instr.INSTR_Z, q2)
            yield self.run()

    def link_Bell_state_to_create_GHZ(self):
        self.node.subcomponents["QuantumMemory1layer_" + str(self.layer) + "left"].execute_program(self.link_program(),qubit_mapping=[1, 2])

    class link_program(QuantumProgram):
        default_num_qubits = 2

        def program(self):
            q1, q2 = self.get_qubit_indices(2)  # q1 electron, q2 nuclear
            self.apply(CNOT_n, [q2, q1])
            self.apply(instr.INSTR_MEASURE, q1, output_key="m1")
            yield self.run()

    def interact(self, photon):
        self.node.subcomponents["QuantumMemory0layer_" + str(self.layer) + "right"].put(photon, [0])
        self.node.subcomponents["QuantumMemory0layer_" + str(self.layer) + "right"].execute_program(self.interact_program(),qubit_mapping=[0, 1])

    class interact_program(QuantumProgram):
        default_num_qubits = 2

        def program(self):
            q2, q1   = self.get_qubit_indices(2)  # q1 electron, q2 photon
            self.apply(instr.INSTR_CNOT, [q1, q2])
            self.apply(instr.INSTR_MEASURE, q2, output_key="m1")
            yield self.run()
            if self.output["m1"][0] == 1:
                self.apply(instr.INSTR_X, q1)
            yield self.run()


class CircuitProtocol(LocalProtocol):
    def __init__(self, network):
        super().__init__(nodes=network.nodes, name="circuit")
        self.start_entangling = np.zeros(largest_layer)
        self.fidelities = [{} for i in range(0,num_round)]
        self.times = np.zeros(num_round)
        # self.debug_fidelities = np.zeros(num_round)
        self.counter = 0
        self.add_signal("restart")
        self.add_signal("layer success")
        self.add_signal("entanglement success")
        self.add_signal("entanglement failure")
        self.add_signal("stop")

        for j in range(1, largest_layer + 1):
            self.add_signal("start entangling layer " + str(j))
            for i in range(1, 2 ** (int(j - 1)) + 1): #only for the nodes in the binary tree!
                self.add_subprotocol(EmitProtocol(node=network.get_node("node_" + str(i) + 'layer_' + str(j)), index=i, layer=j))
                self.add_subprotocol(ReceiveProtocol(node=network.get_node("node_" + str(i) + 'layer_' + str(j)), index=i, layer=j))
                self.add_subprotocol(LinkProtocol(node=network.get_node("node_" + str(i) + 'layer_' + str(j)), index=i, layer=j))
                #print("adding to node " + str(i) + " in layer " + str(j))

        for j2 in range(1, largest_layer + 1):
            self.add_subprotocol(
                EntanglingProtocol(node=network.get_node("node_0layer_" + str(j2)), name="Bob_layer_" + str(j2),
                                   role='Bob', layer=j2))
            self.add_subprotocol(
                EntanglingProtocol(node=network.get_node("node_1layer_" + str(j2)), name="Alice_layer_" + str(j2),
                                   role='Alice', layer=j2))

    def run(self):
        success_expr = self.await_signal(circuit_protocol.subprotocols['link1layer_1'], "success0") # will never receive this

        for j in range(1, largest_layer + 1):
            for i in range(1, 2 ** (j - 1) + 1):
                success_expr = success_expr | self.await_signal(circuit_protocol.subprotocols['link' + str(i) + 'layer_' + str(j)], "success" + str(i))

        entangling_expr = self.await_signal(circuit_protocol.subprotocols['Bob_layer_1'], "entanglement success")
        for j in range(2, largest_layer + 1):
            entangling_expr = entangling_expr | self.await_signal(circuit_protocol.subprotocols['Bob_layer_' + str(j)],"entanglement success")
        expr_se = success_expr | entangling_expr

        while True:
            yield expr_se
            if expr_se.first_term.value:
                success_array = np.zeros(largest_layer)
                layer_success = 0
                for j in range(1, largest_layer + 1):
                    for i in range(1, 2 ** (j - 1) + 1):
                        success_array[j - 1] = success_array[j - 1] + circuit_protocol.subprotocols["link" + str(i) + 'layer_' + str(j)].success

                    if success_array[j - 1] == 2 ** (j - 1) and self.start_entangling[j - 1] < 0.5:
                        self.start_entangling[j - 1] = 1
                        print("big success! Layer" + str(j) + f" time {ns.sim_time()}")
                        layer_success = layer_success + 1
                        for ii in range(1, 2 ** (j - 1) + 1):
                            if circuit_protocol.subprotocols["link" + str(ii) + 'layer_' + str(j)].node.subcomponents["QuantumMemory" + str(ii) + 'layer_' + str(j) + "left"].busy:
                                yield self.await_program(circuit_protocol.subprotocols["link" + str(ii) + 'layer_' + str(j)].node.subcomponents["QuantumMemory" + str(ii) + 'layer_' + str(j) + "left"])
                            if circuit_protocol.subprotocols["link" + str(ii) + 'layer_' + str(j)].node.subcomponents["QuantumMemory" + str(ii) + 'layer_' + str(j) + "right"].busy:
                                yield self.await_program(circuit_protocol.subprotocols["link" + str(ii) + 'layer_' + str(j)].node.subcomponents["QuantumMemory" + str(ii) + 'layer_' + str(j) + "right"])
                        print("start entangling layer " + str(j))
                        self.send_signal("start entangling layer " + str(j))

            if expr_se.second_term.value:
                #print("entangling")
                entanglement_success = 0
                for j_index in range(1, largest_layer + 1):
                    entanglement_success = entanglement_success + circuit_protocol.subprotocols['Bob_layer_' + str(j_index)].success
                #print(entanglement_success)
                if entanglement_success == largest_layer:
                    for jj in range(1, largest_layer + 1):
                        if circuit_protocol.subprotocols['Bob_layer_' + str(jj)].node.subcomponents['QuantumMemory0layer_' + str(jj) + "right"].busy:
                            yield self.await_program(circuit_protocol.subprotocols['Bob_layer_' + str(jj)].node.subcomponents['QuantumMemory0layer_' + str(jj) + "right"])
                        if circuit_protocol.subprotocols['Alice_layer_' + str(jj)].node.subcomponents['QuantumMemory1layer_' + str(jj) + "left"].busy:
                            yield self.await_program(circuit_protocol.subprotocols['Alice_layer_' + str(jj)].node.subcomponents['QuantumMemory1layer_' + str(jj) + "left"])

                    # for i1 in range(1, largest_layer + 1):
                    #     for i2 in range(1, 2 ** (int(i1) - 1) + 1):
                    #         #print("peek-a-boo")
                    #         qubits, = circuit_protocol.subprotocols['rcv' + str(i2) + 'layer_' + str(i1)].node.subcomponents["QuantumMemory" + str(i2) + 'layer_' + str(i1) + "right"].peek(positions=[1])

                    
                    #for i in range(1, largest_layer+1):
                    #    qubits, = circuit_protocol.subprotocols['Bob_layer_' + str(i)].node.subcomponents["QuantumMemory0" + 'layer_' + str(i)].peek(positions=[2])
                 

                    empty_null_entries(noisedict_eletron,dict_eletron)
                    empty_null_entries(noisedict_nuclear,dict_nuclear)
                    fidelity = {'time_electron':noisedict_eletron.copy(),
                                'time_nuclear' :noisedict_nuclear.copy(),
                                'positions_electron' :dict_eletron.copy(),
                                'positions_nuclear' :dict_nuclear.copy(),
                                'ncnots_e' :noisedict_CNOT_e.copy(),
                                'ncnots_n' :noisedict_CNOT_n.copy()}

                    print("calculating fidelity")

                        
                    self.fidelities[self.counter] = fidelity
                    self.times[self.counter] = sim_time()
                    self.counter = self.counter + 1



                    for j_reset in range(1, largest_layer + 1):
                        for i_reset in range(1, 2 ** (int(j_reset) - 1) + 1):
                            circuit_protocol.subprotocols["link" + str(i_reset) + 'layer_' + str(j_reset)].node.subcomponents["QuantumMemory" + str(i_reset) + 'layer_' + str(j_reset) + "left"].reset()
                            circuit_protocol.subprotocols["link" + str(i_reset) + 'layer_' + str(j_reset)].node.subcomponents["QuantumMemory" + str(i_reset) + 'layer_' + str(j_reset) + "right"].reset()
                        circuit_protocol.subprotocols['Bob_layer_' + str(j_reset)].node.subcomponents["QuantumMemory0" + 'layer_' + str(j_reset) + "right"].reset()


                    
                    if self.counter == num_round:
                        
                        #print(self.fidelities)
                        print(self.times)
                        # print(self.debug_fidelities)
                        print(ns.sim_time())
                        pathname = "All_Data_Raw_New/N1_" + str(n_value) +  "_P2_" + str(p_efficiency)
                        if not os.path.exists(pathname):
                            os.makedirs(pathname)
                        np.save(pathname + "/qRAM_teleportation_fidelities_" + str(largest_layer), self.fidelities)
                        np.save(pathname + "/qRAM_teleportation_times_" + str(largest_layer), self.times)

                        yield self.await_signal(self, 'stop')

                    self.start_entangling = np.zeros(largest_layer)

                    print(f"restart {ns.sim_time()}")
                    print("trial: " + str(self.counter))
                    self.send_signal('restart')
                    self.await_timer(0.1)


                    noisedict_eletron.clear()
                    noisedict_nuclear.clear()
                    dict_eletron.clear()
                    dict_nuclear.clear()

                    noisedict_CNOT_e.clear()
                    noisedict_CNOT_n.clear()


network = Network("Circuit")
CNOT_n = instr.IGate("CNOT_n", ops.CNOT)
CNOT_e = instr.IGate("CNOT_e", ops.CNOT)
INIT_n = instr.IGate("INIT_n", instr.INSTR_INIT)

phys_instructions = [
    PhysicalInstruction(instr.INSTR_X, duration=gate_time,parallel=True),
    PhysicalInstruction(instr.INSTR_Z, duration=gate_time,parallel=True),
    PhysicalInstruction(instr.INSTR_H, duration=gate_time,parallel=True),
    PhysicalInstruction(instr.INSTR_INIT, duration=reset_time,parallel=True),
    PhysicalInstruction(INIT_n, duration=0,parallel=True),
    PhysicalInstruction(instr.INSTR_CNOT, duration=interaction_time,parallel=True),
    PhysicalInstruction(instr.INSTR_MEASURE, duration=0),
    PhysicalInstruction(instr.INSTR_MEASURE_X, duration=0),
    PhysicalInstruction(CNOT_n, duration=cnotn_time,q_noise_model=CNOTNoise1),
    PhysicalInstruction(CNOT_e, duration=cnote_time,q_noise_model=CNOTNoise2)]  

for j in range(1, largest_layer + 1):
    for i in range(0, 2 ** (int(j - 1)) + 1):
        network.add_nodes(["node_" + str(i) + 'layer_' + str(j)])

for j in range(1, largest_layer + 1):
    for i in range(0, 2 ** (int(j - 1)) + 1):
        print("QuantumMemory" + str(i) + 'layer_' + str(j))
        network.get_node("node_" + str(i) + 'layer_' + str(j)).add_subcomponent(
            QuantumProcessor("QuantumMemory" + str(i) + 'layer_' + str(j) + 'left', num_positions=3,
                             mem_noise_models=[noise_p, noise_e, noise_n],
                             fallback_to_nonphysical=False,
                             phys_instructions=phys_instructions))
        network.get_node("node_" + str(i) + 'layer_' + str(j)).add_subcomponent(
            QuantumProcessor("QuantumMemory" + str(i) + 'layer_' + str(j) + 'right', num_positions=3,
                             mem_noise_models=[noise_p, noise_e, noise_n],
                             fallback_to_nonphysical=False,
                             phys_instructions=phys_instructions))
    # Memory Left
    # position 0: photon left
    # position 1: electron left 
    # position 2: nuclear left
    
    # Memory Right
    # position 0: photon right
    # position 1: electron right
    # position 2: nuclear right

for j in range(1, largest_layer + 1):
    for i in range(1, 2 ** (int(j - 1)) + 1 ):
        if i == 1 :
            network.add_connection(network.get_node("node_" + str(i) + 'layer_' + str(j)),
                                   network.get_node("node_" + str(i - 1) + 'layer_' + str(j)),
                                   channel_to=QuantumChannel("qchannel_" + str(i) + str(i - 1) + 'layer_' + str(j)),
                                   label="q" + str(i) + str(i - 1) + 'layer_' + str(j),
                                   port_name_node1="port_" + str(i) + str(i - 1) + 'layer_' + str(j),
                                   port_name_node2="port_" + str(i - 1) + str(i) + 'layer_' + str(j))

        if j != 1 and i < 2 ** (int(j - 1)):
            network.add_connection(network.get_node("node_" + str(i) + 'layer_' + str(j)),
                                   network.get_node("node_" + str(i + 1) + 'layer_' + str(j)),
                                   channel_to=QuantumChannel("qchannel_" + str(i) + str(i + 1) + 'layer_' + str(j)),
                                   label="q" + str(i) + str(i + 1) + 'layer_' + str(j),
                                   port_name_node1="port_" + str(i) + str(i + 1) + 'layer_' + str(j),
                                   port_name_node2="port_" + str(i + 1) + str(i) + 'layer_' + str(j))
            

circuit_protocol = CircuitProtocol(network)

ns.sim_reset()
circuit_protocol.start()
circuit_protocol.start_subprotocols()
ns.sim_run(1000000000000)


