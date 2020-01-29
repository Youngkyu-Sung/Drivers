import numpy as np
from numpy import matmul as mul
from numpy.linalg import inv as inv
from numpy.linalg import eig as eig
from numpy import tensordot as tensor
from numpy import dot
import pickle
import random as rnd

import itertools
import sequence_rb 
import gates

import qutip as qt
# list of Paulis in string representation
list_sSign = ['+','-'] #
list_sPauli = ['I','X','Y','Z']
list_s2QBPauli = list(itertools.product(list_sSign,list_sPauli, list_sPauli))

# list of Paulis, 1QB-gates, and 2QB-gates in np.matrix representation
dict_mPauli = {'I': np.matrix('1,0;0,1'),
	'X': np.matrix('0,1;1,0'),
	'Y': np.matrix('0,-1j;1j,0'),
	'Z': np.matrix('1,0;0,-1')}

# Bloch sphere
# Rotation operator (rotate theta about P-axis)
# R(theta, P) = cos(theta/2)*I + sin(theta/2)*P

dict_m1QBGate = {'I': np.matrix('1,0;0,1'),
	'X2p': 1/np.sqrt(2)*np.matrix('1,-1j;-1j,1'),
	'X2m': 1/np.sqrt(2)*np.matrix('1,1j;1j,1'),
	'Y2p': 1/np.sqrt(2)*np.matrix('1,-1;1,1'),
	'Y2m': 1/np.sqrt(2)*np.matrix('1,1;-1,1'),
	'W2p': 1/np.sqrt(2)*np.matrix([[1,(1+1j)/np.sqrt(2)],[(-1+1j)/np.sqrt(2),1]]), #from X/2 (pi/2 rotation around the X axis), Y/2 (pi/2 rotation around the Y axis) and (X + Y)/2 (pi/2 rotation around an axis pi/4 away from the X on the equator of the Bloch sphere)
	'Z2p': np.matrix('1,0;0,1j'),
	'Z2m': np.matrix('1,0;0,-1j'),
	'Xp': np.matrix('0,-1j;-1j,0'),
	'Xm': np.matrix('0,1j;1j,0'),
	'Yp': np.matrix('0,-1;1,0'),
	'Ym': np.matrix('0,1;-1,0'),
	'Zp': np.matrix('1,0;0,-1'),
	'Zm': np.matrix('1,0;0,-1')
	}

dict_m2QBGate = {'SWAP': np.matrix('1,0,0,0; 0,0,1,0; 0,1,0,0; 0,0,0,1'),
	'CZ': np.matrix('1,0,0,0; 0,1,0,0; 0,0,1,0; 0,0,0,-1'),
	'iSWAP': np.matrix('1,0,0,0; 0,0,1j,0; 0,1j,0,0; 0,0,0,1'),
	'CNOT': np.matrix('1,0,0,0; 0,1,0,0; 0,0,0,1; 0,0,1,0')}


# def arb_gate(theta, )
def generate_XEB_circuit(generator = 'CZ', N_cycles = 10, rnd_seed = 0):

	# Generate 2QB XEB sequence
	gateSeq1 = []
	gateSeq2 = []

	rnd.seed(rnd_seed)
	
	if generator in ['iSWAP_Cplr', 'CZ_Cplr']:
		gateSeqCplr = []
	
	else:
		gateSeqCplr = None
	
	for j in range(N_cycles):
		# add random single qubit gates 
		rndnum1 = rnd.randint(0, 2)
		rndnum2 = rnd.randint(0, 2)
		sequence_rb.add_singleQ_forXEB(rndnum1, gateSeq1)
		sequence_rb.add_singleQ_forXEB(rndnum2, gateSeq2)

		if generator in ['iSWAP_Cplr', 'CZ_Cplr']:
			gateSeqCplr.append(gates.I)

		# add two qubit gate 
		if generator == 'CZ':
			gateSeq1.append(gates.I)
			gateSeq2.append(gates.CZ)
		elif generator == 'iSWAP_Cplr':
			gateSeq1.append(gates.I)
			gateSeqCplr.append(gates.iSWAP_Cplr)
			gateSeq2.append(gates.I)
		elif generator == 'CZ_Cplr':
			gateSeq1.append(gates.I)
			gateSeqCplr.append(gates.CZ_Cplr)
			gateSeq2.append(Gates.I)

	return gateSeq1, gateSeq2, gateSeqCplr

def calculate_XEB_fidelity(p_meas, p_sim, entropy_calc = "linear"):
		"""
		estimate XEB function F (F = <D*P_th,q - 1>, where D is the number of states in the Hilbert space, 
		P_th,q is the theoretical probability of a bit-string q at the end of the circuit, and
		<> corresponds to the ensemble average over all measured bit-strings.

		Parameters
		----------
		p_meas: 2-dimensional ndarray (num_of_randomizations, dim)
			The experimentally measured proability of a bit string 

		p_sim: 2-dimensional ndarray (num_of_randomizations, dim)
			The simulated proability of a bit string 

		entropy_calc: string
			linear: calculate linear cross-entropy (a.k.a. linear version of XEB)
			log: calculate logarithmic cross-entropy (a.k.a. standard XEB)
		"""

		# dimension (number of states)
		dim = p_meas.shape[-1]

		if (entropy_calc == 'linear'):

			# the cross entropy 
			pp_cross = p_meas * p_sim
			f_meas = np.mean(dim * np.sum(pp_cross, axis = 1) - 1.0) #fidelity function
			
			# the self-entropy of the measured distribution 
			pp_sim = p_sim**2
			f_sim = np.mean(dim * np.sum(pp_sim, axis = 1) - 1.0) # fidelity function

			return float(f_meas/f_sim)

		elif (entropy_calc == 'log'):

			# the cross entropy
			pp_cross = - p_meas * np.log(p_sim)
			H_cross = np.sum(pp_cross, axis = -1) 

			# the self entropy
			H_self = np.log(dim) + np.euler_gamma
			return float(1 - (H_cross - H_self))

# Use the decay of the XEB fidelity as a cost functio. Te parameters of a generic unitary model were optimized to determine a higher-fidelity representation of the unitary. All errors are quoted as Pauli error

def evaluate_rho(rho0, gate_seq_1, gate_seq_2, gate_seq_Cplr = None, generator = 'CZ'):
	"""
	Evaluate the final rho

	Parameters
	----------
	rho0: np.matrix
		initial state
	gate_seq_1: list of class Gate (defined in "gates.py")
		The gate sequence applied to Qubit "1"

	gate_seq_2: list of class Gate (defined in "gates.py")
		The gate sequence applied to Qubit "2"

	gate_seq_Cplr: list of class Gate (defined in "gates.py")
		The gate sequence applied to "Coupler"

	generator: string
		Native 2QB gate

	Returns
	-------
	rho: np.matrix (shape = (4,4))
		The evaulation result.
	"""
	rho0 = Qobj(rho0)
	for i in range(len(gate_seq_1)):
		gate = evaluate_gate([gate_seq_1[i]], [gate_seq_2[i]], [gate_seq_Cplr[i]] if (gate_seq_Cplr is not None) else None, generator = generator)
		print(gate)

	exit()
def evaluate_gate(gate_seq_1, gate_seq_2, gate_seq_Cplr = None, generator = 'CZ'):
	"""
	Evaluate the two qubit gate sequence.

	Parameters
	----------
	gate_seq_1: list of class Gate (defined in "gates.py")
		The gate sequence applied to Qubit "1"

	gate_seq_2: list of class Gate (defined in "gates.py")
		The gate sequence applied to Qubit "2"

	gate_seq_Cplr: list of class Gate (defined in "gates.py")
		The gate sequence applied to "Coupler"

	generator: string
		Native 2QB gate

	Returns
	-------
	twoQ_gate: np.matrix (shape = (4,4))
		The evaulation result.
	"""
	twoQ_gate = np.matrix(
		[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
	for i in range(len(gate_seq_1)):
		gate_1 = np.matrix([[1, 0], [0, 1]])
		gate_2 = np.matrix([[1, 0], [0, 1]])
		if (gate_seq_1[i] == gates.I):
			pass
		elif (gate_seq_1[i] == gates.X2p):
			gate_1 = np.matmul(
				np.matrix([[1, -1j], [-1j, 1]]) / np.sqrt(2), gate_1)
		elif (gate_seq_1[i] == gates.X2m):
			gate_1 = np.matmul(
				np.matrix([[1, 1j], [1j, 1]]) / np.sqrt(2), gate_1)
		elif (gate_seq_1[i] == gates.Y2p):
			gate_1 = np.matmul(
				np.matrix([[1, -1], [1, 1]]) / np.sqrt(2), gate_1)
		elif (gate_seq_1[i] == gates.Y2m):
			gate_1 = np.matmul(
				np.matrix([[1, 1], [-1, 1]]) / np.sqrt(2), gate_1)
		elif (gate_seq_1[i] == gates.W2p):
			gate_1 = np.matmul(
				np.matrix([[1,(1+1j)/np.sqrt(2)],[(-1+1j)/np.sqrt(2),1]]) / np.sqrt(2), gate_1)
		elif (gate_seq_1[i] == gates.Xp):
			gate_1 = np.matmul(np.matrix([[0, -1j], [-1j, 0]]), gate_1)
		elif (gate_seq_1[i] == gates.Xm):
			gate_1 = np.matmul(np.matrix([[0, 1j], [1j, 0]]), gate_1)
		elif (gate_seq_1[i] == gates.Yp):
			gate_1 = np.matmul(np.matrix([[0, -1], [1, 0]]), gate_1)
		elif (gate_seq_1[i] == gates.Ym):
			gate_1 = np.matmul(np.matrix([[0, 1], [-1, 0]]), gate_1)

		if (gate_seq_2[i] == gates.I):
			pass
		elif (gate_seq_2[i] == gates.X2p):
			gate_2 = np.matmul(
				np.matrix([[1, -1j], [-1j, 1]]) / np.sqrt(2), gate_2)
		elif (gate_seq_2[i] == gates.X2m):
			gate_2 = np.matmul(
				np.matrix([[1, 1j], [1j, 1]]) / np.sqrt(2), gate_2)
		elif (gate_seq_2[i] == gates.Y2p):
			gate_2 = np.matmul(
				np.matrix([[1, -1], [1, 1]]) / np.sqrt(2), gate_2)
		elif (gate_seq_2[i] == gates.Y2m):
			gate_2 = np.matmul(
				np.matrix([[1, 1], [-1, 1]]) / np.sqrt(2), gate_2)
		elif (gate_seq_2[i] == gates.W2p):
			gate_2 = np.matmul(
				np.matrix([[1,(1+1j)/np.sqrt(2)],[(-1+1j)/np.sqrt(2),1]]) / np.sqrt(2), gate_2)
		elif (gate_seq_2[i] == gates.Xp):
			gate_2 = np.matmul(np.matrix([[0, -1j], [-1j, 0]]), gate_2)
		elif (gate_seq_2[i] == gates.Xm):
			gate_2 = np.matmul(np.matrix([[0, 1j], [1j, 0]]), gate_2)
		elif (gate_seq_2[i] == gates.Yp):
			gate_2 = np.matmul(np.matrix([[0, -1], [1, 0]]), gate_2)
		elif (gate_seq_2[i] == gates.Ym):
			gate_2 = np.matmul(np.matrix([[0, 1], [-1, 0]]), gate_2)

		gate_12 = np.kron(gate_1, gate_2)
		if generator == 'CZ':
			if (gate_seq_1[i] == gates.CZ or gate_seq_2[i] == gates.CZ):
				gate_12 = np.matmul(
					np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
							   [0, 0, 0, -1]]), gate_12)
		elif generator == 'iSWAP_Cplr':
			if gate_seq_Cplr[i] == gates.iSWAP_Cplr:
				gate_12 = np.matmul(
					np.matrix([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0],
							   [0, 0, 0, 1]]), gate_12)
		elif generator == 'CZ_Cplr':
			if gate_seq_Cplr[i] == gates.CZ_Cplr:
				gate_12 = np.matmul(
					np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
							   [0, 0, 0, -1]]), gate_12)
		twoQ_gate = np.matmul(gate_12, twoQ_gate)

	return twoQ_gate



if __name__ == '__main__':

	# generate XEB circuit
	generator = 'CZ' #native 2QB gate

	num_randomization = 20
	list_cycles = np.arange(1,50,5)
	for i, cycles in enumerate(list_cycles):
		for j in range(num_randomization):
			# generate XEB circuit
			gateSeq1, gateSeq2, gateSeqCplr = generate_XEB_circuit(generator = generator, N_cycles = cycles, rnd_seed = j)

			# print('gateSeq1: {}'.format(gateSeq1) + ', gateSeq2: {}'.format(gateSeq2))
			print('* Generate XEB circuit')
			print('gateSeq1 (Length: {}): {}'.format(len(gateSeq1),gateSeq1))
			print('gateSeq2 (Length: {}): {}'.format(len(gateSeq2),gateSeq2))
			if gateSeqCplr:
				print('gateSeqCplr (Length: {}): {}'.format(len(gateSeqCplr),gateSeqCplr))
			print('\n')

			print('* Evaluate the circuit')
			# gate_ideal = evaluate_gate(gateSeq1, gateSeq2, gateSeqCplr, generator = generator)
			gate_ideal = evaluate_gate(gateSeq1, gateSeq2, gateSeqCplr, generator = generator)
			print('corresponding unitary matrix: {}'.format(gate))
			print('\n')
			exit()
			print('* Estimate the final state')
			psi_init = np.matrix('1; 0; 0; 0') # ground state |00>
			psi_final = np.matmul(gate, psi_init)
			rho_final = psi_final * psi_final.H
			print('rho_final: {}'.format(rho_final))
			for i in range(2):
				for j in range(2):
					print('population of |{}{}>: {}'.format(i, j, np.real(rho_final[i*2+j,i*2+j])))
