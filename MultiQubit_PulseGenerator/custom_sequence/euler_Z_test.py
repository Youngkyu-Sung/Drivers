#!/usr/bin/env python3
# add logger, to allow logging to Labber's instrument log
import logging
import numpy as np
import sys, os
path_currentdir = os.path.dirname(os.path.abspath(__file__)) # curret directory
path_parentdir =  os.path.abspath(os.path.join(path_currentdir, os.pardir)) # parent directory
sys.path.append(path_parentdir) # add path of parent folder

import gates
from sequence import Sequence

log = logging.getLogger('LabberDriver')

class CustomSequence(Sequence):
	"""Sequence for driving Rabi oscillations in multiple qubits."""

	def generate_sequence(self, config):
		"""Generate sequence by adding gates/pulses to waveforms."""
		num_gates = int(config.get('Parameter #1', 1))
		# for i in range(num_gates):
		self.add_gate(qubit = [0,1,2], gate = [gates.Y2p, gates.I, gates.X2m])
		self.add_gate(qubit = [0,1,2], gate = [gates.X2p, gates.I, gates.Y2m])
		self.add_gate(qubit = [0,1,2], gate = [gates.I, gates.I, gates.X2p])
		self.add_gate(qubit = [0,1,2], gate = [gates.EulerZGate(1.8, name='EulerZ'), gates.I, gates.EulerZGate(1.8, name='EulerZ')])

			# # if alternate_polarity == True:
			# # 	if i % 2 == 0:
			# # 		self.add_gate(qubit=[0,1,2], gate = gates.iSWAP_Cplr)
			# # 	else:
			# # 		self.add_gate(qubit=[0,1,2], gate = gates.iSWAP_Cplr_opposite)
			# # else:
			# 	# self.add_gate(qubit=[0,1,2], gate=gates.iSWAP_Cplr)
			# self.add_gate(qubit=[0,1,2], gate=gates.iSWAP_Cplr_Z_behind)
			# # log.info('iSWAP CPLR_Z behind!')
			# self.add_gate_to_all(gates.IdentityGate(width = pulse_spacing))
			# # self.add_gate(0, gates.X2p)

if __name__ == '__main__':
	pass
