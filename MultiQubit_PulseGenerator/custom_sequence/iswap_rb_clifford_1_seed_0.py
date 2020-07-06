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
		if num_gates > 0:
			self.add_gate(qubit = [0,1,2], gate = [gates.Y2m, gates.I, gates.Yp])
		if num_gates > 1:
			self.add_gate(qubit = [0,1,2], gate = [gates.X2p, gates.I, gates.Xp])
		if num_gates > 2:
			self.add_gate(qubit = [0,1,2], gate = gates.iSWAP_Cplr_Z_ahead)
		if num_gates > 3:
			self.add_gate(qubit = [0,1,2], gate = [gates.X2p, gates.I, gates.I])
		if num_gates > 4:
			self.add_gate(qubit = [0,1,2], gate = gates.iSWAP_Cplr_Z_behind)
		if num_gates > 5:
			self.add_gate(qubit = [0,1,2], gate = [gates.Y2p, gates.I, gates.Y2p])
		if num_gates > 6:
			self.add_gate(qubit = [0,1,2], gate = [gates.X2p, gates.I, gates.I])
		if num_gates > 7:
			self.add_gate(qubit = [0,1,2], gate = [gates.X2m, gates.I, gates.Y2p])
		if num_gates > 8:
			self.add_gate(qubit = [0,1,2], gate = [gates.Y2p, gates.I, gates.X2m])
		if num_gates > 9:
			self.add_gate(qubit = [0,1,2], gate = gates.iSWAP_Cplr_Z_ahead)
		if num_gates > 10:
			self.add_gate(qubit = [0,1,2], gate = [gates.X2p, gates.I, gates.I])
		if num_gates > 11:
			self.add_gate(qubit = [0,1,2], gate = gates.iSWAP_Cplr_Z_behind)
		if num_gates > 12:
			self.add_gate(qubit = [0,1,2], gate = [gates.Y2p, gates.I, gates.X2m])
		if num_gates > 13:
			self.add_gate(qubit = [0,1,2], gate = [gates.X2p, gates.I, gates.Y2m])
		if num_gates > 14:
			self.add_gate(qubit = [0,1,2], gate = [gates.I, gates.I, gates.X2p])

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
