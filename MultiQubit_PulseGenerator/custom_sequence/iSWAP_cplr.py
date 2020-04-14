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
		num_iSWAP_pulses = int(config.get('Parameter #1', 1))
		pulse_spacing = float(config.get('Parameter #2', 0))
		alternate_polarity = bool(config.get('Alternate 2QB pulse polarity', False))

		self.add_gate_to_all(gates.IdentityGate(width = 15e-9)) #Give enough spacing between the first preparation pulse and the 2qb pulse
		for i in range(num_iSWAP_pulses):
			if alternate_polarity == True:
				if i % 2 == 0:
					self.add_gate(qubit=[0,1,2], gate = gates.iSWAP_Cplr)
				else:
					self.add_gate(qubit=[0,1,2], gate = gates.iSWAP_Cplr_opposite)
			else:
				self.add_gate(qubit=[0,1,2], gate=gates.iSWAP_Cplr)
			self.add_gate_to_all(gates.IdentityGate(width = pulse_spacing))
			# self.add_gate(0, gates.X2p)

if __name__ == '__main__':
	pass
