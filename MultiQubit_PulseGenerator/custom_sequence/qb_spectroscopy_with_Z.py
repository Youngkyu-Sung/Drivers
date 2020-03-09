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
		# just add pi-pulses for the number of available qubits
		self.add_gate_to_all(gates.IdentityGate(width = 100e-6), t0 = 50e-6)
		self.add_gate_to_all(gates.Xp, t0 = 50e-6)
		self.add_gate_to_all(gates.Zp, t0 = 50e-6)

		# pass
if __name__ == '__main__':
	pass
