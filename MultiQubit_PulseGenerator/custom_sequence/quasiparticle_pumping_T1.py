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
		num_pi = int(config.get('Parameter #1', 1)) # number of pi pulses
		pump_spacing = float(config.get('Parameter #2', 0)) # spacing between pi pulses
		sequence_duration = float(config.get('Parameter #3', 0)) # sequence duration
		for i in range(num_pi):
			self.add_gate_to_all(gates.Xp)
			self.add_gate_to_all(gates.IdentityGate(width = pump_spacing))
		self.add_gate_to_all(gates.Xp)
		self.add_gate_to_all(gates.IdentityGate(width=sequence_duration), dt=0)
if __name__ == '__main__':
	pass
