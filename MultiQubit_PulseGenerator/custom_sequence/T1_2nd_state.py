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
		sequence_duration = float(config.get('Parameter #1', 0))
		self.add_gate_to_all(gates.Xp)
		self.add_gate_to_all(gates.Xp_12)
		self.add_gate_to_all(gates.IdentityGate(width = sequence_duration))

if __name__ == '__main__':
	pass
