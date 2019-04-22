#!/usr/bin/env python3
import logging
import random as rnd

import numpy as np
import cliffords
import copy

import gates
from sequence import Sequence

log = logging.getLogger('LabberDriver')
import os
path_currentdir  = os.path.dirname(os.path.realpath(__file__)) # curret directory

class CustomSequence(Sequence):
	def generate_sequence(self, config):
		# config.get('Parameter #1', False)

		# apply pi-pulse for QB 3
		self.add_gate(qubit=0, gate=gates.Xp)
		self.add_gate(qubit=2, gate=gates.I)

		# apply Z-pulse for QB3
		self.add_gate(qubit=2, gate=gates.Zp)


if __name__ == '__main__':
	pass
