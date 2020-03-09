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

        num_CZ_pulses = int(config.get('Parameter #1', 1))
        delay_between_iSWAP_CZ = float(config.get('Parameter #2', 0))

        for i in range(num_CZ_pulses):
        	self.add_gate(qubit=[0,1,2], gate=gates.CZ_Cplr)
        	self.add_gate(qubit=0, gate=gates.IdentityGate(delay_between_iSWAP_CZ)) # some delay
        	self.add_gate(qubit=[0,1,2], gate=gates.iSWAP_Cplr)

        # pass
if __name__ == '__main__':
    pass