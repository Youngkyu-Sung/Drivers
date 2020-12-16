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
        # multi_gate_seq = []
        # multi_gate_seq.append(gates.iSWAP_Cplr)

        # self.add_gate(qubit=0, gate=gates.I)
        # self.add_gate(qubit=0, gate=gates.Xp)
        self.add_gate(qubit = [0,1,2], gate = [gates.Yp, gates.I, gates.I])
        self.add_gate(qubit = [0,1,2], gate = [gates.X2p, gates.I, gates.I])
        self.add_gate(qubit= [0,1,2], gate = gates.iSWAP_Cplr_Z_ahead)
        self.add_gate(qubit = [0,1,2], gate = [gates.Yp, gates.I, gates.I])
        self.add_gate(qubit = [0,1,2], gate = [gates.X2p, gates.I, gates.I])
        self.add_gate(qubit= [0,1,2], gate = gates.iSWAP_Cplr_Z_ahead)
        # self.add_gate(qubit = [0,1,2], gate = [gates.Xp, gates.I, gates.X2p])
        self.add_gate(qubit= [0,1,2], gate = gates.iSWAP_Cplr_Z_ahead)
        # pass
if __name__ == '__main__':
    pass