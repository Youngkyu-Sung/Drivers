#!/usr/bin/env python3
import logging
import random as rnd

import numpy as np
import cliffords
import copy

import itertools

import gates
from sequence import Sequence

log = logging.getLogger('LabberDriver')
import os
path_currentdir  = os.path.dirname(os.path.realpath(__file__)) # curret directory

def CheckIdentity(matrix):
    """
    Check whether the matrix is identity by calculating it numerically.

    Parameters
    ----------
    matrix: 4x4 np.matrix
        matrix

    Returns
    -------
    result: bool
        True, if identity.
    """

    #Check all the diagonal entries.
    if ((np.abs(np.abs(matrix[0,0])-1)< 1e-6) and
        (np.abs(np.abs(matrix[1,1])-1)< 1e-6) and
        (np.abs(np.abs(matrix[2,2])-1)< 1e-6) and
        (np.abs(np.abs(matrix[3,3])-1)< 1e-6) and
        (np.abs(matrix[1,1]/matrix[0,0]-1)<1e-6) and
        (np.abs(matrix[2,2]/matrix[0,0]-1)<1e-6) and
        (np.abs(matrix[3,3]/matrix[0,0]-1)<1e-6)):
        return True
    else:
        return False

def add_singleQ_forXEB(index, gate_seq):
    """
        Add single qubit gate for cross-entropy benchmarking (3).
        the single-qubit gates are chosen from X/2 (pi/2 rotation around the X axis), Y/2 (pi/2 rotation around the Y axis) and (X + Y)/2 (pi/2 rotation around an axis pi/4 away from the X on the equator of the Bloch sphere).
    """

    length_before = len(gate_seq)
    # Paulis
    if index == 0:
        gate_seq.append(gates.X2p)
    elif index == 1:
        gate_seq.append(gates.Y2p)
    elif index == 2:
        gate_seq.append(gates.W2p)
    else:
        raise ValueError(
            'index is out of range. it should be smaller than 2 and greater'
            ' or equal to 0: ', str(index))

def add_singleQ_clifford(index, gate_seq, pad_with_I=True):
    """Add single qubit clifford (24)."""
    length_before = len(gate_seq)
    # Paulis
    if index == 0:
        gate_seq.append(gates.I)
    elif index == 1:
        gate_seq.append(gates.Xp)
    elif index == 2:
        gate_seq.append(gates.Yp)
    elif index == 3:
        gate_seq.append(gates.Yp)
        gate_seq.append(gates.Xp)

    # 2pi/3 rotations
    elif index == 4:
        gate_seq.append(gates.X2p)
        gate_seq.append(gates.Y2p)
    elif index == 5:
        gate_seq.append(gates.X2p)
        gate_seq.append(gates.Y2m)
    elif index == 6:
        gate_seq.append(gates.X2m)
        gate_seq.append(gates.Y2p)
    elif index == 7:
        gate_seq.append(gates.X2m)
        gate_seq.append(gates.Y2m)
    elif index == 8:
        gate_seq.append(gates.Y2p)
        gate_seq.append(gates.X2p)
    elif index == 9:
        gate_seq.append(gates.Y2p)
        gate_seq.append(gates.X2m)
    elif index == 10:
        gate_seq.append(gates.Y2m)
        gate_seq.append(gates.X2p)
    elif index == 11:
        gate_seq.append(gates.Y2m)
        gate_seq.append(gates.X2m)

    # pi/2 rotations
    elif index == 12:
        gate_seq.append(gates.X2p)
    elif index == 13:
        gate_seq.append(gates.X2m)
    elif index == 14:
        gate_seq.append(gates.Y2p)
    elif index == 15:
        gate_seq.append(gates.Y2m)
    elif index == 16:
        gate_seq.append(gates.X2m)
        gate_seq.append(gates.Y2p)
        gate_seq.append(gates.X2p)
    elif index == 17:
        gate_seq.append(gates.X2m)
        gate_seq.append(gates.Y2m)
        gate_seq.append(gates.X2p)

    # Hadamard-Like
    elif index == 18:
        gate_seq.append(gates.Xp)
        gate_seq.append(gates.Y2p)
    elif index == 19:
        gate_seq.append(gates.Xp)
        gate_seq.append(gates.Y2m)
    elif index == 20:
        gate_seq.append(gates.Yp)
        gate_seq.append(gates.X2p)
    elif index == 21:
        gate_seq.append(gates.Yp)
        gate_seq.append(gates.X2m)
    elif index == 22:
        gate_seq.append(gates.X2p)
        gate_seq.append(gates.Y2p)
        gate_seq.append(gates.X2p)
    elif index == 23:
        gate_seq.append(gates.X2m)
        gate_seq.append(gates.Y2p)
        gate_seq.append(gates.X2m)
    else:
        raise ValueError(
            'index is out of range. it should be smaller than 24 and greater'
            ' or equal to 0: ', str(index))

    length_after = len(gate_seq)
    if pad_with_I:
        # Force the clifford to have a length of 3 gates
        for i in range(3-(length_after-length_before)):
            gate_seq.append(gates.I)

def add_twoQ_clifford(index, gate_seq_1, gate_seq_2, gate_seq_Cplr = [], generator = 'CZ'):
    """Add single qubit clifford (11520 = 576 + 5184 + 5184 + 576)."""
    # log.info('index num: %d'%(index))
    if (index < 0):
        raise ValueError(
            'index is out of range. it should be smaller than 11520 and '
            'greater or equal to 0: ', str(index))
    elif (index < 576):
        add_singleQ_based_twoQ_clifford(index, gate_seq_1, gate_seq_2, gate_seq_Cplr = gate_seq_Cplr, generator = generator)
    elif (index < 5184 + 576):
        add_CNOT_like_twoQ_clifford(index, gate_seq_1, gate_seq_2, gate_seq_Cplr = gate_seq_Cplr, generator = generator)
    elif (index < 5184 + 5184 + 576):
        add_iSWAP_like_twoQ_clifford(index, gate_seq_1, gate_seq_2, gate_seq_Cplr = gate_seq_Cplr, generator = generator)
    elif (index < 576 + 5184 + 5184 + 576):
        add_SWAP_like_twoQ_clifford(index, gate_seq_1, gate_seq_2, gate_seq_Cplr = gate_seq_Cplr, generator = generator)
    else:
        raise ValueError(
            'index is out of range. it should be smaller than 11520 and '
            'greater or equal to 0: ', str(index))
    pass

def add_singleQ_S1(index, gate_seq):
    """Add single qubit clifford from S1.

    (I-like-subset of single qubit clifford group) (3)
    """
    if index == 0:
        gate_seq.append(gates.I)
        gate_seq.append(gates.I)  # auxiliary
        gate_seq.append(gates.I)  # auxiliary
    elif index == 1:
        gate_seq.append(gates.Y2p)
        gate_seq.append(gates.X2p)
        gate_seq.append(gates.I)  # auxiliary
    elif index == 2:
        gate_seq.append(gates.X2m)
        gate_seq.append(gates.Y2m)
        gate_seq.append(gates.I)  # auxiliary

def add_singleQ_S1_X2p(index, gate_seq):
    """Add single qubit clifford from S1_X2p.

    (X2p-like-subset of single qubit clifford group) (3)
    """
    if index == 0:
        gate_seq.append(gates.X2p)
        gate_seq.append(gates.I)  # auxiliary
        gate_seq.append(gates.I)  # auxiliary
    elif index == 1:
        gate_seq.append(gates.X2p)
        gate_seq.append(gates.Y2p)
        gate_seq.append(gates.X2p)
    elif index == 2:
        gate_seq.append(gates.Y2m)
        gate_seq.append(gates.I)  # auxiliary
        gate_seq.append(gates.I)  # auxiliary

def add_singleQ_S1_Y2p(index, gate_seq):
    """Add single qubit clifford from S1_Y2p.

    (Y2p-like-subset of single qubit clifford group) (3)
    """
    if index == 0:
        gate_seq.append(gates.Y2p)
        gate_seq.append(gates.I)  # auxiliary
        gate_seq.append(gates.I)  # auxiliary
    elif index == 1:
        gate_seq.append(gates.Yp)
        gate_seq.append(gates.X2p)
        gate_seq.append(gates.I)  # auxiliary
    elif index == 2:
        gate_seq.append(gates.X2m)
        gate_seq.append(gates.Y2m)
        gate_seq.append(gates.X2p)

def add_singleQ_S1_Z2p(index, gate_seq):
    """Add single qubit clifford from S1_Z2p.

    (Z2p-like-subset of single qubit clifford group) (3)
    """
    if index == 0:
        gate_seq.append(gates.X2m)
        gate_seq.append(gates.Y2m)
        gate_seq.append(gates.X2p)
    elif index == 1:
        gate_seq.append(gates.Y2p)
        gate_seq.append(gates.I)  # auxiliary
        gate_seq.append(gates.I)  # auxiliary
    elif index == 2:
        gate_seq.append(gates.X2m)
        gate_seq.append(gates.Ym)
        gate_seq.append(gates.I)  # auxiliary

def add_singleQ_based_twoQ_clifford(index, gate_seq_1, gate_seq_2, gate_seq_Cplr = [], generator = 'CZ'):
    """Add single-qubit-gates-only-based two Qubit Clifford.

    (24*24 = 576)
    (gate_seq_1: gate seq. of qubit #1, gate_seq_t: gate seq. of qubit #2)
    """

    # randomly sample from single qubit cliffords (24)
    index_1 = index % 24

    # randomly sample from single qubit cliffords (24)
    index_2 = (index // 24) % 24

    add_singleQ_clifford(index_1, gate_seq_1)
    add_singleQ_clifford(index_2, gate_seq_2)
    if generator in ['iSWAP_Cplr', 'CZ_Cplr']:
        add_singleQ_S1(0, gate_seq_Cplr) # add redundant I gates for a coupler

def add_CNOT_like_twoQ_clifford(index, gate_seq_1, gate_seq_2, gate_seq_Cplr = [], generator = 'CZ'):
    """Add CNOT like two Qubit Clifford.

    (24*24*3*3 = 5184)
    (gate_seq_1: gate seq. of qubit #1, gate_seq_t: gate seq. of qubit #2)
    """
    # randomly sample from single qubit cliffords (24)
    index_1 = index % 24

    # randomly sample from single qubit cliffords (24)
    index_2 = (index // 24) % 24

    # randomly sample from S1 (3)
    index_3 = (index // 24 // 24) % 3

    # randomly sample from S1_Y2p (3) or S1_Z2p (3)
    index_4 = (index // 24 // 24 // 3) % 3

    if generator == 'CZ':
        add_singleQ_clifford(index_1, gate_seq_1)
        add_singleQ_clifford(index_2, gate_seq_2)
        gate_seq_1.append(gates.I)
        gate_seq_2.append(gates.CZ)
        add_singleQ_S1(index_3, gate_seq_1)
        add_singleQ_S1_Y2p(index_4, gate_seq_2)

    elif generator == 'CZ_Cplr':
        add_singleQ_clifford(index_1, gate_seq_1)
        add_singleQ_S1(0, gate_seq_Cplr) # add redundant I gates for a coupler
        add_singleQ_clifford(index_2, gate_seq_2)

        gate_seq_1.append(gates.I)
        gate_seq_Cplr.append(gates.CZ_Cplr)
        gate_seq_2.append(gates.I)

        add_singleQ_S1(index_3, gate_seq_1)
        add_singleQ_S1(0, gate_seq_Cplr) # add redundant I gates for a coupler
        add_singleQ_S1_Y2p(index_4, gate_seq_2)

    elif generator == 'iSWAP_Cplr':
        add_singleQ_clifford(index_1, gate_seq_1)
        add_singleQ_S1(0, gate_seq_Cplr) # add redundant I gates for a coupler
        add_singleQ_clifford(index_2, gate_seq_2)

        gate_seq_1.append(gates.I)
        # gate_seq_Cplr.append(gates.iSWAP_Cplr)
        gate_seq_Cplr.append(gates.iSWAP_Cplr_Z_ahead)
        gate_seq_2.append(gates.I)

        gate_seq_1.append(gates.X2p)
        gate_seq_Cplr.append(gates.I)
        gate_seq_2.append(gates.I)

        gate_seq_1.append(gates.I)
        # gate_seq_Cplr.append(gates.iSWAP_Cplr)
        gate_seq_Cplr.append(gates.iSWAP_Cplr_Z_behind)
        gate_seq_2.append(gates.I)

        add_singleQ_S1(index_3, gate_seq_1)
        add_singleQ_S1(0, gate_seq_Cplr) # add redundant I gates for a coupler
        add_singleQ_S1_Z2p(index_4, gate_seq_2)
        
def add_iSWAP_like_twoQ_clifford(index, gate_seq_1, gate_seq_2, gate_seq_Cplr = [], generator = 'CZ'):
    """Add iSWAP like two Qubit Clifford.

    (24*24*3*3 = 5184)
    (gate_seq_1: gate seq. of qubit #1, gate_seq_t: gate seq. of qubit #2)
    """

    # randomly sample from single qubit cliffords (24)
    index_1 = index % 24

    # randomly sample from single qubit cliffords (24)
    index_2 = (index // 24) % 24

    # randomly sample from S1_Y2p (3) or S1 (3)
    index_3 = (index // 24 // 24) % 3

    # randomly sample from S1_X2p (3) or S1 (3)
    index_4 = (index // 24 // 24 // 3) % 3

    if generator == 'CZ':
        add_singleQ_clifford(index_1, gate_seq_1)
        add_singleQ_clifford(index_2, gate_seq_2)
        gate_seq_1.append(gates.I)
        gate_seq_2.append(gates.CZ)
        gate_seq_1.append(gates.Y2p)
        gate_seq_2.append(gates.X2m)
        gate_seq_1.append(gates.I)
        gate_seq_2.append(gates.CZ)
        add_singleQ_S1_Y2p(index_3, gate_seq_1)
        add_singleQ_S1_X2p(index_4, gate_seq_2)

    elif generator == 'CZ_Cplr':
        add_singleQ_clifford(index_1, gate_seq_1)
        add_singleQ_S1(0, gate_seq_Cplr) # add redundant I gates for a coupler
        add_singleQ_clifford(index_2, gate_seq_2)

        gate_seq_1.append(gates.I)
        gate_seq_Cplr.append(gates.CZ_Cplr)
        gate_seq_2.append(gates.I)

        gate_seq_1.append(gates.Y2p)
        gate_seq_Cplr.append(gates.I) # add redundant I gates for a coupler
        gate_seq_2.append(gates.X2m)
            
        gate_seq_1.append(gates.I)
        gate_seq_Cplr.append(gates.CZ_Cplr)
        gate_seq_2.append(gates.I)

        add_singleQ_S1_Y2p(index_3, gate_seq_1)
        add_singleQ_S1(0, gate_seq_Cplr) # add redundant I gates for a coupler
        add_singleQ_S1_X2p(index_4, gate_seq_2)

    elif generator == 'iSWAP_Cplr':
        add_singleQ_clifford(index_1, gate_seq_1)
        add_singleQ_S1(0, gate_seq_Cplr) # add redundant I gates for a coupler
        add_singleQ_clifford(index_2, gate_seq_2)

        gate_seq_1.append(gates.I)
        # gate_seq_Cplr.append(gates.iSWAP_Cplr)
        gate_seq_Cplr.append(gates.iSWAP_Cplr_Z_ahead)
        gate_seq_2.append(gates.I)

        add_singleQ_S1(index_3, gate_seq_1)
        add_singleQ_S1(0, gate_seq_Cplr) # add redundant I gates for a coupler
        add_singleQ_S1(index_4, gate_seq_2)

def add_SWAP_like_twoQ_clifford(index, gate_seq_1, gate_seq_2, gate_seq_Cplr = [], generator = 'CZ'):
    """Add SWAP like two Qubit Clifford.

    (24*24*= 576)
    (gate_seq_1: gate seq. of qubit #1, gate_seq_t: gate seq. of qubit #2)
    """
    # randomly sample from single qubit cliffords (24)
    index_1 = index % 24

    # randomly sample from single qubit cliffords (24)
    index_2 = (index // 24) % 24

    if generator == 'CZ':
        add_singleQ_clifford(index_1, gate_seq_1)
        add_singleQ_clifford(index_2, gate_seq_2)

        gate_seq_1.append(gates.I)
        gate_seq_2.append(gates.CZ)

        gate_seq_1.append(gates.Y2m)
        gate_seq_2.append(gates.Y2p)

        gate_seq_1.append(gates.I)
        gate_seq_2.append(gates.CZ)

        gate_seq_1.append(gates.Y2p)
        gate_seq_2.append(gates.Y2m)

        gate_seq_1.append(gates.I)
        gate_seq_2.append(gates.CZ)

        gate_seq_1.append(gates.I)
        gate_seq_2.append(gates.Y2p)

    elif generator == 'CZ_Cplr':
        add_singleQ_clifford(index_1, gate_seq_1)
        add_singleQ_S1(0, gate_seq_Cplr) # add redundant I gates for a coupler
        add_singleQ_clifford(index_2, gate_seq_2)

        gate_seq_1.append(gates.I)
        gate_seq_Cplr.append(gates.CZ_Cplr)
        gate_seq_2.append(gates.I)

        gate_seq_1.append(gates.Y2m)
        gate_seq_Cplr.append(gates.I)
        gate_seq_2.append(gates.Y2p)

        gate_seq_1.append(gates.I)
        gate_seq_Cplr.append(gates.CZ_Cplr)
        gate_seq_2.append(gates.I)

        gate_seq_1.append(gates.Y2p)
        gate_seq_Cplr.append(gates.I)
        gate_seq_2.append(gates.Y2m)

        gate_seq_1.append(gates.I)
        gate_seq_Cplr.append(gates.CZ_Cplr)
        gate_seq_2.append(gates.I)

        gate_seq_1.append(gates.I)
        gate_seq_Cplr.append(gates.I)
        gate_seq_2.append(gates.Y2p)


    elif generator == 'iSWAP_Cplr':
        add_singleQ_clifford(index_1, gate_seq_1)
        add_singleQ_S1(0, gate_seq_Cplr) # add redundant I gates for a coupler
        add_singleQ_clifford(index_2, gate_seq_2)

        gate_seq_1.append(gates.I)
        # gate_seq_Cplr.append(gates.iSWAP_Cplr)
        gate_seq_Cplr.append(gates.iSWAP_Cplr_Z_ahead)
        gate_seq_2.append(gates.I)

        gate_seq_1.append(gates.I)
        gate_seq_Cplr.append(gates.I)
        gate_seq_2.append(gates.X2m)

        gate_seq_1.append(gates.I)
        gate_seq_Cplr.append(gates.iSWAP_Cplr_Z_behind)
        gate_seq_2.append(gates.I)
        
        gate_seq_1.append(gates.X2m)
        gate_seq_Cplr.append(gates.I)
        gate_seq_2.append(gates.I)

        gate_seq_1.append(gates.I)
        # gate_seq_Cplr.append(gates.iSWAP_Cplr)
        gate_seq_Cplr.append(gates.iSWAP_Cplr_Z_ahead)
        gate_seq_2.append(gates.I)

        gate_seq_1.append(gates.I)
        add_singleQ_S1(0, gate_seq_Cplr) # add redundant I gates for a coupler
        gate_seq_2.append(gates.X2m)

class SingleQubit_RB(Sequence):
    """Single qubit randomized benchmarking."""

    prev_rnd_seed = np.inf  # store the previous value
    prev_N_cliffords = np.inf  # store the previous value
    prev_interleave = np.inf  # store the previous value
    prev_interleaved_gate = np.inf  # store the previous value
    prev_sequence = ''
    prev_gate_seq = []

    def generate_sequence(self, config):
        """Generate sequence by adding gates/pulses to waveforms."""
        # get parameters
        sequence = config['Sequence']
        # Number of Cliffords to generate
        N_cliffords = int(config['Number of Cliffords'])
        rnd_seed = config['Random Seed']
        interleave = config['Interleave 1-QB Gate']
        multi_seq = config.get('Output multiple sequences', False)
        write_seq = config.get('Write sequence as txt file', False)

        # log.info('Assign seed %d' %(rnd_seed))
        rnd.seed(rnd_seed + N_cliffords*100)
        if interleave is True:
            interleaved_gate = config['Interleaved 1-QB Gate']
        else:
            interleaved_gate = np.inf
        # generate new randomized clifford gates only if configuration changes
        if (self.prev_sequence != sequence or
                self.prev_rnd_seed != rnd_seed or
                self.prev_N_cliffords != N_cliffords or
                self.prev_interleave != interleave or
                multi_seq or
                self.prev_interleaved_gate != interleaved_gate or
                self.prev_n_qubit != self.n_qubit):

            self.prev_rnd_seed = rnd_seed
            self.prev_N_cliffords = N_cliffords
            self.prev_interleave = interleave
            self.prev_sequence = sequence
            self.prev_n_qubit = self.n_qubit

            multi_gate_seq = []
            for n in range(self.n_qubit):
                # Generate 1QB RB sequence
                single_gate_seq = []

                for i in range(N_cliffords):
                    rndnum = rnd.randint(0, 23)
                    # log.info('Random number %d' %(rndnum))
                    add_singleQ_clifford(rndnum, single_gate_seq,
                                         pad_with_I=False)
                    # If interleave gate,
                    if interleave is True:
                        self.prev_interleaved_gate = interleaved_gate

                        # To step over "Reference Randomized Benchmarking" 05/15/2019
                        if interleaved_gate == 'Ref':
                            pass
                        else:
                            single_gate_seq.append(getattr(gates, interleaved_gate))

                recovery_gate = self.get_recovery_gate(single_gate_seq)

                # print 1QB-RB sequence
                if write_seq == True:
                    import os
                    from datetime import datetime
                    directory = os.path.join(path_currentdir,'1QB_RBseq')
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    filename = datetime.now().strftime('%Y-%m-%d %H-%M-%S-f')[:-3] + '_qbNum=%d_N_cliffords=%d_seed=%d.txt'%(n, N_cliffords,rnd_seed)
                    filepath = os.path.join(directory,filename)
                    # log.info('make file: ' + filepath)
                    with open(filepath, "w") as text_file:
                        print('New Sequence', file=text_file)
                        for i in range(len(single_gate_seq)):
                             print("CliffordIndex: %d, Gate: ["%(i) + cliffords.Gate_to_strGate(single_gate_seq[i]) +"]", file=text_file)
                        print("Corresponding recovery sequence:")
                        print("Recovery Gate: [" + cliffords.Gate_to_strGate(recovery_gate) +"]", file=text_file)
                single_gate_seq.append(recovery_gate)
                multi_gate_seq.append(single_gate_seq)
            # transpose list of lists
            # - (06/23/2019 Update) Fill identity gates to the shorter sequence at the end -> at the beginning
            multi_gate_seq_reversed = [i[::-1] for i in multi_gate_seq]
            multi_gate_seq_reversed_tr = list(map(list, itertools.zip_longest(*multi_gate_seq_reversed, fillvalue=gates.I))) # Not to chop
            multi_gate_seq = multi_gate_seq_reversed_tr[::-1]

            self.add_gates(multi_gate_seq)
            self.prev_gate_seq = multi_gate_seq
        else:
            self.add_gates(self.prev_gate_seq)

    def evaluate_sequence(self, gate_seq):
        """
        Evaluate a single qubit gate sequence.
        (Reference: http://www.vcpc.univie.ac.at/~ian/hotlist/qc/talks/bloch-sphere-rotations.pdf)

        Parameters
        ----------
        gate_seq_1: list of class Gate (defined in "gates.py")
            The gate sequence applied to a qubit

        Returns
        -------
        singleQ_gate: np.matrix
            The evaulation result.
        """

        singleQ_gate = np.matrix([[1, 0], [0, 1]])
        for i in range(len(gate_seq)):
            if (gate_seq[i] == gates.I):
                pass
            elif (gate_seq[i] == gates.X2p):
                singleQ_gate = np.matmul(
                    np.matrix([[1, -1j], [-1j, 1]]) / np.sqrt(2), singleQ_gate)
            elif (gate_seq[i] == gates.X2m):
                singleQ_gate = np.matmul(
                    np.matrix([[1, 1j], [1j, 1]]) / np.sqrt(2), singleQ_gate)
            elif (gate_seq[i] == gates.Y2p):
                singleQ_gate = np.matmul(
                    np.matrix([[1, -1], [1, 1]]) / np.sqrt(2), singleQ_gate)
            elif (gate_seq[i] == gates.Y2m):
                singleQ_gate = np.matmul(
                    np.matrix([[1, 1], [-1, 1]]) / np.sqrt(2), singleQ_gate)
            elif (gate_seq[i] == gates.Xp):
                singleQ_gate = np.matmul(
                    np.matrix([[0, -1j], [-1j, 0]]), singleQ_gate)
            elif (gate_seq[i] == gates.Xm):
                singleQ_gate = np.matmul(
                    np.matrix([[0, 1j], [1j, 0]]), singleQ_gate)
            elif (gate_seq[i] == gates.Yp):
                singleQ_gate = np.matmul(
                    np.matrix([[0, -1], [1, 0]]), singleQ_gate)
            elif (gate_seq[i] == gates.Ym):
                singleQ_gate = np.matmul(
                    np.matrix([[0, 1], [-1, 0]]), singleQ_gate)
            elif (gate_seq[i] in (gates.Zp, gates.VZp)):
                singleQ_gate = np.matmul(
                    np.matrix([[-1j, 0], [0, 1j]]), singleQ_gate)
        return singleQ_gate

    def get_recovery_gate(self, gate_seq):
        """
        Get the recovery (the inverse) gate

        Parameters
        ----------
        gate_seq: list of class Gate
            The gate sequence applied to a qubit

        Returns
        -------
        recovery_gate: Gate
            The recovery gate
        """

        qubit_state = np.matrix('1; 0')
        # initial state: ground state, following the QC community's convention
        qubit_state = np.matmul(self.evaluate_sequence(gate_seq), qubit_state)
        # find recovery gate which makes qubit_state return to initial state
        if (np.abs(np.linalg.norm(qubit_state.item((0, 0))) - 1) < 0.1):
            # ground state -> I
            recovery_gate = gates.I
        elif (np.abs(np.linalg.norm(qubit_state.item((1, 0))) - 1) < 0.1):
            # excited state -> X Pi
            recovery_gate = gates.Xp
        elif (np.linalg.norm(qubit_state.item((1, 0)) /
                             qubit_state.item((0, 0)) + 1) < 0.1):
            # X State  -> Y +Pi/2
            recovery_gate = gates.Y2p
        elif (np.linalg.norm(qubit_state.item((1, 0)) /
                             qubit_state.item((0, 0)) - 1) < 0.1):
            # -X State -> Y -Pi/2
            recovery_gate = gates.Y2m
        elif (np.linalg.norm(qubit_state.item((1, 0)) /
                             qubit_state.item((0, 0)) + 1j) < 0.1):
            # Y State -> X -Pi/2
            recovery_gate = gates.X2m
        elif (np.linalg.norm(qubit_state.item((1, 0)) /
                             qubit_state.item((0, 0)) - 1j) < 0.1):
            # -Y State -> X +Pi/2
            recovery_gate = gates.X2p
        else:
            raise InstrumentDriver.Error(
                'Error in calculating recovery gates. qubit state:' +
                str(qubit_state))
        return recovery_gate

# # from scipy.spatial.transform import Rotation as R # requires scipy 1.5.0, requirements: conda update -n base conda (update conda), conda install -c anaconda scipy (install the latest version)
# # from sequence import *
# import math

# def rotation_matrix(axis, theta):
#     """
#     Return the rotation matrix associated with counterclockwise rotation about
#     the given axis by theta radians.
#     """
#     axis = np.asarray(axis)
#     axis = axis / math.sqrt(np.dot(axis, axis))
#     a = math.cos(theta / 2.0)
#     b, c, d = -axis * math.sin(theta / 2.0)
#     aa, bb, cc, dd = a * a, b * b, c * c, d * d
#     bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
#     return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
#                      [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
#                      [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

# def mat_to_euler_angles_xyx(matrix):
#     # find euler angles (x-y-x) give 3d rotation matrix
#     threshold = 1e-8
#     theta = np.arccos(np.clip(matrix[0,0],-1,1))
#     if np.abs(np.sin(theta)) < threshold: 
#         # is theta np.pi or zero?
#         if np.abs(theta) < threshold:
#             # if theta = 0, one X gate is enough
#             phi = 0
#             # with np.errstate(divide='ignore'):  # ignore runtimewarning (divide by zero)
#             #     psi = np.arctan(matrix[2,1]/matrix[1,1])
#             psi = np.arctan2(matrix[2,1],matrix[1,1])

#             # if np.abs(np.sin(psi)) < threshold:
#             #     # distinguish psi np.pi vs zero.
#             #     psi = np.arccos(np.clip(matrix[1,1],-1,1))
#         else:
#             # if theta = pi
#             phi = 0
#             psi = np.arccos(matrix[1,1])

#     elif np.abs(np.cos(theta)) < threshold:
#         # is theta +np.pi/2 or -np.pi/2?
#         # calcualte phi, psi
#         phi = np.arctan2(matrix[0,1], matrix[0,2])
#         psi = np.arctan2(matrix[0,1], -matrix[0,2])

#         if ((np.abs(matrix[1,0])>threshold) and (np.abs(np.sin(psi)) > 0)):
#             theta = np.arcsin(np.clip(matrix[1,0]/np.sin(psi),-1,1))
#         elif ((np.abs(matrix[2,0])>threshold) and (np.abs(np.cos(psi)) > 0)):
#             theta = np.arcsin(-np.clip(matrix[2,0]/np.cos(psi),-1,1))
#         else:
#             raise ValueError('Singular Matrix!')
#     else:
#         #is theta positive / negative?

#         phi = np.arctan2(matrix[0,1], matrix[0,2])
#         psi = np.arctan2(matrix[0,1], -matrix[0,2])
#         # with np.errstate(divide='ignore'):  # ignore runtimewarning (divide by zero)
#         #     phi = np.arctan(matrix[0,1]/matrix[0,2])
#         #     psi = np.arctan(-matrix[1,0]/matrix[2,0])

#         sign_theta = None
#         if np.abs(np.cos(phi)) > threshold:
#             sign_theta = np.sign(matrix[0,2]/np.cos(phi))
#         # elif np.abs(np.sin(phi))> threshold:
#         #     sign_theta = np.sign(matrix[0,1]/np.sin(phi))
#         # print('sign from phi: {}'.format(sign_theta))
#         if np.abs(np.cos(psi)) > threshold:
#             sign_theta = np.sign(-matrix[2,0]/np.cos(psi))

#         if sign_theta is None:
#             pass
#         elif sign_theta > 0:
#             theta = np.abs(theta)
#         else:
#             theta = -np.abs(theta)

#         # if (np.abs(matrix[0,2]) < threshold):
#         #     phi = np.arcsin(np.clip(matrix[0,1]/np.sin(theta),-1,1))

#         # if (np.abs(matrix[2,0]) < threshold):
#         #     psi = np.arcsin(np.clip(matrix[1,0]/np.sin(theta),-1,1))

#     # print('phi(X): {}, theta(Y): {}, psi(X): {}'.format(phi/np.pi*180, theta/np.pi*180, psi/np.pi*180))
#     return (phi, theta, psi)

# def mat_to_euler_angles_yxy(matrix):
#     # find euler angles (y-x-y) give 3d rotation matrix
#     theta = np.arccos(np.clip(matrix[1,1],-1,1))
#     if theta == 0: # only X gate is enough
#         phi = np.arccos(np.clip(matrix[1,1],-1,1))
#         psi = 0
#     else:
#         with np.errstate(divide='ignore'): # ignore runtimewarning (divide by zero)
#             phi = np.arctan(-matrix[1,0]/matrix[1,2])
#             psi = np.arctan(matrix[0,1]/matrix[2,1])
#     # print('phi(Y): {}, theta(X): {}, psi(Y): {}'.format(phi/np.pi*180, theta/np.pi*180, psi/np.pi*180))
#     return (phi, theta, psi)

# def gate_to_euler_gates(list_1qb_gates, convention = 'x-y-x'):
#     threshold = 1e-10
#     phi, theta, psi = gate_to_euler_angles(list_1qb_gates, convention)

#     if convention == 'x-y-x':
#         # first X-gate
#         if np.abs(phi) > threshold:
#             X1 = gates.SingleQubitXYRotation(phi=0, theta=phi, name='X={:+.2f}'.format(phi))
#         else:
#             X1 = gates.IdentityGate(width=None)

#         # second Y-gate
#         if np.abs(theta) > threshold:
#             Y2 = gates.SingleQubitXYRotation(phi=np.pi*0.5, theta=theta, name='Y={:+.2f}'.format(theta))
#         else:
#             Y2 = gates.IdentityGate(width=None)

#         # third X-gate
#         if np.abs(psi) > threshold:
#             X3 = gates.SingleQubitXYRotation(phi=0, theta=psi, name='X={:+.2f}'.format(psi))
#         else:
#             X3 = gates.IdentityGate(width=None)
#         return [X1, Y2, X3]

#     elif convention == 'y-x-y':
#         # first Y-gate
#         if np.abs(phi) > threshold:
#             Y1 = gates.SingleQubitXYRotation(phi=np.pi*0.5, theta=phi, name='Y={:+.2f}'.format(phi))
#         else:
#             Y1 = gates.IdentityGate(width=None)

#         # second X-gate
#         if np.abs(theta) > threshold:
#             X2 = gates.SingleQubitXYRotation(phi=0, theta=theta, name='X={:+.2f}'.format(theta))
#         else:
#             X2 = gates.IdentityGate(width=None)

#         # third Y-gate
#         if np.abs(psi) > threshold:
#             Y3 = gates.SingleQubitXYRotation(phi=np.pi*0.5, theta=psi, name='Y={:+.2f}'.format(psi))
#         else:
#             Y3 = gates.IdentityGate(width=None)
#         return [Y1, X2, Y3]

# def gate_to_euler_angles(list_1qb_gates, convention = 'x-y-x'):
#     # represent single qubit gate in a three-dimensional (3x3) rotation matrix in (x,y,z) basis

#     mat_3d = np.identity(3)
#     for gate in list_1qb_gates:
#         if (gate == gates.I):
#             mat_3d = np.matmul(np.identity(3), mat_3d) 
#         elif (gate == gates.X2p):
#             mat_3d = np.matmul(rotation_matrix([1,0,0], np.pi*0.5), mat_3d)
#         elif (gate == gates.X2m):
#             mat_3d = np.matmul(rotation_matrix([1,0,0], -np.pi*0.5), mat_3d)
#         elif (gate == gates.Y2p):
#             mat_3d = np.matmul(rotation_matrix([0,1,0], np.pi*0.5), mat_3d)
#         elif (gate == gates.Y2m):
#             mat_3d = np.matmul(rotation_matrix([0,1,0], -np.pi*0.5), mat_3d)
#         elif (gate == gates.Z2p):
#             mat_3d = np.matmul(rotation_matrix([0,0,1], +np.pi*0.5), mat_3d)
#         elif (gate == gates.Z2m):
#             mat_3d = np.matmul(rotation_matrix([0,0,1], -np.pi*0.5), mat_3d)
#         elif (gate == gates.Xp):
#             mat_3d = np.matmul(rotation_matrix([1,0,0], np.pi), mat_3d)
#         elif (gate == gates.Xm):
#             mat_3d = np.matmul(rotation_matrix([1,0,0], -np.pi), mat_3d)
#         elif (gate == gates.Yp):
#             mat_3d = np.matmul(rotation_matrix([0,1,0], np.pi), mat_3d)
#         elif (gate == gates.Ym):
#             mat_3d = np.matmul(rotation_matrix([0,1,0], -np.pi), mat_3d)
#         elif (gate == gates.Zp):
#             mat_3d = np.matmul(rotation_matrix([0,0,1], np.pi), mat_3d)
#         elif (gate == gates.Zm):
#             mat_3d = np.matmul(rotation_matrix([0,0,1], -np.pi), mat_3d)
#         elif isinstance(gate, gates.EulerZGate):
#             mat_3d = np.matmul(rotation_matrix([0,0,1], gate.theta), mat_3d)
#         else:
#             raise ValueError('non-identified gate: ' + str(gate))
#     # print('mat_3d_before: {}'.format(mat_3d))
#     if convention == 'x-y-x':
#         phi, theta, psi = mat_to_euler_angles_xyx(mat_3d)
#         mat_3d_after = np.identity(3)
#         mat_3d_after = np.matmul(rotation_matrix([1,0,0], phi), mat_3d_after)
#         mat_3d_after = np.matmul(rotation_matrix([0,1,0], theta), mat_3d_after)
#         mat_3d_after = np.matmul(rotation_matrix([1,0,0], psi), mat_3d_after)
#         # print('mat_3d_after: {}'.format(mat_3d_after))
#         if not np.allclose(mat_3d, mat_3d_after):
#             raise ValueError('Wrong Euler Angles!')
#     elif convention == 'y-x-y':
#         phi, theta, psi = mat_to_euler_angles_yxy(mat_3d)

#     return phi, theta, psi 

# def gate_to_mat(step):
#     """
#         convert step object into matrix representation
#     """

#     if isinstance(step.gates[1].gate, gates.ZGate_Cplr_iSWAP):
#         mat = np.matrix('1,0,0,0; 0,0,1j,0; 0,1j,0,0; 0,0,0,1')
#     elif isinstance(step.gates[1].gate, gates.ZGate_Cplr_CZ):
#         mat = np.matrix('1,0,0,0; 0,1,0,0; 0,0,1,0; 0,0,0,-1')
#     else:
#         # if one-qubit gate
#         mat = 1
#         # print(step.gates[0].gate)
#         for gate in [step.gates[0].gate, step.gates[2].gate]:
#             # print(gate)
#             if (gate == gates.I):
#                 mat = np.kron(mat, np.matrix('1,0;0,1'))
#             elif (gate == gates.X2p):
#                 mat = np.kron(mat, 1/np.sqrt(2)*np.matrix('1,-1j;-1j,1'))
#             elif (gate == gates.X2m):
#                 mat = np.kron(mat, 1/np.sqrt(2)*np.matrix('1,1j;1j,1')) 
#             elif (gate == gates.Y2p):
#                 mat = np.kron(mat, 1/np.sqrt(2)*np.matrix('1,-1;1,1'))
#             elif (gate == gates.Y2m):
#                 mat = np.kron(mat, 1/np.sqrt(2)*np.matrix('1,1;-1,1'))
#             elif (gate == gates.Z2p):
#                 mat = np.kron(mat, 1/np.sqrt(2)*np.matrix('1,0;0,1j'))
#             elif (gate == gates.Z2m):
#                 mat = np.kron(mat, 1/np.sqrt(2)*np.matrix('1,0;0,-1j'))
#             elif (gate == gates.Xp):
#                 mat = np.kron(mat, np.matrix('0,-1j;-1j,0'))
#             elif (gate == gates.Xm):
#                 mat = np.kron(mat, np.matrix('0,1j;1j,0'))
#             elif (gate == gates.Yp):
#                 mat = np.kron(mat, np.matrix('0,-1;1,0'))
#             elif (gate == gates.Ym):
#                 mat = np.kron(mat, np.matrix('0,1;-1,0'))
#             elif (gate == gates.Zp):
#                 mat = np.kron(mat, np.matrix('1,0;0,-1'))
#             elif (gate == gates.Zm):
#                 mat = np.kron(mat, np.matrix('1,0;0,-1'))
#             elif isinstance(gate, gates.EulerZGate):
#                 mat = np.kron(mat, np.matrix([[1,0],[0,np.exp(1j*gate.theta)]]))
#     return mat


class TwoQubit_RB(Sequence):
    """Two qubit randomized benchmarking."""

    prev_seed = np.inf  # store the previous value
    prev_N_cliffords = np.inf  # store the previous value
    prev_interleave = np.inf  # store the previous value
    prev_interleaved_gate = np.inf  # store the previous value
    prev_sequence = ''
    prev_gate_seq = []

    filepath_lookup_table = ""

    # def _convert_z_to_euler_gates(self):
    #     # 1. find where two-qubit gate locates
    #     # 2. split sequence into sub-sequence by two qubit gates. Each sub-sequence consisting of only single qubit gates. 
    #     # 3. check if Euler Z gate exists in each sub-sequence 
    #     # 4. if there is Euler Z, then convert it into the Euler rotations
    #     # 5. build un new sequence list 

    #     # find where 2qb locates
    #     list_2qb_gate_indicies = list()
    #     list_steps_2qb_gate = list()
    #     for n in range(len(self.sequence_list)):
    #         step = self.sequence_list[n]
    #         if isinstance(step.gates[1].gate, gates.ZGate_Cplr_iSWAP):
    #             list_2qb_gate_indicies.append(n)
    #             list_steps_2qb_gate.append(step)


    #     # split sequence into sub-sequence by two qubit gates. 
    #     list_subseqs = list()
    #     step_readout = None
    #     for j in range(len(list_2qb_gate_indicies)+1):
    #         if j == 0:
    #             list_subseqs.append(self.sequence_list[:list_2qb_gate_indicies[j]])
    #         elif j == len(list_2qb_gate_indicies):
    #             # log.info(self.sequence_list[-1])
    #             if isinstance(self.sequence_list[-1].gates[0].gate, gates.ReadoutGate):
    #                 list_subseqs.append(self.sequence_list[list_2qb_gate_indicies[j-1]+1:-1])
    #                 step_readout = self.sequence_list[-1]
    #             else:
    #                 list_subseqs.append(self.sequence_list[list_2qb_gate_indicies[j-1]+1:])

    #         else:
    #             list_subseqs.append(self.sequence_list[list_2qb_gate_indicies[j-1]+1: list_2qb_gate_indicies[j]])
                
    #     # find which sub-sequence has eulerZ gates
    #     list_eulerz_indices = list()
    #     for k in range(len(list_subseqs)):
    #         eulerz_found = False
    #         for l in range(len(list_subseqs[k])):
    #             step = list_subseqs[k][l] 
    #             if isinstance(step.gates[0].gate, gates.EulerZGate):
    #                 list_eulerz_indices.append(k)
    #                 eulerz_found = True
    #         if ((not eulerz_found) and len(list_subseqs[k])>3):
    #             # if the number of single qubit gates more than 3, consider rewrite in euler-format
    #             list_eulerz_indices.append(k)

    #     # for sub-sequence having euler-Z gates, convert them into XYX-convention
    #     list_subseqs_eulerz = list()
    #     for index in list_eulerz_indices:
    #         subseq = list_subseqs[index]
    #         len_seq = len(subseq)
    #         list_qb1_gates = list()
    #         list_qb2_gates = list()
    #         for m in range(len_seq):
    #             list_qb1_gates.append(subseq[m].gates[0].gate)
    #             list_qb2_gates.append(subseq[m].gates[2].gate)
    #         # print('list_qb1_gates: {}'.format(list_qb1_gates))
    #         list_qb1_euler_gates = gate_to_euler_gates(list_qb1_gates, convention = 'x-y-x')
    #         # print('list_qb2_gates: {}'.format(list_qb2_gates))
    #         list_qb2_euler_gates = gate_to_euler_gates(list_qb2_gates, convention = 'x-y-x')
    #         subseq_eulerz = []
    #         for m in range(3):
    #             step = Step()
    #             step.add_gate(0,list_qb1_euler_gates[m])
    #             step.add_gate(1,gates.IdentityGate(width = None))
    #             step.add_gate(2,list_qb2_euler_gates[m])
    #             subseq_eulerz.append(step)
    #         list_subseqs_eulerz.append(subseq_eulerz)

    #     # build-up the sequence
    #     new_sequence_list = list()
    #     cnt_eulerz = 0
    #     cnt_2qb_gate = 0
    #     for j, subseq in enumerate(list_subseqs):
    #         if j in list_eulerz_indices:
    #             # print(list_subseqs_eulerz[cnt])
    #             new_sequence_list.extend(list_subseqs_eulerz[cnt_eulerz])
    #             cnt_eulerz += 1
    #         else:
    #             new_sequence_list.extend(subseq)
    #         if j != len(list_subseqs)-1:
    #             new_sequence_list.append(list_steps_2qb_gate[cnt_2qb_gate])
    #             cnt_2qb_gate +=1
    #     if step_readout is not None:
    #         new_sequence_list.append(step_readout)

    #     self.sequence_list = new_sequence_list

    # def _explode_composite_gates(self):
    #     # Loop through the sequence until all CompositeGates are removed
    #     # Note that there could be nested CompositeGates
    #     n = 0
    #     while n < len(self.sequence_list):
    #         step = self.sequence_list[n]
    #         i = 0
    #         while i < len(step.gates):
    #             gate = step.gates[i]
    #             if isinstance(gate.gate, gates.CompositeGate):
    #                 # log.info('In exploded composite, handling composite gate {} at step {}'.format(gate, n))
    #                 for m, g in enumerate(gate.gate.sequence):
    #                     new_gate = [x.gate for x in g.gates]
    #                     # Single gates shouldn't be lists
    #                     if len(new_gate) == 1:
    #                         new_gate = new_gate[0]

    #                     # Translate gate qubit number to device qubit number
    #                     new_qubit = [x.qubit for x in g.gates]
    #                     for j, q in enumerate(new_qubit):
    #                         if isinstance(q, int):
    #                             if isinstance(gate.qubit, int):
    #                                 new_qubit[j] = gate.qubit
    #                                 continue
    #                             new_qubit[j] = gate.qubit[q]
    #                         else:
    #                             new_qubit[j] = []
    #                             for k in q:
    #                                 new_qubit[j].append(gate.qubit[k])

    #                     # Single qubit shouldn't be lists
    #                     if len(new_qubit) == 1:
    #                         new_qubit = new_qubit[0]
    #                     log.info('In explode composite; modifying {} by adding gate {} at index {}'.format(gate, new_gate, n+m))
    #                     log.info('before adding new-qubit(' + str(new_qubit) + ') new-gate (' + str(new_gate) + '): ' + str(self.sequence_list))
    #                     self.sequence.add_gate(
    #                         copy.copy(new_qubit), copy.copy(new_gate), index=n + m)
    #                     log.info('after: ' + str(self.sequence_list))

    #                 del step.gates[i]
    #                 # log.info('In composite gates, removing step {}'.format(i))

    #                 continue
    #             i = i + 1
    #         n = n + 1

    #     # Remove any empty steps where the composite gates were
    #     i = 0
    #     while i < len(self.sequence_list):
    #         step = self.sequence_list[i]
    #         # log.info(str(step.gates))
    #         if len(step.gates) == 0:
    #             # log.info('In composite gates, removing step {}'.format(i))
    #             del self.sequence_list[i]
    #             continue
    #         i = i + 1

    #     for i, step in enumerate(self.sequence_list):
    #         log.info('At end of explode, step {} is {}'.format(i, step.gates))

    def generate_sequence(self, config):
        """
        Generate sequence by adding gates/pulses to waveforms.

        Parameters
        ----------
        config: dict
            configuration

        Returns
        -------
        """

        # get parameters

        sequence = config['Sequence']
        # qubits_to_benchmark = [int(config['Qubits to Benchmark'][0]) - 1,
        #                        int(config['Qubits to Benchmark'][2]) - 1]


        # Number of Cliffords to generate
        # N_cliffords = int(config['Number of Cliffords'])
        N_cliffords = int(config.get('Number of Cliffords', 1))
        rnd_seed = config.get('Random Seed', 0)
        interleave = config.get('Interleave 2-QB Gate', False)
        multi_seq = config.get('Output multiple sequences', False)
        write_seq = config.get('Write sequence as txt file', False)
        generator = config.get('Native 2-QB gate', 'CZ_Cplr')
        alternate_polarity = bool(config.get('Alternate 2QB pulse polarity', False))

        if generator == 'CZ':
            qubits_to_benchmark = [int(config['Qubits to Benchmark'][0]) - 1, 
                int(config['Qubits to Benchmark'][2]) - 1] 

        elif generator in ['iSWAP_Cplr', 'CZ_Cplr']:
            qubits_to_benchmark = [int(config['Qubits to Benchmark (Coupler)'][0]) - 1, 
                int(config['Qubits to Benchmark (Coupler)'][2]) - 1,
                int(config['Qubits to Benchmark (Coupler)'][4]) - 1,] 

        rnd.seed(rnd_seed + N_cliffords*100)
        # rnd.seed(rnd_seed)
        if interleave is True:
            interleaved_gate = config['Interleaved 2-QB Gate']
        else:
            interleaved_gate = np.inf

        # generate new randomized clifford gates only if configuration changes
        if (self.prev_sequence != sequence or
                self.prev_rnd_seed != rnd_seed or
                self.prev_N_cliffords != N_cliffords or
                self.prev_interleave != interleave or
                self.prev_generator != generator or
                self.prev_alternate_polarity != alternate_polarity or 
                multi_seq or
                self.prev_interleaved_gate != interleaved_gate):

            self.prev_rnd_seed = rnd_seed
            self.prev_N_cliffords = N_cliffords
            self.prev_interleave = interleave
            self.prev_sequence = sequence
            self.prev_generator = generator
            self.prev_alternate_polarity = alternate_polarity

            multi_gate_seq = []

            # Generate 2QB RB sequence
            cliffordSeq1 = []
            cliffordSeq2 = []
            cliffordSeqCplr = []
            for j in range(N_cliffords):
                # log.info('Seed number: %d'%(randomize))
                rndnum = rnd.randint(0, 11519)
                # rndnum = rnd.randint(0, 576) #Only applying single qubit gates
                add_twoQ_clifford(rndnum, cliffordSeq1, cliffordSeq2, gate_seq_Cplr = cliffordSeqCplr, generator = generator)
                # If interleave gate,
                if interleave is True:
                    self.prev_interleaved_gate = interleaved_gate
                    if interleaved_gate == 'CZ':
                        cliffordSeq1.append(gates.I)
                        cliffordSeq2.append(gates.CZ)
                    elif interleaved_gate == 'CZEcho':
                        # CZEcho is a composite gate, so get each gate
                        gate = gates.CZEcho
                        for g in gate.sequence:
                            cliffordSeq1.append(g[1])
                            cliffordSeq2.append(g[0])
                    elif interleaved_gate == 'iSWAP_Cplr':
                        gate = gates.iSWAP_Cplr_Z_behind
                        cliffordSeq1.append(gates.I)
                        # cliffordSeqCplr.append(gates.iSWAP_Cplr)
                        cliffordSeqCplr.append(gates.iSWAP_Cplr_Z_behind)
                        cliffordSeq2.append(gates.I)
                    elif interleaved_gate == 'CZ_Cplr':
                        gate = gates.CZ_Cplr
                        cliffordSeq1.append(gates.I)
                        cliffordSeqCplr.append(gates.CZ_Cplr)
                        cliffordSeq2.append(gates.I)
                    elif interleaved_gate == 'I':
                        # TBA: adjust the duration of I gates?
                        # log.info('Qubits to benchmark: ' + str(qubits_to_benchmark))
                        # gate = gates.I(width = self.pulses_2qb[qubit]).value
                        I_2QB = gates.IdentityGate(width =config.get('Width, 2QB'))

                        cliffordSeq1.append(I_2QB)
                        cliffordSeq2.append(I_2QB)
                        # cliffordSeq1.append(gates.I)
                        # cliffordSeq2.append(gates.I)


            # remove redundant Identity gates for cliffordSeq1
            index_identity_clifford = [] # find where Identity gates are
            for p in range(len(cliffordSeq1)):

                if not cliffordSeqCplr: # If clifford_seq_Cplr is an empty list,
                    if (cliffordSeq1[p] == gates.I and cliffordSeq2[p] == gates.I):
                        index_identity_clifford.append(p)
                else:
                    if (cliffordSeq1[p] == gates.I and cliffordSeq2[p] == gates.I and cliffordSeqCplr[p] == gates.I):
                        index_identity_clifford.append(p)

            cliffordSeq1 = [m for n, m in enumerate(cliffordSeq1) if n not in index_identity_clifford]
            cliffordSeq2 = [m for n, m in enumerate(cliffordSeq2) if n not in index_identity_clifford]
            if cliffordSeqCplr: 
                cliffordSeqCplr = [m for n, m in enumerate(cliffordSeqCplr) if n not in index_identity_clifford]

            # log.info('*** clifford sequence *** ')
            # log.info("QB1 clifford gate sequence: " + str([cliffords.Gate_to_strGate(g) for g in cliffordSeq1]))
            # log.info("QB2 clifford gate sequence: " + str([cliffords.Gate_to_strGate(g) for g in cliffordSeq2]))
            # if generator in ['iSWAP_Cplr', 'CZ_Cplr']:
            #     log.info("Coupler clifford gate sequence: " + str([cliffords.Gate_to_strGate(g) for g in cliffordSeqCplr]))
            # log.info("=================================================")

            # get recovery gate seq
            (recoverySeq1, recoverySeq2, recoverySeqCplr) = self.get_recovery_gate(
                cliffordSeq1, cliffordSeq2, config, gate_seq_Cplr = cliffordSeqCplr, generator = generator)

            # Remove redundant identity gates in recovery gate seq
            index_identity_recovery = [] # find where Identity gates are
            for p in range(len(recoverySeq1)):
                if not recoverySeqCplr: # If recoverySeqCplr is an empty list,
                    if (recoverySeq1[p] == gates.I and recoverySeq2[p] == gates.I):
                        index_identity_recovery.append(p)
                else:
                    if (recoverySeq1[p] == gates.I and recoverySeq2[p] == gates.I and recoverySeqCplr[p] == gates.I):
                        index_identity_recovery.append(p)
            recoverySeq1 = [m for n, m in enumerate(recoverySeq1) if n not in index_identity_recovery]
            recoverySeq2 = [m for n, m in enumerate(recoverySeq2) if n not in index_identity_recovery]
            if recoverySeqCplr:
                recoverySeqCplr = [m for n, m in enumerate(recoverySeqCplr) if n not in index_identity_recovery]

            # Construct the total gate sequence.
            gateSeq1 = []
            gateSeq2 = []
            gateSeqCplr = []
            gateSeq1.extend(cliffordSeq1)
            gateSeq1.extend(recoverySeq1)
            gateSeq2.extend(cliffordSeq2)
            gateSeq2.extend(recoverySeq2)
            gateSeqCplr.extend(cliffordSeqCplr)
            gateSeqCplr.extend(recoverySeqCplr)

            # Avoid Error: zero-size array to reduction operation maximum which has no identity (05/05/2019)
            if (gateSeq1 == [] and gateSeq2 == []):
                gateSeq1.append(gates.I)
                gateSeq2.append(gates.I)
                if generator in ['iSWAP_Cplr', 'CZ_Cplr']:
                    gateSeqCplr.append(gates.I)

            # Alternate polarity of 2qb pulse
            if (alternate_polarity == True):
                if generator == 'CZ_Cplr':
                    cnt = 0
                    for i, gate in enumerate(gateSeqCplr):
                        if gate == gates.CZ_Cplr:
                            if cnt %2 == 0:
                                pass
                            else:
                                gateSeqCplr[i] = gates.CZ_Cplr_opposite
                            cnt = cnt + 1

            # test the recovery gate
            psi_gnd = np.matrix('1; 0; 0; 0') # ground state |00>
            if write_seq == True:
                import os
                from datetime import datetime
                directory = os.path.join(path_currentdir,'2QB_RBseq')
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = datetime.now().strftime('%Y-%m-%d %H-%M-%S-f')[:-3] + '_N_cliffords=%d_rnd_seed=%d.txt'%(N_cliffords,rnd_seed)
                # filename = datetime.now().strftime('%Y-%m-%d %H-%M-%S-%f')[:-3] + '_N_cliffords=%d_seed=%d.txt'%(N_cliffords,randomize)
                filepath = os.path.join(directory,filename)
                log.info('make file: ' + filepath)
                with open(filepath, "w") as text_file:
                    print('New Sequence', file=text_file)
                    if (generator in ['iSWAP_Cplr', 'CZ_Cplr']):
                        for i in range(len(gateSeq1)):
                            print("Index: %d, Gate: ["%(i) + cliffords.Gate_to_strGate(gateSeq1[i]) + ", " + cliffords.Gate_to_strGate(gateSeqCplr[i]) + ", " + cliffords.Gate_to_strGate(gateSeq2[i]) +']', file=text_file)
                        # for i in range(len(cliffordSeq1)):
                        #      print("CliffordIndex: %d, Gate: ["%(i) + cliffords.Gate_to_strGate(cliffordSeq1[i]) + ", " + cliffords.Gate_to_strGate(cliffordSeq2[i]) +']', file=text_file)
                        # for i in range(len(recoverySeq1)):
                        #      print("RecoveryIndex: %d, Gate: ["%(i) + cliffords.Gate_to_strGate(recoverySeq1[i]) + ", " + cliffords.Gate_to_strGate(recoverySeq2[i]) +']', file=text_file)

                    else:
                        for i in range(len(gateSeq1)):
                            print("Index: %d, Gate: ["%(i) + cliffords.Gate_to_strGate(gateSeq1[i]) + ", " + cliffords.Gate_to_strGate(gateSeq2[i]) +']', file=text_file)
                        for i in range(len(cliffordSeq1)):
                             print("CliffordIndex: %d, Gate: ["%(i) + cliffords.Gate_to_strGate(cliffordSeq1[i]) + ", " + cliffords.Gate_to_strGate(cliffordSeq2[i]) +']', file=text_file)
                        for i in range(len(recoverySeq1)):
                             print("RecoveryIndex: %d, Gate: ["%(i) + cliffords.Gate_to_strGate(recoverySeq1[i]) + ", " + cliffords.Gate_to_strGate(recoverySeq2[i]) +']', file=text_file)
            # psi = np.matmul(self.evaluate_sequence(gateSeq1, gateSeq2), psi_gnd)
            psi = np.matmul(self.evaluate_sequence(gateSeq1, gateSeq2, gate_seq_Cplr = gateSeqCplr, generator = generator), psi_gnd)

            # np.set_printoptions(precision=2)
            # log.info('The matrix of the overall gate sequence:')
            # log.info(self.evaluate_sequence(gateSeq1, gateSeq2))
            # log.info(self.evaluate_sequence(gateSeq1, gateSeq2, gate_seq_Cplr = gateSeqCplr, generator = generator))

            log.info('--- TESTING THE RECOVERY GATE ---')
            # log.info('The probability amplitude of the final state vector: ' + str(np.matrix(psi).flatten()))
            log.info('The population of the ground state after the gate sequence: %.4f'%(np.abs(psi[0,0])**2))
            log.info('-------------------------------------------')

            # Assign two qubit gate sequence to where we want
            # for i in range(qubits_to_benchmark[0] - 1):
            #     multi_gate_seq.append([None] * len(gateSeq1))
            # log.info('')
            if (generator == 'CZ'):
                # for i in range(qubits_to_benchmark[0] - 1):
                #     multi_gate_seq.append([None] * len(gate_seq_1))
                multi_gate_seq.append(gateSeq2)
                multi_gate_seq.append(gateSeq1)
                # for i in range(self.n_qubit - qubits_to_benchmark[1]):
                #     multi_gate_seq.append([None] * len(gate_seq_1))
            elif (generator in ['iSWAP_Cplr', 'CZ_Cplr']):
                # multi_gate_seq.append(gateSeq2)
                # multi_gate_seq.append(gateSeqCplr)
                # multi_gate_seq.append(gateSeq1)

                # log.info('\n\n')
                # log.info('SWAP THE ORDER to use better 1QB gate (QB1 is better than QB 2)!')
                # log.info('\n\n')
                multi_gate_seq.append(gateSeq2)
                multi_gate_seq.append(gateSeqCplr)
                multi_gate_seq.append(gateSeq1)
            # multi_gate_seq.append(gateSeq2)
            # multi_gate_seq.append(gateSeq1)
            # for i in range(self.n_qubit - qubits_to_benchmark[1]):
            #     multi_gate_seq.append([None] * len(gateSeq1))

            # transpose list of lists
            multi_gate_seq = list(map(list, itertools.zip_longest(*multi_gate_seq, fillvalue=gates.I))) # Not to chop

            # self.add_gates(multi_gate_seq)
            # for gate_seq in multi_gate_seq:
            #     if gate_seq[0] == gates.CZ:
            #         self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq[0])
            #     else:
            #         self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq)
            for gate_seq in multi_gate_seq:
                # for i in range(len(gate_seq)):
                #     log.info(str(gate_seq[i]))
                if generator == 'CZ':
                    if gate_seq[0] == gates.CZ:
                        # self.add_gate(qubit=[0, 1], gate=gate_seq[0])
                        self.add_gate(qubit = qubits_to_benchmark, gate = gate_seq[0])
                    else:
                        # self.add_gate(qubit=[0, 1], gate=gate_seq)
                        self.add_gate(qubit = qubits_to_benchmark, gate = gate_seq)
                elif generator == 'iSWAP_Cplr':
                    if gate_seq[1] in [gates.iSWAP_Cplr, gates.iSWAP_Cplr_Z_behind, gates.iSWAP_Cplr_Z_ahead]:
                        self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq[1])
                    else:
                        self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq)
                elif generator == 'CZ_Cplr':
                    if gate_seq[1] in [gates.CZ_Cplr, gates.CZ_Cplr_opposite]:
                        self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq[1])

                    else:
                        self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq)

            self.prev_gate_seq = multi_gate_seq
        else:
            # for gate_seq in self.prev_gate_seq:
            #     if gate_seq[0] == gates.CZ:
            #         self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq[0])
            #     else:
            #         self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq)
            for gate_seq in self.prev_gate_seq:
                if generator == 'CZ':
                    if gate_seq[0] == gates.CZ:
                        # self.add_gate(qubit=[0, 1], gate=gate_seq[0])
                        self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq[0])
                    else:
                        # self.add_gate(qubit=[0, 1], gate=gate_seq)
                        self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq)
                elif generator == 'iSWAP_Cplr':
                    if gate_seq[1] in [gates.iSWAP_Cplr, gates.iSWAP_Cplr_Z_behind, gates.iSWAP_Cplr_Z_ahead]:
                        self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq[1])
                    else:
                        self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq)
                elif generator == 'CZ_Cplr':
                    if gate_seq[1] in [gates.CZ_Cplr, gates.CZ_Cplr_opposite]:
                        self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq[1])
                    else:
                        self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq)

    # def print_sequence(self):

    def evaluate_sequence(self, gate_seq_1, gate_seq_2, gate_seq_Cplr = None, generator = 'CZ'):
        """
        Evaluate the two qubit gate sequence.

        Parameters
        ----------
        gate_seq_1: list of class Gate (defined in "gates.py")
            The gate sequence applied to Qubit "1"

        gate_seq_2: list of class Gate (defined in "gates.py")
            The gate sequence applied to Qubit "2"

        Returns
        -------
        twoQ_gate: np.matrix (shape = (4,4))
            The evaulation result.
        """
        twoQ_gate = np.matrix(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        # log.info('gate_seq_1: ' + str(gate_seq_1))
        # log.info('gate_seq_Cplr: ' + str(gate_seq_Cplr))
        # log.info('gate_seq_2: ' + str(gate_seq_2))
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
                if ((gate_seq_Cplr[i] == gates.iSWAP_Cplr)
                    or (gate_seq_Cplr[i] == gates.iSWAP_Cplr_Z_behind)
                    or (gate_seq_Cplr[i] == gates.iSWAP_Cplr_Z_ahead)):
                    gate_12 = np.matmul(
                        np.matrix([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0],
                                   [0, 0, 0, 1]]), gate_12)
            elif generator == 'CZ_Cplr':
                if gate_seq_Cplr[i] == gates.CZ_Cplr:
                    gate_12 = np.matmul(
                        np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                                   [0, 0, 0, -1]]), gate_12)
            twoQ_gate = np.matmul(gate_12, twoQ_gate)

            # gate_12 = np.kron(gate_1, gate_2)
            # if (gate_seq_1[i] == gates.CZ or gate_seq_2[i] == gates.CZ):
            #     gate_12 = np.matmul(
            #         np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
            #                    [0, 0, 0, -1]]), gate_12)
            # iSWAP <- To be added.
            # elif (gate_seq_1[i] == gates.iSWAP or gate_seq_2[i] == gates.iSWAP):
            #     gate_12 = np.matmul(
            #         np.matrix([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0],
            #                    [0, 0, 0, 1]]), gate_12)

            # twoQ_gate = np.matmul(gate_12, twoQ_gate)
        # log.info('two qubit gate: ' + str(twoQ_gate))
        return twoQ_gate
    def get_recovery_gate(self, gate_seq_1, gate_seq_2, config, gate_seq_Cplr = [], generator = 'CZ'):
        """
        Get the recovery (the inverse) gate

        Parameters
        ----------
        gate_seq_1: list of class Gate
            The gate sequence applied to Qubit "1"

        gate_seq_2: list of class Gate
            The gate sequence applied to Qubit "2"

        config: dict
            The configuration

        gate_seq_Cplr: list of class Gate (defined in "gates.py")
            The gate sequence applied to Coupler (optional)

        generator: string
            Type of Native 2QB gate (optional)

        Returns
        -------
        (recovery_seq_1, recovery_seq_2): tuple of the lists
            The recovery gate
        """


        qubit_state = np.matrix(
            '1; 0; 0; 0')  # initial state: ground state |00>

        qubit_state = np.matmul(self.evaluate_sequence(
            gate_seq_1, gate_seq_2, gate_seq_Cplr = gate_seq_Cplr, generator = generator), qubit_state)

        # find recovery gate which makes qubit_state return to initial state
        total_num_cliffords = 11520
        recovery_seq_1 = []
        recovery_seq_2 = []
        recovery_seq_Cplr = []

        # Search the recovery gate in two Qubit clifford group
        find_cheapest = config['Find the cheapest recovery Clifford']

        cheapest_recovery_seq_1 = []
        cheapest_recovery_seq_2 = []
        cheapest_recovery_seq_Cplr = []

        # log.info('*** get recovery gate *** ')
        if (find_cheapest == True):
            min_N_2QB_gate = np.inf
            min_N_1QB_gate = np.inf
            max_N_I_gate = -np.inf
            cheapest_index = None

            use_lookup_table = config['Use a look-up table']
            if (use_lookup_table == True):
                filepath_lookup_table = config['File path of the look-up table']
                if len(filepath_lookup_table) == 0:
                    if (generator == 'CZ'):
                        filepath_lookup_table = os.path.join(path_currentdir, 'recovery_rb_table.pickle')
                    elif (generator == 'iSWAP_Cplr'):
                        filepath_lookup_table = os.path.join(path_currentdir, 'recovery_rb_table_iSWAP_Cplr.pickle')
                    elif (generator == 'CZ_Cplr'):
                        filepath_lookup_table = os.path.join(path_currentdir, 'recovery_rb_table_CZ_Cplr.pickle')
                if filepath_lookup_table != self.filepath_lookup_table:
                    # log.info("Load Look-up table.")
                    self.filepath_lookup_table = filepath_lookup_table
                    self.dict_lookup_table = cliffords.loadData(filepath_lookup_table)
                stabilizer = cliffords.get_stabilizer(qubit_state)
                for index, item in enumerate(self.dict_lookup_table['psi_stabilizer']):
                    if stabilizer == item:
                        seq1 = self.dict_lookup_table['recovery_gates_QB1'][index]
                        for str_Gate in seq1:
                            cheapest_recovery_seq_1.append(cliffords.strGate_to_Gate(str_Gate))
                        seq2 = self.dict_lookup_table['recovery_gates_QB2'][index]
                        for str_Gate in seq2:
                            cheapest_recovery_seq_2.append(cliffords.strGate_to_Gate(str_Gate))

                        if generator in ['iSWAP_Cplr', 'CZ_Cplr']:
                            seqCplr = self.dict_lookup_table['recovery_gates_Cplr'][index]
                            for str_Gate in seqCplr:
                                cheapest_recovery_seq_Cplr.append(cliffords.strGate_to_Gate(str_Gate))

                        # log.info("=== FOUND THE CHEAPEST RECOVERY GATE IN THE LOOK-UP TABLE. ===")
                        # log.info("QB1 recovery gate sequence: " + str(seq1))
                        # log.info("QB2 recovery gate sequence: " + str(seq2))
                        # if generator in ['iSWAP_Cplr', 'CZ_Cplr']:
                        #     log.info("Coupler recovery gate sequence: " + str (seqCplr))
                        # log.info("=================================================")
                        return (cheapest_recovery_seq_1, cheapest_recovery_seq_2, cheapest_recovery_seq_Cplr)

            log.info("--- COULDN'T FIND THE RECOVERY GATE IN THE LOOK-UP TABLE... ---")


        # Calculate the matrix of the clifford sequence
        matrix_cliffords = self.evaluate_sequence(gate_seq_1,gate_seq_2, gate_seq_Cplr = gate_seq_Cplr, generator = generator)

        for i in range(total_num_cliffords):
            recovery_seq_1 = []
            recovery_seq_2 = []
            recovery_seq_Cplr = []
            add_twoQ_clifford(i, recovery_seq_1, recovery_seq_2, gate_seq_Cplr = recovery_seq_Cplr, generator = generator)

            # Calculate the matrix of the recovery clifford
            matrix_recovery = self.evaluate_sequence(recovery_seq_1, recovery_seq_2, gate_seq_Cplr = recovery_seq_Cplr, generator = generator)

            # Calculate the matrix of the total clifford sequence
            matrix_total = np.matmul(matrix_recovery,matrix_cliffords)
            if (CheckIdentity(matrix_total)):
                if (find_cheapest == True):
                    # Less 2QB Gates, Less 1QB Gates, and More I Gates = the cheapest gates.
                    # The priority: less 2QB gates > less 1QB gates > more I gates
                    N_2QB_gate = 0
                    N_1QB_gate = 0
                    N_I_gate = 0

                    # count the numbers of the gates
                    for j in range(len(recovery_seq_1)):
                        if generator == 'CZ':
                            if (recovery_seq_1[j] == gates.CZ or recovery_seq_2[j] == gates.CZ):
                                N_2QB_gate += 1
                            else:
                                N_1QB_gate += 2
                        elif generator == 'iSWAP_Cplr':
                            if ((recovery_seq_Cplr[j] == gates.iSWAP_Cplr)
                                or (recovery_seq_Cplr[j] == gates.iSWAP_Cplr_Z_behind)
                                or (recovery_seq_Cplr[j] == gates.iSWAP_Cplr_Z_ahead)):
                                N_2QB_gate += 1
                            else:
                                N_1QB_gate += 2
                        elif generator == 'CZ_Cplr':
                            if (recovery_seq_Cplr[j] == gates.CZ_Cplr):
                                N_2QB_gate += 1
                            else:
                                N_1QB_gate += 2

                        if (recovery_seq_1[j] == gates.I):
                            N_I_gate += 1
                        if (recovery_seq_2[j] == gates.I):
                            N_I_gate += 1

                    if (N_2QB_gate <= min_N_2QB_gate): # if it has less 2QB gates, always update it
                        min_N_2QB_gate, min_N_1QB_gate, max_N_I_gate, cheapest_index = (N_2QB_gate, N_1QB_gate, N_I_gate, j)

                        if (N_1QB_gate <= min_N_1QB_gate): # *only if it has less 2QB gates*, check whether it has less 1QB gates
                            min_N_2QB_gate, min_N_1QB_gate, max_N_I_gate, cheapest_index = (N_2QB_gate, N_1QB_gate, N_I_gate, j)

                            if (N_I_gate >= max_N_I_gate): # *only if it has less 2QB gates & only if it has less 1QB gates*, check whether it has more I gates
                                min_N_2QB_gate, min_N_1QB_gate, max_N_I_gate, cheapest_index = (N_2QB_gate, N_1QB_gate, N_I_gate, j)

                    # check whether it is the cheapest
                    # if it has less 2QB gates, always update it.
                    if (N_2QB_gate < min_N_2QB_gate):
                        min_N_2QB_gate, min_N_1QB_gate, max_N_I_gate, cheapest_index = (N_2QB_gate, N_1QB_gate, N_I_gate, j)
                        log.info('the cheapest sequence update! [N_2QB_gate, N_1QB_gate, N_I_gate, seq. index] ' + str([min_N_2QB_gate, min_N_1QB_gate, max_N_I_gate, cheapest_index]))
                    else:
                        # if it has equal # of 2QB gates and less 1QB gates, update it.
                        if (N_2QB_gate == min_N_2QB_gate and
                            N_1QB_gate < min_N_1QB_gate):
                            min_N_2QB_gate, min_N_1QB_gate, max_N_I_gate, cheapest_index = (N_2QB_gate, N_1QB_gate, N_I_gate, j)
                            log.info('the cheapest sequence update! [N_2QB_gate, N_1QB_gate, N_I_gate, seq. index] ' + str([min_N_2QB_gate, min_N_1QB_gate, max_N_I_gate, cheapest_index]))
                        else:
                            # if it has equal # of 2QB & 1QB gates, and more 1QB gates, update it.
                            if (N_2QB_gate == min_N_2QB_gate and
                                N_1QB_gate == min_N_1QB_gate and
                                N_I_gate >= max_N_I_gate):
                                min_N_2QB_gate, min_N_1QB_gate, max_N_I_gate, cheapest_index = (N_2QB_gate, N_1QB_gate, N_I_gate, j)
                                log.info('the cheapest sequence update! [N_2QB_gate, N_1QB_gate, N_I_gate, seq. index] ' + str([min_N_2QB_gate, min_N_1QB_gate, max_N_I_gate, cheapest_index]))

                else:
                    return(recovery_seq_1,recovery_seq_2, recovery_seq_Cplr)

        if (find_cheapest == True):
            recovery_seq_1 = []
            recovery_seq_2 = []
            recovery_seq_Cplr = []
            log.info('The index of the cheapest recovery clifford: %d'%(cheapest_index))
            add_twoQ_clifford(cheapest_index, recovery_seq_1, recovery_seq_2, gate_seq_Cplr = recovery_seq_Cplr, generator = generator)


        if (recovery_seq_1 == [] and recovery_seq_2 == []):
            recovery_seq_1 = [None]
            recovery_seq_2 = [None]
            recovery_seq_Cplr = [None]


        return (recovery_seq_1, recovery_seq_2, recovery_seq_Cplr)

class TwoQubit_XEB(Sequence):
    """Two qubit randomized benchmarking."""
    prev_rnd_seed = np.inf  # store the previous value
    prev_N_cycles = np.inf  # store the previous value
    prev_sequence = ''
    prev_gate_seq = []
    prev_alternate_polarity = False

    def generate_sequence(self, config):
        """
        Generate sequence by adding gates/pulses to waveforms.

        Parameters
        ----------
        config: dict
            configuration

        Returns
        -------
        """

        # get parameters

        sequence = config['Sequence']
        rnd_seed = config.get('Random Seed', 0)
        N_cycles = int(config.get('Number of Cycles', 1))
        multi_seq = config.get('Output multiple sequences', False)
        write_seq = config.get('Write sequence as txt file', False)
        generator = config.get('Native 2-QB gate', 'CZ_Cplr')
        alternate_polarity = bool(config.get('Alternate 2QB pulse polarity', False))

        if generator in ['CZ']:
            qubits_to_benchmark = [int(config['Qubits to Benchmark'][0]) - 1, 
                int(config['Qubits to Benchmark'][2]) - 1] 

        elif generator in ['iSWAP_Cplr', 'CZ_Cplr', 'I']:
            qubits_to_benchmark = [int(config['Qubits to Benchmark (Coupler)'][0]) - 1, 
                int(config['Qubits to Benchmark (Coupler)'][2]) - 1,
                int(config['Qubits to Benchmark (Coupler)'][4]) - 1,] 


        rnd.seed(rnd_seed + N_cycles*100)
        log.info('rnd_seed: %d'%(rnd_seed + N_cycles*100))

        # generate new randomized clifford gates only if configuration changes
        if (self.prev_sequence != sequence or
                self.prev_rnd_seed != rnd_seed or
                self.prev_N_cycles != N_cycles or
                self.prev_generator != generator or
                self.prev_alternate_polarity != alternate_polarity or
                multi_seq):

            self.prev_rnd_seed = rnd_seed
            self.prev_N_cycles = N_cycles
            self.prev_sequence = sequence
            self.prev_generator = generator
            self.prev_alternate_polarity = alternate_polarity

            multi_gate_seq = []

            # Generate 2QB XEB sequence
            gateSeq1 = []
            gateSeq2 = []
            gateSeqCplr = []
            for j in range(N_cycles):

                # add random single qubit gates 
                rndnum1 = rnd.randint(0, 2)
                rndnum2 = rnd.randint(0, 2)
                add_singleQ_forXEB(rndnum1, gateSeq1)
                add_singleQ_forXEB(rndnum2, gateSeq2)
                if generator in ['iSWAP_Cplr', 'CZ_Cplr']:
                    gateSeqCplr.append(gates.I)
                # add two qubit gate 
                if generator == 'CZ':
                    gateSeq1.append(gates.I)
                    gateSeq2.append(gates.CZ)
                elif generator == 'I':
                    # gateSeq1.append(gates.I)
                    # gateSeq2.append(gates.I)
                    pass
                elif generator == 'iSWAP_Cplr':
                    gateSeq1.append(gates.I)
                    # gateSeqCplr.append(gates.iSWAP_Cplr)
                    gateSeqCplr.append(gates.iSWAP_Cplr_Z_ahead)
                    gateSeq2.append(gates.I)
                elif generator == 'CZ_Cplr':
                    gateSeq1.append(gates.I)
                    gateSeqCplr.append(gates.CZ_Cplr)
                    gateSeq2.append(gates.I)

                
            if (generator in ['CZ']):
                multi_gate_seq.append(gateSeq1)
                multi_gate_seq.append(gateSeq2)
            elif (generator in ['iSWAP_Cplr', 'CZ_Cplr', 'I']):
                log.info('\n\n')
                log.info('SWAP THE ORDER!')
                log.info('\n\n')
                multi_gate_seq.append(gateSeq1)
                multi_gate_seq.append(gateSeqCplr)
                multi_gate_seq.append(gateSeq2)

            # Alternate polarity of 2qb pulse
            if (alternate_polarity == True):
                if generator == 'CZ_Cplr':
                    cnt = 0
                    for i, gate in enumerate(gateSeqCplr):
                        if gate == gates.CZ_Cplr:
                            if cnt %2 == 0:
                                pass
                            else:
                                gateSeqCplr[i] = gates.CZ_Cplr_opposite
                            cnt = cnt + 1

            # transpose list of lists
            multi_gate_seq = list(map(list, itertools.zip_longest(*multi_gate_seq, fillvalue=gates.I))) # Not to chop

            for gate_seq in multi_gate_seq:
                if generator == 'CZ':
                    if gate_seq[0] == gates.CZ:
                        self.add_gate(qubit = qubits_to_benchmark, gate = gate_seq[0])
                    else:
                        self.add_gate(qubit = qubits_to_benchmark, gate = gate_seq)
                elif generator == 'I':
                    self.add_gate(qubit = qubits_to_benchmark, gate = gate_seq)
                elif generator == 'iSWAP_Cplr':
                    if ((gate_seq[1] == gates.iSWAP_Cplr)
                        or (gate_seq[1] == gates.iSWAP_Cplr_Z_ahead)
                        or (gate_seq[1] == gates.iSWAP_Cplr_Z_behind)):
                        self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq[1])
                    else:
                        self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq)
                elif generator == 'CZ_Cplr':
                    if gate_seq[1] in [gates.CZ_Cplr, gates.CZ_Cplr_opposite]:
                        self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq[1])
                    else:
                        self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq)
                        
            self.prev_gate_seq = multi_gate_seq
        
            if write_seq == True:
                import os
                from datetime import datetime
                directory = os.path.join(path_currentdir,'2QB_XEBseq')
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = datetime.now().strftime('%Y-%m-%d %H-%M-%S-f')[:-3] + '_N_cycles=%d_rnd_seed=%d.txt'%(N_cycles,rnd_seed)
                filepath = os.path.join(directory,filename)
                log.info('make file: ' + filepath)
                with open(filepath, "w") as text_file:
                    print('New Sequence', file=text_file)
                    if (generator in ['iSWAP_Cplr', 'CZ_Cplr']):
                        for i in range(len(gateSeq1)):
                            print("Index: %d, Gate: ["%(i) + cliffords.Gate_to_strGate(gateSeq1[i]) + ", " + cliffords.Gate_to_strGate(gateSeqCplr[i]) + ", " + cliffords.Gate_to_strGate(gateSeq2[i]) +']', file=text_file)
                    else:
                        for i in range(len(gateSeq1)):
                            print("Index: %d, Gate: ["%(i) + cliffords.Gate_to_strGate(gateSeq1[i]) + ", " + cliffords.Gate_to_strGate(gateSeq2[i]) +']', file=text_file)
        else:
            for gate_seq in self.prev_gate_seq:
                if generator == 'CZ':
                    if gate_seq[0] == gates.CZ:
                        self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq[0])
                    else:
                        self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq)
                elif generator == 'I':
                    self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq)
                elif generator == 'iSWAP_Cplr':
                    if ((gate_seq[1] == gates.iSWAP_Cplr)
                        or (gate_seq[1] == gates.iSWAP_Cplr_Z_ahead)
                        or (gate_seq[2] == gates.iSWAP_Cplr_Z_behind)):
                        self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq[1])
                    else:
                        self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq)
                elif generator == 'CZ_Cplr':
                    if gate_seq[1] in [gates.CZ_Cplr, gates.CZ_Cplr_opposite]:
                        self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq[1])
                    else:
                        self.add_gate(qubit=qubits_to_benchmark, gate=gate_seq)
if __name__ == '__main__':
    pass
