import numpy as np
from numpy import matmul as mul
from numpy.linalg import inv as inv
from numpy.linalg import eig as eig
from numpy import tensordot as tensor
from numpy import dot
import pickle

import itertools
import sequence_rb
import gates
import sequence


if __name__ == "__main__":
    # --------------------------------

    generator = 'iSWAP_Cplr'
    seq_recovery_QB1 = []
    seq_recovery_QB2 = []
    seq_recovery_Cplr = []
    index = 5000
    # seq = sequence.SequenceToWaveforms(n_qubit = 3)
    _config = dict()
    _config['Sequence'] = '2QB-RB'
    _config['Number of Cliffords'] = 2
    _config['Random Seed'] = 2
    _config['Interleave 2-QB Gate'] = False
    _config['Output multiple sequences'] = False
    _config['Write sequence as txt file'] = False
    _config['Find the cheapest recovery Clifford'] = True
    _config['Use a look-up table'] = True
    _config['File path of the look-up table'] = 'recovery_rb_table_iSWAP_Cplr.pickle'
    _config['Native 2-QB gate'] = 'iSWAP_Cplr'
    _config['Qubits to Benchmark (Coupler)'] = '1-2-3'
    # gates.iSWAP_Cplr_Z_ahead.new_angles(30/180*np.pi,30/180*np.pi)
    # gates.iSWAP_Cplr_Z_behind.new_angles(30/180*np.pi,30/180*np.pi)
    # mat_3d = sequence_rb.gate_to_euler_angles([gates.X2p])
    # print(mat_3d)
    seq = sequence_rb.TwoQubit_RB(n_qubit = 3)
    seq.generate_sequence(config = _config)
    seq.sequence = seq
    seq._explode_composite_gates()
    # print(seq.sequence.sequence_list)
    seq._convert_z_to_euler_gates()
    # print(seq.sequence.sequence_list)

            #     mat = np.kron(mat, np.matrix('1,0;0,1'))
            # elif (gate == gates.X2p):
            #     mat = np.kron(mat, 1/np.sqrt(2)*np.matrix('1,-1j;-1j,1'))
            # elif (gate == gates.X2m):
            #     mat = np.kron(mat, 1/np.sqrt(2)*np.matrix('1,1j;1j,1')) 
            # elif (gate == gates.Y2p):
            #     mat = np.kron(mat, 1/np.sqrt(2)*np.matrix('1,-1;1,1'))
            # elif (gate == gates.Y2m):
            #     mat = np.kron(mat, 1/np.sqrt(2)*np.matrix('1,1;-1,1'))
            # elif (gate == gates.Z2p):
            #     mat = np.kron(mat, 1/np.sqrt(2)*np.matrix('1,0;0,1j'))
            # elif (gate == gates.Z2m):
            #     mat = np.kron(mat, 1/np.sqrt(2)*np.matrix('1,0;0,-1j'))
            # elif (gate == gates.Xp):
            #     mat = np.kron(mat, np.matrix('0,-1j;-1j,0'))
            # elif (gate == gates.Xm):
            #     mat = np.kron(mat, np.matrix('0,1j;1j,0'))
            # elif (gate == gates.Yp):
            #     mat = np.kron(mat, np.matrix('0,-1;1,0'))
            # elif (gate == gates.Ym):
            #     mat = np.kron(mat, np.matrix('0,1;-1,0'))
            # elif (gate == gates.Zp):
            #     mat = np.kron(mat, np.matrix('1,0;0,-1'))
            # elif (gate == gates.Zm):
            #     mat = np.kron(mat, np.matrix('1,0;0,-1'))