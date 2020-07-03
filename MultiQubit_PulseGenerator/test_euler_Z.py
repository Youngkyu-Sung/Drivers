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
    _config['Number of Cliffords'] = 1
    _config['Random Seed'] = 1
    _config['Interleave 2-QB Gate'] = False
    _config['Output multiple sequences'] = False
    _config['Write sequence as txt file'] = False
    _config['Find the cheapest recovery Clifford'] = True
    _config['Use a look-up table'] = True
    _config['File path of the look-up table'] = 'recovery_rb_table_iSWAP_Cplr.pickle'
    _config['Native 2-QB gate'] = 'iSWAP_Cplr'
    _config['Qubits to Benchmark (Coupler)'] = '1-2-3'
    gates.iSWAP_Cplr_Z_ahead.new_angles(np.pi*0.3,np.pi*0.3)
    gates.iSWAP_Cplr_Z_behind.new_angles(np.pi*0.3,np.pi*0.3)
    seq = sequence_rb.TwoQubit_RB(n_qubit = 3)
    seq.generate_sequence(config = _config)
    seq.sequence = seq
    # seq.generate_sequence(config = _config)
    # print(seq.sequence_list)
    # seq.get_waveforms()
    seq._explode_composite_gates()
    seq._convert_z_to_euler_gates()
    # print(seq)
    # print(seq.sequence_list)
    # sequence_rb.add_twoQ_clifford(index, seq_recovery_QB1, seq_recovery_QB2, gate_seq_Cplr = seq_recovery_Cplr, generator = generator)
    # print(seq_recovery_Cplr)