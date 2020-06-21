#!/usr/bin/env python3

import numpy as np
from scipy.linalg import inv

# Allow logging to Labber's instrument log
import logging
log = logging.getLogger('LabberDriver')

class Microwave_Crosstalk(object):
    """This class is used to compensate crosstalk qubit Z control."""

    def __init__(self):
        # define variables
        self.matrix_path = ''
        # TODO(dan): define variables for matrix, etc

    def set_parameters(self, config={}):
        """Set base parameters using config from from Labber driver.

        Parameters
        ----------
        config : dict
            Configuration as defined by Labber driver configuration window

        """
        # return directly if not in use
        log.info('compensate microwave cross-talk: {}'.format(config.get('Compensate microwave cross-talk')))
        if not config.get('Compensate microwave cross-talk'):
            return
        self.import_crosstalk_matrix(config)

        dTranslate = {'One': int(1), 'Two': int(
            2), 'Three': int(3), 'Four': int(4), 'Five': int(5)}
        nQBs = dTranslate[config.get('Number of qubits')]

        self.Sequence = []
        if nQBs > 0:
            for QB in range(0, nQBs):
                element = config.get('Microwave CT matrix element #%d' % (QB + 1), 'None')
                log.info('QB element: {}'.format(element))
                if element == 'None':
                    continue
                else:
                    self.Sequence.append(dTranslate[element])
                # if self.compensation_matrix.shape[0] < dTranslate[element]:
                #     raise 'Element of Cross-talk matrix is too large for '\
                #         'actual matrix size'

        mat_length = np.max(self.Sequence)
        # log.info('mat_length: {}'.format(mat_length))
        self.compensation_matrix = np.diag(np.ones(mat_length, dtype = complex)) #np.matrix(np.zeros(mat_length, mat_length))
        # log.info('diag self.mat: {}'.format(self.mat))

        for index_r, element_r in enumerate(self.Sequence):
            for index_c, element_c in enumerate(self.Sequence):
                self.compensation_matrix[element_r - 1, element_c - 1] = \
                    self.reduced_c_mat[index_r, index_c]
        # log.info('self.mat: {}'.format(self.mat))
    def import_crosstalk_matrix(self, config):
        """Import crosstalk matrix data.

        Parameters
        ----------
        config : dict
            Configuration as defined by Labber driver configuration window

        """
        # store new path
        C11 = config.get('Microwave CT compensation matrix, C11 mag') * np.exp(1j*config.get('Microwave CT compensation matrix, C11 angle')/180.0*np.pi )
        C12 = config.get('Microwave CT compensation matrix, C12 mag') * np.exp(1j*config.get('Microwave CT compensation matrix, C12 angle')/180.0*np.pi )
        C21 = config.get('Microwave CT compensation matrix, C21 mag') * np.exp(1j*config.get('Microwave CT compensation matrix, C21 angle')/180.0*np.pi )
        C22 = config.get('Microwave CT compensation matrix, C22 mag') * np.exp(1j*config.get('Microwave CT compensation matrix, C22 angle')/180.0*np.pi )

        self.reduced_c_mat = np.matrix([[C11,C12],[C21,C22]])
        # log.info('microwave crosstalk compensation matrix: {}'.format(self.c_mat))
    def compensate(self, waveforms):
        """Compensate crosstalk on Z-control waveforms.

        Parameters
        ----------
        waveforms : list on 1D numpy arrays
            Input data to apply crosstalk compensation on

        Returns
        -------
        waveforms : list of 1D numpy arrays
            Waveforms with crosstalk compensation

        """
        # mat_voltage_vs_phi0 = inv(self.phi0_vs_voltage)

        wavform_length = len(waveforms[0])
        wavform_num = len(self.Sequence)
        wav_array = np.array(np.zeros((wavform_num, wavform_length)))
        wav_toCorrect = []
        for index, waveform in enumerate(waveforms):
            if index + 1 in self.Sequence:
                wav_array[index] = waveform
                wav_toCorrect.append(index)

        # new_array = np.dot(mat_voltage_vs_phi0, wav_array)

        # dot product between the matrix and the waveforms at each timestep
        new_array = np.einsum('ij,jk->ik', self.compensation_matrix , wav_array)

        for Corr_index, index in zip(wav_toCorrect,
                                     range(0, len(self.Sequence))):
            waveforms[Corr_index] = new_array[index]

        return waveforms


if __name__ == '__main__':
    pass
