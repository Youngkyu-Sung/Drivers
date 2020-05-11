

import logging
# Allow logging to Labber's instrument log
log = logging.getLogger('LabberDriver')
import numpy as np
class Crosstalk_Compensation(object):

    def __init__(self, id):
        self.A1 = 0
        self.tau1 = 0
        self.A2 = 0
        self.tau2 = 0
        self.A3 = 0
        self.tau3 = 0
        self.A4 = 0
        self.tau4 = 0
        self.A5 = 0
        self.tau5 = 0
        self.A6 = 0
        self.tau6 = 0
        self.dt = 1
        self.id = "2-3"

    def set_parameters(self, config={}):
        """Set base parameters using config from from Labber driver.

        Parameters
        ----------
        config : dict
            Configuration as defined by Labber driver configuration window

        """
        self.A1 = config.get('Predistort Z{} - A1'.format(self.id))
        self.tau1 = config.get('Predistort Z{} - tau1'.format(self.id))
        self.A2 = config.get('Predistort Z{} - A2'.format(self.id))
        self.tau2 = config.get('Predistort Z{} - tau2'.format(self.id))

        self.A3 = config.get('Predistort Z{} - A3'.format(self.id))
        self.tau3 = config.get('Predistort Z{} - tau3'.format(self.id))
        self.A4 = config.get('Predistort Z{} - A4'.format(self.id))
        self.tau4 = config.get('Predistort Z{} - tau4'.format(self.id))

        self.A5 = config.get('Predistort Z{} - A5'.format(self.id))
        self.tau5 = config.get('Predistort Z{} - tau5'.format(self.id))
        self.A6 = config.get('Predistort Z{} - A6'.format(self.id))
        self.tau6 = config.get('Predistort Z{} - tau6'.format(self.id))
        self.dt = 1 / config.get('Sample rate')

    def compensate(self, waveform):

        """
        # pad with zeros at end to make sure response has time to go to zero

		"""
        pad_time = max(6 * max([self.tau1, self.tau2, self.tau3]),50e-6)
        padded = np.zeros(len(waveform) + round(pad_time / self.dt))
        padded[:len(waveform)] = waveform
        Y = np.fft.rfft(padded, norm='ortho')

        omega = 2 * np.pi * np.fft.rfftfreq(len(padded), self.dt)
        H = (1 +
             (1j * self.A1 * omega * self.tau1) /
             (1j * omega * self.tau1 + 1) +
             (1j * self.A2 * omega * self.tau2) /
             (1j * omega * self.tau2 + 1) +
             (1j * self.A3 * omega * self.tau3) /
             (1j * omega * self.tau3 + 1) +
             (1j * self.A4 * omega * self.tau4) /
             (1j * omega * self.tau4 + 1) +
             (1j * self.A5 * omega * self.tau5) /
             (1j * omega * self.tau5 + 1) +
             (1j * self.A6 * omega * self.tau6) /
             (1j * omega * self.tau6 + 1))
        Yc = H * Y

        yc = np.fft.irfft(Yc, norm='ortho')[:len(waveform)] - waveform
        return yc
