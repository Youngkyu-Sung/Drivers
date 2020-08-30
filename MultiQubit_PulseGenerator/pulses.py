#!/usr/bin/env python3
import numpy as np
import logging
import copy
log = logging.getLogger('LabberDriver')


import os
path_currentdir  = os.path.dirname(os.path.realpath(__file__)) # curret directory

# TODO Private methods and variables


class Pulse:
    """Represents physical pulses played by an AWG.

    Parameters
    ----------
    complex : bool
        If True, pulse has both I and Q, otherwise it's real valued.
        Phase, frequency and drag only applies for complex waveforms.

    Attributes
    ----------
    amplitude : float
        Pulse amplitude.
    width : float
        Pulse width.
    plateau : float
        Pulse plateau.
    frequency : float
        SSB frequency.
    phase : float
        Pulse phase.
    use_drag : bool
        If True, applies DRAG correction.
    drag_coefficient : float
        Drag coefficient.
    drag_detuning : float
        Applies a frequnecy detuning for DRAG pulses.
    start_at_zero : bool
        If True, forces the pulse to start in 0.

    """

    def __init__(self, complex):

        # set variables
        self.amplitude = 0.5
        self.amplitude_opposite = -0.5
        self.amplitude_half = 0.25 # This is to define pi/2 pulses. by default it is half of the amplitude
        self.width = 10E-9
        self.width_opposite = 0
        self.plateau = 0.0
        self.plateau_opposite = 0.0
        self.frequency = 0.0
        self.phase = 0.0
        self.use_drag = False
        self.drag_coefficient = 0.0
        self.drag_detuning = 0.0
        self.start_at_zero = False
        self.complex = complex

        self.dfdV = 500E6
        self.qubit = None
        self.negative_amplitude = False
        self.init_freq = 6e9
        self.final_freq = 4e9

        self.transduce_pulse = False
        self.transduce_func = lambda x: x


    def total_duration(self):
        """Get the total duration for the pulse.

        Returns
        -------
        float
            Total duration in seconds.

        """
        raise NotImplementedError()

    def calculate_envelope(self, t0, t):
        """Calculate pulse envelope.

        Parameters
        ----------
        t0 : float
            Pulse position, referenced to center of pulse.

        t : numpy array
            Array with time values for which to calculate the pulse envelope.

        Returns
        -------
        waveform : numpy array
            Array containing pulse envelope.

        """
        raise NotImplementedError()

    def calculate_waveform(self, t0, t):
        """Calculate pulse waveform including phase shifts and SSB-mixing.

        Parameters
        ----------
        t0 : float
            Pulse position, referenced to center of pulse.

        t : numpy array
            Array with time values for which to calculate the pulse waveform.

        Returns
        -------
        waveform : numpy array
            Array containing pulse waveform.

        """
        y = self.calculate_envelope(t0, t)
        # Make sure the waveform is zero outside the pulse
        y[t < (t0 - self.total_duration() / 2)] = 0
        y[t > (t0 + self.total_duration() / 2)] = 0

        if self.use_drag and self.complex:
            beta = self.drag_coefficient / (t[1] - t[0])
            y = y + 1j * beta * np.gradient(y)
            y = y * np.exp(1j * 2 * np.pi * self.drag_detuning *
                           (t - t0 + self.total_duration() / 2))
        if self.complex:
            # Apply phase and SSB
            phase = self.phase
            # single-sideband mixing, get frequency
            omega = 2 * np.pi * self.frequency
            # apply SSBM transform
            data_i = (y.real * np.cos(omega * t - phase) +
                      -y.imag * np.cos(omega * t - phase + +np.pi / 2))
            data_q = (y.real * np.sin(omega * t - phase) +
                      -y.imag * np.sin(omega * t - phase + +np.pi / 2))
            y = data_i + 1j * data_q

        return y


class Gaussian(Pulse):
    def __init__(self, complex):
        super().__init__(complex)
        self.truncation_range = 5

    def total_duration(self):
        return self.truncation_range * self.width + self.plateau

    def calculate_envelope(self, t0, t):
        # width is two t std
        # std = self.width/2;
        # alternate; std is set to give total pulse area same as a square
        std = self.width / np.sqrt(2 * np.pi)
        values = np.zeros_like(t)
        if self.plateau == 0:
            # pure gaussian, no plateau
            if std > 0:
                values = np.exp(-(t - t0)**2 / (2 * std**2))
        else:
            # add plateau
            values = np.array(
                ((t >= (t0 - self.plateau / 2)) & (t <
                                                   (t0 + self.plateau / 2))),
                dtype=float)
            if std > 0:
                # before plateau
                values += ((t < (t0 - self.plateau / 2)) * np.exp(
                    -(t - (t0 - self.plateau / 2))**2 / (2 * std**2)))
                # after plateau
                values += ((t >= (t0 + self.plateau / 2)) * np.exp(
                    -(t - (t0 + self.plateau / 2))**2 / (2 * std**2)))

        # TODO  Fix this
        if self.start_at_zero:
            values = values - values.min()
            values = values / values.max()
        values = values * self.amplitude


        if self.qubit is None:
            pass
        else:
            final_V = 1
            init_V = 0
            values = (values-init_V)/(final_V-init_V)*(self.final_freq-self.init_freq) + self.init_freq
            values = self.qubit.f_to_V(values)
            
        if self.negative_amplitude is True:
            values = -values


        return values


class Ramp(Pulse):
    def total_duration(self):
        return 2 * self.width + self.plateau

    def calculate_envelope(self, t0, t):
        # rising and falling slopes
        vRise = ((t - (t0 - self.plateau / 2 - self.width)) / self.width)
        vRise[vRise < 0.0] = 0.0
        vRise[vRise > 1.0] = 1.0
        vFall = (((t0 + self.plateau / 2 + self.width) - t) / self.width)
        vFall[vFall < 0.0] = 0.0
        vFall[vFall > 1.0] = 1.0
        values = vRise * vFall


        if self.qubit is None:
            values = values * self.amplitude
        else:
            final_V = 1
            init_V = 0
            values = (values-init_V)/(final_V-init_V)*(self.final_freq-self.init_freq) + self.init_freq
            values = self.qubit.f_to_V(values)
            
        if self.negative_amplitude is True:
            values = -values

        return values


class Square(Pulse):
    def total_duration(self):
        return self.width + self.plateau

    def calculate_envelope(self, t0, t):
        # reduce risk of rounding errors by putting checks between samples
        if len(t) > 1:
            t0 += (t[1] - t[0]) / 2.0

        values = ((t >= (t0 - (self.width + self.plateau) / 2)) &
                  (t < (t0 + (self.width + self.plateau) / 2)))


        if self.qubit is None:
            values = values * self.amplitude
        else:
            final_V = 1
            init_V = 0
            values = (values-init_V)/(final_V-init_V)*(self.final_freq-self.init_freq) + self.init_freq
            values = self.qubit.f_to_V(values)

        if self.negative_amplitude is True:
            values = -values

        return values

class Half_Cosine(Pulse):
    def total_duration(self):
        return self.width + self.plateau

    def calculate_envelope(self, t0, t):
        tau = self.width
        if self.plateau == 0:
            values = (1 *
                      (np.sin(np.pi * (t - t0 + tau / 2) / tau)))

        else:
            values = np.ones_like(t)
            values[t < t0 - self.plateau / 2] = 1 * \
                (np.sin(np.pi *
                            (t[t < t0 - self.plateau / 2] - t0 +
                             self.plateau / 2 + tau / 2) / tau))
            values[t > t0 + self.plateau / 2] = 1 * \
                (np.sin(np.pi *
                            (t[t > t0 + self.plateau / 2] - t0 -
                             self.plateau / 2 + tau / 2) / tau))
        if self.qubit is None:
            values = values * self.amplitude
        else:
            final_V = 1
            init_V = 0
            values = (values-init_V)/(final_V-init_V)*(self.final_freq-self.init_freq) + self.init_freq
            values = self.qubit.f_to_V(values)
            
        if self.negative_amplitude is True:
            values = -values

        return values

class Cosine(Pulse):
    def total_duration(self):
        return self.width + self.plateau

    def calculate_envelope(self, t0, t):
        # log.info('t0: {}'.format(t0))
        tau = self.width
        if self.plateau == 0:
            values = (1 / 2 *
                      (1 - np.cos(2 * np.pi * (t - t0 + tau / 2) / tau)))
        else:
            values = np.ones_like(t)
            values[t < t0 - self.plateau / 2] = 1 / 2 * \
                (1 - np.cos(2 * np.pi *
                            (t[t < t0 - self.plateau / 2] - t0 +
                             self.plateau / 2 + tau / 2) / tau))
            values[t > t0 + self.plateau / 2] = 1 / 2 * \
                (1 - np.cos(2 * np.pi *
                            (t[t > t0 + self.plateau / 2] - t0 -
                             self.plateau / 2 + tau / 2) / tau))


        if self.qubit is None:
            values = values * self.amplitude
        else:
            final_V = 1
            init_V = 0
            values = (values-init_V)/(final_V-init_V)*(self.final_freq-self.init_freq) + self.init_freq
            values = self.qubit.f_to_V(values)
            
        if self.negative_amplitude is True:
            values = -values


        return values

class Optimal(Pulse):
    # For optimal coupler pulse
    def __init__(self, *args, **kwargs):
        super().__init__(False)
        self.F_terms = 2
        self.Hx = 0.07e9 # (Hz)
        self.init_Hz = 0.725e9 # (Hz)
        self.final_Hz = 0.1e9 # (Hz)
        self.Hz_offset = 4.164e9 # (Hz)
        self.F_coeffs = np.array([1.0866, -0.0866])
        self.t_tau = None

    def total_duration(self):
        return self.width+self.plateau


    def calculate_envelope(self, t0, t):
        if self.t_tau is None:
            self.calculate_optimal_waveform()

        # Plateau is added as an extra extension of theta_f.
        theta_t = np.ones(len(t)) * self.theta_i
        for i in range(len(t)):
            if 0 < (t[i] - t0 + self.plateau / 2) < self.plateau:
                theta_t[i] = self.theta_f
            elif (0 < (t[i] - t0 + self.width / 2 + self.plateau / 2) <
                  (self.width + self.plateau) / 2):
                theta_t[i] = np.interp(
                    t[i] - t0 + self.width / 2 + self.plateau / 2, self.t_tau,
                    self.theta_tau)

            elif (0 < (t[i] - t0 + self.width / 2 + self.plateau / 2) <
                  (self.width + self.plateau)):
                theta_t[i] = np.interp(
                    t[i] - t0 + self.width / 2 - self.plateau / 2, self.t_tau,
                    self.theta_tau)

        df = 2*self.Hx * (1 / np.tan(theta_t) - 1 / np.tan(self.theta_i))

        if self.qubit is None:
            # Use linear dependence if no qubit was given
            # log.info('---> df (linear): ' +str(df))
            values = df / self.dfdV
            # values = theta_t
        else:
            values = self.qubit.df_to_dV(df)

        if self.negative_amplitude is True:
            values = -values

        return values

    def calculate_optimal_waveform(self):
        # calculate initial & final angles.
        self.theta_i = np.arctan(self.Hx / self.init_Hz)
        if self.final_Hz == 0:
            self.theta_f = np.pi *0.5
        else:
            self.theta_f = np.arctan(self.Hx / self.final_Hz)  

        if self.theta_f < 0:
            self.theta_f = self.theta_f + np.pi
                
        # # Renormalize fourier coefficients to initial and final angles
        normalize_const = (self.theta_f - self.theta_i) / 2 
        sum_even_coeffs = np.sum(self.F_coeffs[range(0, self.F_terms, 2)])
        for i in range(len(self.F_coeffs)):
            self.F_coeffs[i] = self.F_coeffs[i] * (normalize_const / sum_even_coeffs)
            
        # defining helper variables
        n = np.arange(1, self.F_terms + 1, 1)
        n_points = 1000  # Number of points in the numerical integration

        # Calculate pulse width in tau variable - See paper for details
        tau = np.linspace(0, 1, n_points)
        self.theta_tau = np.zeros(n_points)
        # This corresponds to the sum in Eq. (15) in Martinis & Geller
        for i in range(n_points):
            self.theta_tau[i] = (
                np.sum(self.F_coeffs * (1 - np.cos(2 * np.pi * n * tau[i]))) +
                self.theta_i)

        _t_tau = np.trapz(np.sin(self.theta_tau), x=tau)

        # Find the width in units of tau:
        Width_tau = self.width / _t_tau

        # Calculating time as functions of tau
        # we normalize to width_tau (calculated above)
        tau = np.linspace(0, Width_tau, n_points)
        self.t_tau = np.zeros(n_points)

        for i in range(n_points):
            if i > 0:
                self.t_tau[i] = np.trapz(
                    np.sin(self.theta_tau[0:i+1]), x=tau[0:i+1])


class CZ(Pulse):
    def __init__(self, *args, **kwargs):
        super().__init__(False)
        # For CZ pulses
        self.F_Terms = 1
        self.Coupling = 20E6
        self.Offset = 300E6
        self.Lcoeff = np.array([0.3])
        # self.dfdV = 500E6
        # self.qubit = None
        # self.negative_amplitude = False

        self.t_tau = None

    def total_duration(self):
        return self.width+self.plateau

    def calculate_envelope(self, t0, t):
        if self.t_tau is None:
            self.calculate_cz_waveform()

        # Plateau is added as an extra extension of theta_f.
        theta_t = np.ones(len(t)) * self.theta_i
        for i in range(len(t)):
            if 0 < (t[i] - t0 + self.plateau / 2) < self.plateau:
                theta_t[i] = self.theta_f
            elif (0 < (t[i] - t0 + self.width / 2 + self.plateau / 2) <
                  (self.width + self.plateau) / 2):
                theta_t[i] = np.interp(
                    t[i] - t0 + self.width / 2 + self.plateau / 2, self.t_tau,
                    self.theta_tau)

            elif (0 < (t[i] - t0 + self.width / 2 + self.plateau / 2) <
                  (self.width + self.plateau)):
                theta_t[i] = np.interp(
                    t[i] - t0 + self.width / 2 - self.plateau / 2, self.t_tau,
                    self.theta_tau)
        # Clip theta_t to remove numerical outliers:
        theta_t = np.clip(theta_t, self.theta_i, None)

        # clip theta_f to remove numerical outliers
        theta_t = np.clip(theta_t, self.theta_i, None)
        df = 2*self.Coupling * (1 / np.tan(theta_t) - 1 / np.tan(self.theta_i))
        # log.info('self.qubit: ' +str(self.qubit))
        if self.qubit is None:
            # Use linear dependence if no qubit was given
            # log.info('---> df (linear): ' +str(df))
            values = df / self.dfdV
            # values = theta_t
        else:
            values = self.qubit.df_to_dV(df)
        if self.negative_amplitude is True:
            values = -values

        return values

    def calculate_cz_waveform(self):
        """Calculate waveform for c-phase and store in object"""
        # notation and calculations are based on
        # "Fast adiabatic qubit gates using only sigma_z control"
        # PRA 90, 022307 (2014)
        # Initial and final angles on the |11>-|02> bloch sphere
        self.theta_i = np.arctan(2*self.Coupling / self.Offset)
        self.theta_f = np.arctan(2*self.Coupling / self.amplitude)
        # log.log(msg="calc", level=30)

        # Renormalize fourier coefficients to initial and final angles
        # Consistent with both Martinis & Geller and DiCarlo 1903.02492
        Lcoeff = self.Lcoeff
        Lcoeff[0] = (((self.theta_f - self.theta_i) / 2)
                     - np.sum(self.Lcoeff[range(2, self.F_Terms, 2)]))

        # defining helper variabels
        n = np.arange(1, self.F_Terms + 1, 1)
        n_points = 1000  # Number of points in the numerical integration

        # Calculate pulse width in tau variable - See paper for details
        tau = np.linspace(0, 1, n_points)
        self.theta_tau = np.zeros(n_points)
        # This corresponds to the sum in Eq. (15) in Martinis & Geller
        for i in range(n_points):
            self.theta_tau[i] = (
                np.sum(Lcoeff * (1 - np.cos(2 * np.pi * n * tau[i]))) +
                self.theta_i)
        # Now calculate t_tau according to Eq. (20)
        t_tau = np.trapz(np.sin(self.theta_tau), x=tau)
        # log.info('t tau: ' + str(t_tau))
        # t_tau = np.sum(np.sin(self.theta_tau))*(tau[1] - tau[0])
        # Find the width in units of tau:
        Width_tau = self.width / t_tau

        # Calculating time as functions of tau
        # we normalize to width_tau (calculated above)
        tau = np.linspace(0, Width_tau, n_points)
        self.t_tau = np.zeros(n_points)
        self.t_tau2 = np.zeros(n_points)
        for i in range(n_points):
            if i > 0:
                self.t_tau[i] = np.trapz(
                    np.sin(self.theta_tau[0:i+1]), x=tau[0:i+1])
                # self.t_tau[i] = np.sum(np.sin(self.theta_tau[0:i+1]))*(tau[1]-tau[0])

class NetZero(CZ):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.slepian = None

    def total_duration(self):
        return 2*self.slepian.total_duration()

    def calculate_cz_waveform(self):
        self.slepian = CZ()
        self.slepian.__dict__ = copy.copy(self.__dict__)
        self.slepian.width /= 2
        self.slepian.plateau /= 2
        self.slepian.calculate_cz_waveform()

    def calculate_envelope(self, t0, t):
        return (self.slepian.calculate_envelope(t0-self.total_duration()/4, t) -
                self.slepian.calculate_envelope(t0+self.total_duration()/4, t))

class CompositePulse():
    def __init__(self, *args, **kwargs):
        self.list_pulses = kwargs.get('list_pulses', []) # list of Pulse() objects 
        self.list_delays = kwargs.get('list_delays', [])

    def total_duration(self, NetZero = False):
        _duration = 0
        for i in range(len(self.list_pulses)):
            _pulse = self.list_pulses[i]
            if (NetZero == True):
                #replace the pulse by the opposite-polarity pulse
                prev_amplitude = _pulse.amplitude
                prev_width = _pulse.width
                prev_plateau = _pulse.plateau
                _pulse.amplitude = _pulse.amplitude_opposite
                _pulse.width = _pulse.width_opposite
                _pulse.plateau = _pulse.plateau_opposite

            #skip zero-duration pulse
            if _pulse.total_duration == 0:
                pass
            else:
                if i > 0:
                    _duration = max(_duration, _pulse.total_duration() + self.list_delays[i-1] )
                else:
                    _duration = max(_duration, _pulse.total_duration())
            if (NetZero == True):
                #return the pulse back to its original pulse
                _pulse.amplitude = prev_amplitude
                _pulse.width = prev_width
                _pulse.plateau = prev_plateau
        return _duration


    def calculate_waveform(self, t0, t, NetZero = False):#, _recursive = False):
        _waveform = 0

        for i in range(len(self.list_pulses)):
            _pulse = self.list_pulses[i]
            if (NetZero == True):
                prev_amplitude = _pulse.amplitude
                prev_width = _pulse.width
                prev_plateau = _pulse.plateau
                _pulse.amplitude = _pulse.amplitude_opposite
                _pulse.width = _pulse.width_opposite
                _pulse.plateau = _pulse.plateau_opposite

            if i > 0:
                t_center = t0 + self.list_delays[i-1] + (_pulse.total_duration() - self.total_duration(NetZero = NetZero))*0.5
            else:
                t_center = t0 + (_pulse.total_duration() - self.total_duration(NetZero = NetZero))*0.5

            if _pulse.total_duration() > 0:
                y = _pulse.calculate_envelope(t_center, t)
                phase = _pulse.phase
                omega = 2 * np.pi * _pulse.frequency
                if _pulse.frequency > 0:
                    y = y*np.cos(omega * t + phase/180.0*np.pi)

                # Make sure the waveform is zero outside the pulse
                y[t < (t_center - _pulse.total_duration() / 2)] = 0
                y[t > (t_center + _pulse.total_duration() / 2)] = 0

                if _pulse.transduce_pulse:
                    y = _pulse.transduce_func(y)

                if (NetZero == True):
                    _pulse.amplitude = prev_amplitude
                    _pulse.width = prev_width
                    _pulse.plateau = prev_plateau
                _waveform += y
        return _waveform

class NetZero_CompositePulse():
    def __init__(self, composite_pulse, *args, **kwargs):
        # super().__init__()
        self.composite_pulse = composite_pulse
        self.net_zero_spacing = kwargs.get('net_zero_spacing', 0)
        # self.composite_pulse.width /= 2
        # self.composite_pulse.plateau /= 2

    def total_duration(self):
        return self.composite_pulse.total_duration() + self.composite_pulse.total_duration(NetZero = True) + self.net_zero_spacing

    def calculate_waveform(self, t0, t):
        return (self.composite_pulse.calculate_waveform(t0 - self.composite_pulse.total_duration(NetZero = True)*0.5 - self.net_zero_spacing *0.5, t, NetZero = False) 
                +
                self.composite_pulse.calculate_waveform(t0 + self.composite_pulse.total_duration(NetZero = False)*0.5 + self.net_zero_spacing * 0.5, t, NetZero = True)
                )
if __name__ == '__main__':
    pass
