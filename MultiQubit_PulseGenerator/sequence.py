#!/usr/bin/env python3
import logging
import numpy as np
import copy

import crosstalk
import microwave_crosstalk # to compensate microwave crosstalk (YS)
import gates
import predistortion
import pulses
import qubits
import readout
import tomography
import crosstalk_compensation

import dill as pickle
import os
path_currentdir  = os.path.dirname(os.path.realpath(__file__)) # curret directory


# Allow logging to Labber's instrument log
log = logging.getLogger('LabberDriver')

# TODO Reduce calc of CZ by finding all unique TwoQubitGates in seq and calc.
# TODO Make I(width=None) have the width of the longest gate in the step
# TODO Add checks so that not both t0 and dt are given
# TODO Two composite gates should be able to be parallell
# TODO check number of qubits in seq and in gate added to seq
# TODO Remove pulse from I gates


import math

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def mat_to_euler_angles_xyx(matrix):
    # find euler angles (x-y-x) give 3d rotation matrix
    threshold = 1e-8
    theta = np.arccos(np.clip(matrix[0,0],-1,1))
    if np.abs(np.sin(theta)) < threshold: 
        # is theta np.pi or zero?
        phi = 0
        psi = np.arctan2(matrix[2,1],matrix[1,1])
        # if np.abs(theta) < threshold:
        #     # if theta = 0, one X gate is enough
        #     phi = 0
        #     # with np.errstate(divide='ignore'):  # ignore runtimewarning (divide by zero)
        #     #     psi = np.arctan(matrix[2,1]/matrix[1,1])
        #     psi = np.arctan2(matrix[2,1],matrix[1,1])

        #     # if np.abs(np.sin(psi)) < threshold:
        #     #     # distinguish psi np.pi vs zero.
        #     #     psi = np.arccos(np.clip(matrix[1,1],-1,1))
        # else:
        #     # if theta = pi
        #     phi = 0
        #     psi = np.arccos(np.clip(matrix[1,1],-1,1))

    elif np.abs(np.cos(theta)) < threshold:
        # log.info("HELLO!")
        # is theta +np.pi/2 or -np.pi/2?
        # calcualte phi, psi
        phi = np.arctan2(matrix[0,1], matrix[0,2])
        psi = np.arctan2(matrix[1,0], -matrix[2,0])

        if ((np.abs(matrix[1,0])>threshold) and (np.abs(np.sin(psi)) > 0)):
            theta = np.arcsin(np.clip(matrix[1,0]/np.sin(psi),-1,1))
        elif ((np.abs(matrix[2,0])>threshold) and (np.abs(np.cos(psi)) > 0)):
            theta = np.arcsin(-np.clip(matrix[2,0]/np.cos(psi),-1,1))
        else:
            raise ValueError('Singular Matrix, matrix: {}'.format(matrix))
    else:
        #is theta positive / negative?

        phi = np.arctan2(matrix[0,1], matrix[0,2])
        psi = np.arctan2(matrix[1,0], -matrix[2,0])
        # with np.errstate(divide='ignore'):  # ignore runtimewarning (divide by zero)
        #     phi = np.arctan(matrix[0,1]/matrix[0,2])
        #     psi = np.arctan(-matrix[1,0]/matrix[2,0])

        sign_theta = None
        if np.abs(np.cos(phi)) > threshold:
            sign_theta = np.sign(matrix[0,2]/np.cos(phi))
        # elif np.abs(np.sin(phi))> threshold:
        #     sign_theta = np.sign(matrix[0,1]/np.sin(phi))
        # print('sign from phi: {}'.format(sign_theta))
        if np.abs(np.cos(psi)) > threshold:
            sign_theta = np.sign(-matrix[2,0]/np.cos(psi))

        if sign_theta is None:
            pass
        elif sign_theta > 0:
            theta = np.abs(theta)
        else:
            theta = -np.abs(theta)

        # if (np.abs(matrix[0,2]) < threshold):
        #     phi = np.arcsin(np.clip(matrix[0,1]/np.sin(theta),-1,1))

        # if (np.abs(matrix[2,0]) < threshold):
        #     psi = np.arcsin(np.clip(matrix[1,0]/np.sin(theta),-1,1))

    # print('phi(X): {}, theta(Y): {}, psi(X): {}'.format(phi/np.pi*180, theta/np.pi*180, psi/np.pi*180))
    return (phi, theta, psi)

def mat_to_euler_angles_yxy(matrix):
    # find euler angles (y-x-y) give 3d rotation matrix
    theta = np.arccos(np.clip(matrix[1,1],-1,1))
    if theta == 0: # only X gate is enough
        phi = np.arccos(np.clip(matrix[1,1],-1,1))
        psi = 0
    else:
        with np.errstate(divide='ignore'): # ignore runtimewarning (divide by zero)
            phi = np.arctan(-matrix[1,0]/matrix[1,2])
            psi = np.arctan(matrix[0,1]/matrix[2,1])
    # print('phi(Y): {}, theta(X): {}, psi(Y): {}'.format(phi/np.pi*180, theta/np.pi*180, psi/np.pi*180))
    return (phi, theta, psi)

def gate_to_euler_gates(list_1qb_gates, convention = 'x-y-x'):
    threshold = 1e-10
    phi, theta, psi = gate_to_euler_angles(list_1qb_gates, convention)
    # phi, theta, psi = 0,0,0
    if convention == 'x-y-x':
        # first X-gate
        if np.abs(phi) > threshold:
            X1 = gates.SingleQubitXYRotation(phi=0, theta=phi, name='X={:+.2f}'.format(phi))
        else:
            X1 = gates.IdentityGate(width=None)

        # second Y-gate
        if np.abs(theta) > threshold:
            Y2 = gates.SingleQubitXYRotation(phi=np.pi*0.5, theta=theta, name='Y={:+.2f}'.format(theta))
        else:
            Y2 = gates.IdentityGate(width=None)

        # third X-gate
        if np.abs(psi) > threshold:
            X3 = gates.SingleQubitXYRotation(phi=0, theta=psi, name='X={:+.2f}'.format(psi))
        else:
            X3 = gates.IdentityGate(width=None)
        return [X1, Y2, X3]

    elif convention == 'y-x-y':
        # first Y-gate
        if np.abs(phi) > threshold:
            Y1 = gates.SingleQubitXYRotation(phi=np.pi*0.5, theta=phi, name='Y={:+.2f}'.format(phi))
        else:
            Y1 = gates.IdentityGate(width=None)

        # second X-gate
        if np.abs(theta) > threshold:
            X2 = gates.SingleQubitXYRotation(phi=0, theta=theta, name='X={:+.2f}'.format(theta))
        else:
            X2 = gates.IdentityGate(width=None)

        # third Y-gate
        if np.abs(psi) > threshold:
            Y3 = gates.SingleQubitXYRotation(phi=np.pi*0.5, theta=psi, name='Y={:+.2f}'.format(psi))
        else:
            Y3 = gates.IdentityGate(width=None)
        return [Y1, X2, Y3]

def gate_to_euler_angles(list_1qb_gates, convention = 'x-y-x'):
    # represent single qubit gate in a three-dimensional (3x3) rotation matrix in (x,y,z) basis

    mat_3d = np.identity(3)
    for gate in list_1qb_gates:
        if (gate == gates.I):
            mat_3d = np.matmul(np.identity(3), mat_3d) 
        elif (gate == gates.X2p):
            mat_3d = np.matmul(rotation_matrix([1,0,0], np.pi*0.5), mat_3d)
        elif (gate == gates.X2m):
            mat_3d = np.matmul(rotation_matrix([1,0,0], -np.pi*0.5), mat_3d)
        elif (gate == gates.Y2p):
            mat_3d = np.matmul(rotation_matrix([0,1,0], np.pi*0.5), mat_3d)
        elif (gate == gates.Y2m):
            mat_3d = np.matmul(rotation_matrix([0,1,0], -np.pi*0.5), mat_3d)
        elif (gate == gates.Z2p):
            mat_3d = np.matmul(rotation_matrix([0,0,1], +np.pi*0.5), mat_3d)
        elif (gate == gates.Z2m):
            mat_3d = np.matmul(rotation_matrix([0,0,1], -np.pi*0.5), mat_3d)
        elif (gate == gates.Xp):
            mat_3d = np.matmul(rotation_matrix([1,0,0], np.pi), mat_3d)
        elif (gate == gates.Xm):
            mat_3d = np.matmul(rotation_matrix([1,0,0], -np.pi), mat_3d)
        elif (gate == gates.Yp):
            mat_3d = np.matmul(rotation_matrix([0,1,0], np.pi), mat_3d)
        elif (gate == gates.Ym):
            mat_3d = np.matmul(rotation_matrix([0,1,0], -np.pi), mat_3d)
        elif (gate == gates.Zp):
            mat_3d = np.matmul(rotation_matrix([0,0,1], np.pi), mat_3d)
        elif (gate == gates.Zm):
            mat_3d = np.matmul(rotation_matrix([0,0,1], -np.pi), mat_3d)
        elif isinstance(gate, gates.EulerZGate):
            mat_3d = np.matmul(rotation_matrix([0,0,1], gate.theta), mat_3d)
        elif isinstance(gate, gates.IdentityGate):
            mat_3d = np.matmul(np.identity(3), mat_3d)
        else:
            raise ValueError('non-identified gate: ' + str(gate))
    # print('mat_3d_before: {}'.format(mat_3d))
    if convention == 'x-y-x':
        phi, theta, psi = mat_to_euler_angles_xyx(mat_3d)
        mat_3d_after = np.identity(3)
        mat_3d_after = np.matmul(rotation_matrix([1,0,0], phi), mat_3d_after)
        mat_3d_after = np.matmul(rotation_matrix([0,1,0], theta), mat_3d_after)
        mat_3d_after = np.matmul(rotation_matrix([1,0,0], psi), mat_3d_after)
        log.info('mat_3d_before: {}'.format(mat_3d))
        log.info('mat_3d_after: {}'.format(mat_3d_after))
        if not np.allclose(mat_3d, mat_3d_after, atol = 1e-5):
            raise ValueError('Wrong Euler Angles! list_1qb_gates: {}, \n mat_before: {}, \n mat_after: {} \n, phi(X):{}, theta(Y):{}, psi(X):{}'.format(list_1qb_gates, mat_3d, mat_3d_after, phi, theta, psi))
    elif convention == 'y-x-y':
        phi, theta, psi = mat_to_euler_angles_yxy(mat_3d)

    return phi, theta, psi 

class GateOnQubit:
    def __init__(self, gate, qubit, pulse=None):
        self.gate = gate
        self.qubit = qubit
        self.pulse = pulse
        if pulse is None:
            self.duration = 0
        else:
            self.duration = pulse.total_duration()

    def __str__(self):
        return "Gate {} on qubit {}".format(self.gate, self.qubit)

    def __repr__(self):
        return self.__str__()


class Step:
    """Represent one step in a sequence.

    Parameters
    ----------
    n_qubit : int
        Number of qubits in the sequece.
    t0 : float
        Center of the sequence in seconds (the default is None).
    dt : float
        Spacing to previous pulse in seconds (the default is None). Use only
        either t0 or dt.
    align : str {'left', 'center', 'right'}
        The alignment of pulses if they have different lengths,
        (the default is 'center').

    Attributes
    ----------
    gates : list of :dict:
        The different gates in the step.

    """

    def __init__(self, t0=None, dt=None, align='center'):
        self.gates = []
        self.align = align
        self.t0 = t0
        self.dt = dt
        self.t_start = None
        self.t_end = None

    def add_gate(self, qubit, gate):
        """Add the given gate to the specified qubit(s).

        The number of gates must equal the number of qubits.

        If the number of qubits given are less than the number of qubits in the
        step, I gates are added to the other qubits.
        Parameters
        ----------
        qubit : int or list of int
            The qubit indices.
        gate : :obj:`BaseGate`
            The gate(s).

        """
        if gate.number_of_qubits() > 1 and not isinstance(qubit, list):
            raise ValueError(
                "Provide a list of qubits for gates with more than one qubit")

        if gate.number_of_qubits() > 1 and not gate.number_of_qubits() == len(
                qubit):
            raise ValueError(
                """Number of qubits in the gate must equal the number of qubit
                indices given. gate.number_of_qubits() = %d, len(qubit) = %d"""%(gate.number_of_qubits(),len(qubit)))

        if gate.number_of_qubits() == 1 and not isinstance(qubit, int):
            raise ValueError("Provide qubit as int for gates with one qubit")

        if isinstance(qubit, int):
            if self._qubit_in_step(qubit):
                raise ValueError("Qubit {} already in step.".format(qubit))
        else:
            for n in qubit:
                if self._qubit_in_step(n):
                    raise ValueError("Qubit {} already in step.".format(n))

        self.gates.append(GateOnQubit(gate, qubit))

    def time_shift(self, shift):
        """Shift the timings of the step.

        Parameters
        ----------
        shift : float
            The amount of shift to apply in seconds.

        """
        self.t_start += shift
        self.t0 += shift
        self.t_end += shift

    def _qubit_in_step(self, qubit):
        """Returns whatever the given qubit is in the step or not. """
        if not isinstance(qubit, int):
            raise ValueError("Qubit index should be int.")

        def _in(input_list, n):
            flat_list = []
            for sublist_or_el in input_list:
                if isinstance(sublist_or_el, list):
                    if _in(sublist_or_el, n) is True:
                        return True
                elif sublist_or_el == n:
                    return True
            return False

        return _in([x.qubit for x in self.gates], qubit)

    def __str__(self):
        return str(self.gates)

    def __repr__(self):
        return str(self.gates)


class Sequence:
    """A multi qubit seqence.

    Parameters
    ----------
    n_qubit : type
        The number of qubits in the sequence.

    Attributes
    ----------
    sequences : list of :obj:`Step`
        Holds the steps of the sequence.
    perform_process_tomography : bool
        Flag for performing process tomography.
    perform_state_tomography : bool
        Flag for performing state tomography.
    readout_delay : float
        Delay time between last pulse and readout, in seconds.
    n_qubit

    """

    def __init__(self, n_qubit):
        self.n_qubit = n_qubit

        # log.info('initiating empty seqence list')
        self.sequence_list = []

        # process tomography
        self.perform_process_tomography = False
        self._process_tomography = tomography.ProcessTomography()

        # state tomography
        self.perform_state_tomography = False
        self._state_tomography = tomography.StateTomography()

        # readout
        self.readout_delay = 0.0

    # Public methods
    def generate_sequence(self, config):
        """Generate sequence by adding gates/pulses to waveforms.

        Parameters
        ----------
        config : dict
            Configuration as defined by Labber driver configuration window.

        """
        raise NotImplementedError()

    def get_sequence(self, config):
        """Compile sequence and return it.

        Parameters
        ----------
        config : dict
            Labber instrument configuration.

        Returns
        -------
        list of :obj:`Step`
            The compiled qubit sequence.

        """
        self.sequence_list = []

        if self.perform_process_tomography:
            self._process_tomography.add_pulses(self)

        self.generate_sequence(config)
        if self.perform_state_tomography:
            self._state_tomography.add_pulses(self)

        if self.readout_delay > 0:
            delay = gates.IdentityGate(width=self.readout_delay)
            self.add_gate_to_all(delay, dt=0)
        self.add_gate_to_all(gates.ReadoutGate(), dt=0, align='left')

        return self

    # Public methods for adding pulses and gates to the sequence.
    def add_single_pulse(self,
                         qubit,
                         pulse,
                         t0=None,
                         dt=None,
                         align_left=False):
        """Add single qubit pulse to specified qubit.

        This function still exist to not break existing
        funcationallity. You should really use the add_gate method.

        t0 or dt can be used to override the global pulse spacing.

        Parameters
        ----------
        qubit : int
            Qubit number, indexed from 0.

        pulse : :obj:`Pulse`
            Definition of pulse to add.

        t0 : float, optional
            Absolute pulse position.

        dt : float, optional
            Pulse spacing, referenced to the previous pulse.

        align_left: bool, optional
            If True, aligns the pulse to the left. Defaults to False.

        """
        gate = gates.CustomGate(pulse)
        if align_left is True:
            self.add_gate(qubit, gate, t0, dt, 'left')
        else:
            self.add_gate(qubit, gate, t0, dt, 'center')

    def add_single_gate(self, qubit, gate, t0=None, dt=None, align_left=False):
        """Add single gate to specified qubit sequence.

        Note, this function still exist is to not break existing
        funcationallity. You should really use the add_gate method.

        t0 or dt can be used to override the global pulse spacing.

        Parameters
        ----------
        qubit : int
            Qubit number, indexed from 0.

        gate : :obj:`Gate`
            Definition of gate to add.

        t0 : float, optional
            Absolute pulse position.

        dt : float, optional
            Pulse spacing, referenced to the previous pulse.

        align_left : boolean, optional
            If True, t0 is the start of the pulse, otherwise it is the center
            of the pulse. False is the default.

        """
        if align_left is True:
            self.add_gate(qubit, gate, t0, dt, 'left')
        else:
            self.add_gate(qubit, gate, t0, dt, 'center')

    def add_gate(self,
                 qubit,
                 gate,
                 t0=None,
                 dt=None,
                 align='center',
                 index=None):
        """Add a set of gates to the given qubit sequences.

        For the qubits with no specificied gate, an IdentityGate will be given.
        The length of the step is given by the longest pulse.

        Parameters
        ----------
        qubit : int or list of int
            The qubit(s) to add the gate(s) to.
        gate : :obj:`BaseGate` or list of :obj:`BaseGate`
            The gate(s) to add.
        t0 : float, optional
            Absolute gate position (the default is None).
        dt : float, optional
            Gate spacing, referenced to the previous pulse
            (the default is None).
        align : str, optional
            If two or more qubits have differnt pulse lengths, `align`
            specifies how those pulses should be aligned. 'Left' aligns the
            start, 'center' aligns the centers, and 'right' aligns the end,
            (the default is 'center').
        index : int, optional
            Where in the sequence to insert the new gate.

        """
        step = Step(t0=t0, dt=dt, align=align)
        if isinstance(gate, list):
            if not isinstance(qubit, list):
                raise ValueError(
                    """Provide qubit indices as a list when adding more than
                    one gate.""")
            if len(gate) != len(qubit):
                raise ValueError(
                    "Length of gate list must equal length of qubit list.")

            # log.info('step_add_gate, q: '+ str())
            for q, g in zip(qubit, gate):
                step.add_gate(q, g)
        else:
            if gate.number_of_qubits() > 1:
                if not isinstance(qubit, list):
                    raise ValueError(
                        "Provide qubit list for gates with more than one qubit"
                    )
            else:
                if not isinstance(qubit, int):
                    raise ValueError(
                        "For single gates, give qubit as int (not list).")
            step.add_gate(qubit, gate)
            # log.info('adding gate {} to {}. 2qb gate: {}'.format(gate, qubit, isinstance(gate, gates.TwoQubitGate)))

        if index is None:
            self.sequence_list.append(step)
            # log.info('adding step to sequence list')
            # log.info('sequence len is {}'.format(len(self.sequence_list)))
        else:
            self.sequence_list.insert(index + 1, step)
            # log.info('inserting step in sequence list')
            # log.info('sequence len is {}'.format(len(self.sequence_list)))


    def add_gate_to_all(self, gate, t0=None, dt=None, align='center'):
        """Add a single gate to all qubits.

        Pulses are added at the end of the sequence, with the gate spacing set
        by either the spacing parameter or the aboslut position.
        """
        if isinstance(gate, list):
            raise ValueError("Only single gates allowed.")
        if isinstance(gate, (gates.BaseGate, gates.CompositeGate)):
            if gate.number_of_qubits() > 1:
                raise ValueError(
                    "Not clear how to add multi-qubit gates to all qubits.")

        qubit = list(range((self.n_qubit)))
        gate = [gate for n in range(self.n_qubit)]
        self.add_gate(qubit, gate, t0=t0, dt=dt, align=align)

    def add_gates(self, gates):
        """Add multiple gates to the qubit waveform.

        Pulses are added at the end of the sequence, with the gate spacing set
        by the spacing parameter.

        Examples
        --------
        Add three gates to a two-qubit sequence, first a positive pi-pulse
        around X to qubit 1, then a negative pi/2-pulse to qubit 2, finally
        simultaneous positive pi-pulses to qubits 1 and 2.

        >>> add_gates([[gates.Xp,  None    ],
                       [None,     gates.Y2m],
                       [gates.Xp,  gates.Xp]])

        Parameters
        ----------
        gates : list of list of :obj:`BaseGate`
            List of lists defining gates to add. The innermost list should
            have the same length as number of qubits in the sequence.

        """
        # make sure we have correct input
        if not isinstance(gates, (list, tuple)):
            raise Exception('The input must be a list of list with gates')
        if len(gates) == 0:
            return
        if not isinstance(gates[0], (list, tuple)):
            raise Exception('The input must be a list of list with gates')
        # add gates sequence to waveforms
        for gate in gates:
            # add gate to specific qubit waveform
            qubit = list(range(len(gate)))
            self.add_gate(qubit, gate)

    def set_parameters(self, config={}):
        """Set base parameters using config from from Labber driver.

        Parameters
        ----------
        config : dict
            Configuration as defined by Labber driver configuration window

        """
        # sequence parameters
        d = dict(
            Zero=0,
            One=1,
            Two=2,
            Three=3,
            Four=4,
            Five=5,
            Six=6,
            Seven=7,
            Eight=8,
            Nine=9)
        # If the number of qubits changed, we need to re-init
        if self.n_qubit != d[config.get('Number of qubits')]:
            self.__init__(d[config.get('Number of qubits')])

        # Readout
        self.readout_delay = config.get('Readout delay')

        # process tomography prepulses
        self.perform_process_tomography = \
            config.get('Generate process tomography prepulse', False)
        self._process_tomography.set_parameters(config)

        # state tomography
        self.perform_state_tomography = config.get(
            'Generate state tomography postpulse', False)
        self._state_tomography.set_parameters(config)


class SequenceToWaveforms:
    """Compile a multi qubit sequence into waveforms.

    Parameters
    ----------
    n_qubit : type
        The maximum number of qubits.

    Attributes
    ----------
    dt : float
        Pulse spacing, in seconds.
    local_xy : bool
        If False, collate all waveforms into one.
    simultaneous_pulses : bool
        If False, seperate all pulses in time.
    sample_rate : float
        AWG Sample rate.
    n_pts : float
        Number of points in the waveforms.
    first_delay : float
        Delay between start of waveform and start of the first pulse.
    trim_to_sequence : bool
        If True, adjust `n_points` to just fit the sequence.
    align_to_end : bool
        Align the whole sequence to the end of the waveforms.
        Only relevant if `trim_to_sequence` is False.
    sequences : list of :obj:`Step`
        The qubit sequences.
    qubits : list of :obj:`Qubit`
        Parameters of each qubit.
    wave_xy_delays : list of float
        Indiviudal delays for the XY waveforms.
    wave_z_delays : list of float
        Indiviudal delays for the Z waveforms.
    n_qubit

    """

    def __init__(self, n_qubit):
        self.n_qubit = n_qubit
        self.dt = 10E-9
        self.local_xy = True
        self.simultaneous_pulses = True

        # waveform parameter
        self.sample_rate = 1.2E9
        self.n_pts = 240E3
        self.first_delay = 100E-9
        self.trim_to_sequence = True
        self.align_to_end = False

        self.sequence_list = []
        self.qubits = [qubits.Qubit() for n in range(self.n_qubit)]

        # waveforms
        self._wave_xy = [
            np.zeros(0, dtype=np.complex) for n in range(self.n_qubit)
        ]
        # log.info('_wave_z initiated to 0s')
        self._wave_z = [np.zeros(0) for n in range(self.n_qubit)]
        self._wave_gate = [np.zeros(0) for n in range(self.n_qubit)]
        self.phase_offsets = [np.zeros(0) for n in range(self.n_qubit)]

        # waveform delays
        self.wave_xy_delays = np.zeros(self.n_qubit)
        self.wave_z_delays = np.zeros(self.n_qubit)

        # define pulses
        self.pulses_1qb_xy = [None for n in range(self.n_qubit)]
        self.pulses_1qb_xy_12 = [None for n in range(self.n_qubit)]
        self.pulses_1qb_z = [None for n in range(self.n_qubit)]
        self.pulses_2qb = [None for n in range(self.n_qubit - 1)]
        self.pulses_readout = [None for n in range(self.n_qubit)]
        self.pulses_iSWAP_cplr = [None for n in range(self.n_qubit)]
        self.pulses_iSWAP_tqb = [None for n in range(self.n_qubit)]
        self.pulses_CZ_cplr = [None for n in range(self.n_qubit)]
        self.pulses_CZ_cplr_opposite = [None for n in range(self.n_qubit)]
        self.pulses_CZ_tqb = [None for n in range(self.n_qubit)]
        self.pulses_CZ_tqb_opposite = [None for n in range(self.n_qubit)]

        # cross-talk
        self.compensate_crosstalk = False
        self._crosstalk = crosstalk.Crosstalk()

        # cross-talk
        self.compensate_microwave_crosstalk = False
        self._microwave_crosstalk = microwave_crosstalk.Microwave_Crosstalk()

        # predistortion
        self.perform_predistortion = False
        self._predistortions = [
            predistortion.Predistortion(n) for n in range(self.n_qubit)
        ]
        self._predistortions_z = [
            predistortion.ExponentialPredistortion(n)
            for n in range(self.n_qubit)
        ]

        self._predistortions_z2_z3 = crosstalk_compensation.Crosstalk_Compensation("2-3")
        
        # gate switch waveform
        self.generate_gate_switch = False
        self.uniform_gate = False
        self.gate_delay = 0.0
        self.gate_overlap = 20E-9
        self.minimal_gate_time = 20E-9

        # filters
        self.use_gate_filter = False
        self.use_z_filter = False

        # readout trig settings
        self.readout_trig_generate = False

        # readout wave object and settings
        self.readout = readout.Demodulation(self.n_qubit)
        self.readout_trig = np.array([], dtype=float)
        self.readout_iq = np.array([], dtype=np.complex)

        self.convert_z_to_euler_gates = False

    def get_waveforms(self, sequence):
        """Compile the given sequence into waveforms.

        Parameters
        ----------
        sequences : list of :obj:`Step`
            The qubit sequence to be compiled.

        Returns
        -------
        type
            Description of returned object.

        """
        self.sequence = sequence
        self.sequence_list = sequence.sequence_list
        # log.info('Start of get_waveforms. Len sequence list: {}'.format(len(self.sequence_list)))
        # log.info('Point 1: Sequence_list = {}'.format(self.sequence_list))

        if not self.simultaneous_pulses:
            self._seperate_gates()
        # log.info('Point 2: Sequence_list[6].gates = {}'.format(self.sequence_list[6].gates))
        # log.info('Point 2: Sequence_list = {}'.format(self.sequence_list))

        self._explode_composite_gates()
        # log.info('Point 3: Sequence_list[1].gates = {}'.format(self.sequence_list[1].gates))
        # log.info('Point 3: Sequence_list = {}'.format(self.sequence_list))

        if self.apply_euler_thm == True:
            self._convert_z_to_euler_gates()
        # log.info('Point 4: Sequence_list = {}'.format(self.sequence_list))

        self._add_pulses_and_durations()
        # log.info('Point 4: Sequence_list[6].gates = {}'.format(self.sequence_list[6].gates))

        self._add_timings()
        # log.info('Point 5: Sequence_list[6].gates = {}'.format(self.sequence_list[6].gates))

        self._init_waveforms()
        # log.info('Point 6: Sequence_list = {}'.format(self.sequence_list))


        if self.align_to_end:
            shift = self._round((self.n_pts - 2) / self.sample_rate -
                                self.sequence_list[-1].t_end)
            for step in self.sequence_list:
                step.time_shift(shift)

        # self._perform_virtual_z()
        self._perform_virtual_z_iswap()
        self._generate_waveforms()
        
        # collapse all xy pulses to one waveform if no local XY control
        if not self.local_xy:
            # sum all waveforms to first one
            self._wave_xy[0] = np.sum(self._wave_xy[:self.n_qubit], 0)
            # clear other waveforms
            for n in range(1, self.n_qubit):
                self._wave_xy[n][:] = 0.0

        # apply phase offset
        for n in range(self.n_qubit):
            self._wave_xy[n] = self._wave_xy[n] * np.exp(1j*self.phase_offsets[n]/180.0*np.pi)
        # log.info('before predistortion, _wave_z max is {}'.format(np.max(self._wave_z)))
        # if self.compensate_crosstalk:
        #     self._perform_crosstalk_compensation()
        if self.perform_predistortion:
            self._predistort_xy_waveforms()
        if self.perform_predistortion_z:
            # compensate predistortion for z pulses
            self._wave_z[2] += self._predistortions_z2_z3.compensate(self._wave_z[1])
            self._predistort_z_waveforms()

        if self.readout_trig_generate:
            self._add_readout_trig()
        if self.generate_gate_switch:
            self._add_microwave_gate()
        self._filter_output_waveforms()
        self._zero_last_z_point()
        # Apply offsets
        self.readout_iq += self.readout_i_offset + 1j * self.readout_q_offset


        # create and return dictionary with waveforms
        waveforms = dict()
        waveforms['xy'] = self._wave_xy
        waveforms['z'] = self._wave_z
        waveforms['gate'] = self._wave_gate
        waveforms['readout_trig'] = self.readout_trig
        waveforms['readout_iq'] = self.readout_iq

        log.info('returning z waveforms in get_waveforms. Max is {}'.format(np.max(waveforms['z'])))
        return waveforms

    def _seperate_gates(self):
        new_sequences = []
        for step in self.sequence_list:
            if any(
                    isinstance(gate, (gates.ReadoutGate, gates.IdentityGate))
                    for gate in step.gates):
                # Don't seperate I gates or readouts since we do
                # multiplexed readout
                new_sequences.append(step)
                continue
            for gate in step.gates:
                # log.info('In seperate gates, handling gate {}'.format(gate))
                if gate.gate is not None:
                    new_step = Step(
                        t0=step.t_start, dt=step.dt, align=step.align)
                    new_step.add_gate(gate.qubit, gate.gate)
                    new_sequences.append(new_step)
        # log.info('New sequence [6] is {}'.format(new_sequences[6].gates))
        self.sequence_list = new_sequences

    def _add_timings(self):
        t_start = 0
        for step in self.sequence_list:
            if step.dt is None and step.t0 is None:
                # Use global pulse spacing
                step.dt = self.dt
            # Find longest gate in sequence
            max_duration = np.max([x.duration for x in step.gates])
            if step.t0 is None:
                step.t_start = self._round(t_start + step.dt)
                step.t0 = self._round(step.t_start + max_duration / 2)
            else:
                step.t_start = self._round(step.t0 - max_duration / 2)
            step.t_end = self._round(step.t_start + max_duration)
            t_start = step.t_end  # Next step starts where this one ends
            # Avoid double spacing for steps with 0 duration
            if max_duration == 0:
                t_start = t_start - step.dt

        # Make sure that the sequence is sorted chronologically.
        # self.sequence_list.sort(key=lambda x: x.t_start) # TODO Fix this

        # Make sure that sequnce starts on first delay
        time_diff = self._round(self.first_delay -
                                self.sequence_list[0].t_start)
        for step in self.sequence_list:
            step.time_shift(time_diff)

    def _add_pulses_and_durations(self):
        for step in self.sequence_list:
            for gate in step.gates:
                if gate.pulse is None:
                    gate.pulse = self._get_pulse_for_gate(gate)
                if gate.pulse is None:
                    gate.duration = 0
                else:
                    gate.duration = gate.pulse.total_duration()

    def _get_pulse_for_gate(self, gate):
        qubit = gate.qubit
        gate = gate.gate
        # Virtual Z is special since it has no length
        if isinstance(gate, gates.VirtualZGate):
            pulse = None
        # Get the corresponding pulse for other gates
        elif isinstance(gate, gates.SingleQubitXYRotation):
            pulse = gate.get_adjusted_pulse(self.pulses_1qb_xy[qubit])
        elif isinstance(gate, gates.SingleQubitXYRotation_12):
            pulse = gate.get_adjusted_pulse(self.pulses_1qb_xy_12[qubit])
        elif isinstance(gate, gates.SingleQubitZRotation):
            pulse = gate.get_adjusted_pulse(self.pulses_1qb_z[qubit])
        elif isinstance(gate, gates.IdentityGate):
            pulse = gate.get_adjusted_pulse(self.pulses_1qb_xy[qubit])
        elif isinstance(gate, gates.TwoQubitGate):
            pulse = gate.get_adjusted_pulse(self.pulses_2qb[qubit[0]])
        elif isinstance(gate, gates.ZGate_Cplr_iSWAP):
            pulse = gate.get_adjusted_pulse(self.pulses_iSWAP_cplr[0])
        elif isinstance(gate, gates.ZGate_TQB_iSWAP):
            pulse = gate.get_adjusted_pulse(self.pulses_iSWAP_tqb[0])
        elif isinstance(gate, gates.ZGate_Cplr_CZ):
            pulse = gate.get_adjusted_pulse(self.pulses_CZ_cplr[0])
        elif isinstance(gate, gates.ZGate_Cplr_CZ_opposite):
            pulse = gate.get_adjusted_pulse(self.pulses_CZ_cplr_opposite[0])
        elif isinstance(gate, gates.ZGate_TQB_CZ):
            pulse = gate.get_adjusted_pulse(self.pulses_CZ_tqb[0])
        elif isinstance(gate, gates.ZGate_TQB_CZ_opposite):
            pulse = gate.get_adjusted_pulse(self.pulses_CZ_tqb_opposite[0])
        elif isinstance(gate, gates.ReadoutGate):
            pulse = gate.get_adjusted_pulse(self.pulses_readout[qubit])
        elif isinstance(gate, gates.CustomGate):
            pulse = gate.get_adjusted_pulse(gate.pulse)
        else:
            raise ValueError('Please provide a pulse for {}'.format(gate))

        return pulse
    def _swap_pulse_polarity(self, composite_pulse):
        composite_pulse_new = copy.deepcopy(composite_pulse)

        for i, pulse in enumerate(composite_pulse_new.list_pulses):

            pulse_new = copy.deepcopy(pulse)
            pulse_new.amplitude = pulse.amplitude_opposite
            pulse_new.width = pulse.width_opposite
            pulse_new.plateau = pulse.plateau_opposite
            pulse_new.amplitude_opposite = pulse.amplitude
            pulse_new.width_opposite = pulse.width
            pulse_new.plateau_opposite = pulse.plateau_opposite

            # replace the old pulse by the new pulse
            composite_pulse_new.list_pulses[i] = pulse_new

        return composite_pulse_new

    def _predistort_xy_waveforms(self):
        """Pre-distort the waveforms."""
        # go through and predistort all xy waveforms
        n_wave = self.n_qubit if self.local_xy else 1
        for n in range(n_wave):
            self._wave_xy[n] = self._predistortions[n].predistort(
                self._wave_xy[n])

    def _predistort_z_waveforms(self):
        # go through and predistort all waveforms
        for n in range(self.n_qubit):
            self._wave_z[n] = self._predistortions_z[n].predistort(
                self._wave_z[n])


    # def _perform_crosstalk_compensation(self):
    #     """Compensate for Z-control crosstalk."""
    #     self._wave_z = self._crosstalk.compensate(self._wave_z)

    def _explode_composite_gates(self):
        # Loop through the sequence until all CompositeGates are removed
        # Note that there could be nested CompositeGates
        n = 0
        while n < len(self.sequence_list):
            step = self.sequence_list[n]
            i = 0
            while i < len(step.gates):
                gate = step.gates[i]
                if isinstance(gate.gate, gates.CompositeGate):
                    # log.info('In exploded composite, handling composite gate {} at step {}'.format(gate, n))
                    for m, g in enumerate(gate.gate.sequence):
                        new_gate = [x.gate for x in g.gates]
                        # Single gates shouldn't be lists
                        if len(new_gate) == 1:
                            new_gate = new_gate[0]

                        # Translate gate qubit number to device qubit number
                        new_qubit = [x.qubit for x in g.gates]
                        for j, q in enumerate(new_qubit):
                            if isinstance(q, int):
                                if isinstance(gate.qubit, int):
                                    new_qubit[j] = gate.qubit
                                    continue
                                new_qubit[j] = gate.qubit[q]
                            else:
                                new_qubit[j] = []
                                for k in q:
                                    new_qubit[j].append(gate.qubit[k])

                        # Single qubit shouldn't be lists
                        if len(new_qubit) == 1:
                            new_qubit = new_qubit[0]
                        # log.info('In explode composite; modifying {} by adding gate {} at index {}'.format(gate, new_gate, n+m))
                        # log.info('before adding new-qubit(' + str(new_qubit) + ') new-gate (' + str(new_gate) + '): ' + str(self.sequence_list))
                        self.sequence.add_gate(
                            copy.copy(new_qubit), copy.copy(new_gate), index=n + m)
                        # log.info('after: ' + str(self.sequence_list))

                    del step.gates[i]
                    # log.info('In composite gates, removing step {}'.format(i))

                    continue
                i = i + 1
            n = n + 1

        # Remove any empty steps where the composite gates were
        i = 0
        while i < len(self.sequence_list):
            step = self.sequence_list[i]
            # log.info(str(step.gates))
            if len(step.gates) == 0:
                # log.info('In composite gates, removing step {}'.format(i))
                del self.sequence_list[i]
                continue
            i = i + 1

        # for i, step in enumerate(self.sequence_list):
        #     log.info('At end of explode, step {} is {}'.format(i, step.gates))

    def _perform_virtual_z(self):
        """Shifts the phase of pulses subsequent to virtual z gates."""
        for qubit in range(self.n_qubit):
            phase = 0
            for step in self.sequence_list:
                for gate in step.gates:
                    gate_obj = None
                    if qubit == gate.qubit:  # TODO Allow for 2 qb
                        gate_obj = gate.gate
                    if isinstance(gate_obj, gates.VirtualZGate):
                        phase += gate_obj.theta
                        continue
                    if (isinstance(gate_obj, gates.SingleQubitXYRotation)
                            and phase != 0):
                        gate.gate = copy.copy(gate_obj)
                        gate.gate.phi += phase
                        # Need to recomput the pulse
                        gate.pulse = self._get_pulse_for_gate(gate)



    def _convert_z_to_euler_gates(self):
        # 1. find where two-qubit gate locates
        # 2. split sequence into sub-sequence by two qubit gates. Each sub-sequence consisting of only single qubit gates. 
        # 3. check if Euler Z gate exists in each sub-sequence 
        # 4. if there is Euler Z, then convert it into the Euler rotations
        # 5. build un new sequence list 

        # find where 2qb locates
        list_2qb_gate_indicies = list()
        list_steps_2qb_gate = list()
        for n in range(len(self.sequence_list)):
            step = self.sequence_list[n]
            if (isinstance(step.gates[1].gate, gates.ZGate_Cplr_iSWAP) or
               isinstance(step.gates[1].gate, gates.ZGate_Cplr_CZ)) :
                list_2qb_gate_indicies.append(n)
                list_steps_2qb_gate.append(step)


        # split sequence into sub-sequence by two qubit gates. 
        list_subseqs = list()
        step_readout = None
        for j in range(len(list_2qb_gate_indicies)+1):
            if j == 0:
                if len(list_2qb_gate_indicies) == 0: #there is no 2qb gates.
                    if isinstance(self.sequence_list[-1].gates[0].gate, gates.ReadoutGate):
                        list_subseqs.append(self.sequence_list[:-1])
                        step_readout = self.sequence_list[-1]
                    else:
                        list_subseqs.append(self.sequence_list)     

                else:
                    list_subseqs.append(self.sequence_list[:list_2qb_gate_indicies[j]])

            elif j == len(list_2qb_gate_indicies):
                # log.info(self.sequence_list[-1])
                if isinstance(self.sequence_list[-1].gates[0].gate, gates.ReadoutGate):
                    list_subseqs.append(self.sequence_list[list_2qb_gate_indicies[j-1]+1:-1])
                    step_readout = self.sequence_list[-1]
                else:
                    list_subseqs.append(self.sequence_list[list_2qb_gate_indicies[j-1]+1:])

            else:
                list_subseqs.append(self.sequence_list[list_2qb_gate_indicies[j-1]+1: list_2qb_gate_indicies[j]])
        log.info('list_subseqs:{}'.format(list_subseqs))
        # find which sub-sequence has eulerZ gates
        list_eulerz_indices = list()
        for k in range(len(list_subseqs)):
            eulerz_found = False
            for l in range(len(list_subseqs[k])):
                step = list_subseqs[k][l] 
                if isinstance(step.gates[0].gate, gates.EulerZGate):
                    if not eulerz_found:
                        list_eulerz_indices.append(k)
                        log.info('eulerz append: {}'.format(k))
                    eulerz_found = True
            if ((not eulerz_found) and len(list_subseqs[k])>3):
                # if the number of single qubit gates more than 3, consider rewrite in euler-format
                list_eulerz_indices.append(k)
                eulerz_found = True

        # for sub-sequence having euler-Z gates, convert them into XYX-convention
        list_subseqs_eulerz = list()
        for index in list_eulerz_indices:
            subseq = list_subseqs[index]

            log.info('subseq in euler_z: {}'.format(subseq))
            len_seq = len(subseq)
            list_qb1_gates = list()
            list_qb2_gates = list()
            for m in range(len_seq):
                for gate_on_qb in (subseq[m].gates):
                    if gate_on_qb.qubit == 0:
                        list_qb1_gates.append(gate_on_qb.gate)
                    elif gate_on_qb.qubit == 2:
                        list_qb2_gates.append(gate_on_qb.gate)
            print('list_qb1_gates: {}'.format(list_qb1_gates))
            list_qb1_euler_gates = gate_to_euler_gates(list_qb1_gates, convention = 'x-y-x')
            print('list_qb2_gates: {}'.format(list_qb2_gates))
            list_qb2_euler_gates = gate_to_euler_gates(list_qb2_gates, convention = 'x-y-x')
            subseq_eulerz = []
            for m in range(3):
                step = Step()
                step.add_gate(0,list_qb1_euler_gates[m])
                step.add_gate(1,gates.IdentityGate(width = None))
                step.add_gate(2,list_qb2_euler_gates[m])
                subseq_eulerz.append(step)
            list_subseqs_eulerz.append(subseq_eulerz)

        # build-up the sequence
        new_sequence_list = list()
        cnt_eulerz = 0
        cnt_2qb_gate = 0
        for j, subseq in enumerate(list_subseqs):
            log.info("j: {}, list_eulerz_indices: {}".format(j, list_eulerz_indices))
            if j in list_eulerz_indices:
                # print(list_subseqs_eulerz[cnt])
                log.info('%d / %d'%(cnt_eulerz, len(list_subseqs_eulerz)))
                log.info('extend for new_sequence_list: {}'.format(list_subseqs_eulerz[cnt_eulerz]))
                new_sequence_list.extend(list_subseqs_eulerz[cnt_eulerz])
                cnt_eulerz += 1
            else:
                new_sequence_list.extend(subseq)
            if j != len(list_subseqs)-1:
                new_sequence_list.append(list_steps_2qb_gate[cnt_2qb_gate])
                cnt_2qb_gate +=1
        if step_readout is not None:
            new_sequence_list.append(step_readout)

        log.info('sequence_list (before): {}'.format(self.sequence_list))
        self.sequence_list = new_sequence_list
        log.info('sequence_list (after): {}'.format(self.sequence_list))

    def _perform_virtual_z_iswap(self):
        # log.info('_perform_virtual_z_iswap, n_qubit: {}'.format(self.n_qubit))
        arr_phase = np.zeros(self.n_qubit) # only works for two-qubit system
        num_iswap = 0
        for step in self.sequence_list:
            for gate in step.gates:
                for qubit in range(self.n_qubit):
                    gate_obj = None

                    if qubit == gate.qubit:  # TODO Allow for 2 qb
                        gate_obj = gate.gate
                        # log.info('qubit: {}, gate: {}'.format(qubit, str(gate)))
                    if isinstance(gate_obj, gates.ZGate_Cplr_iSWAP):
                        # # swap 0 <-> 2
                        # temp = arr_phase[0]
                        # arr_phase[0] = arr_phase[2]
                        # arr_phase[2] = temp
                        # num_iswap +=1
                        # log.info('num_iswap: {}, qubit: {}'.format(num_iswap, qubit))
                        # log.info('arr_phase_swapped!')
                        # log.info('Phase offset for QBs: {}'.format( gates.iSWAP_Cplr.phi_offsets))
                        # if qubit == 0:
                        arr_phase[0] += gates.iSWAP_Cplr.phi1_Symm
                        arr_phase[2] += gates.iSWAP_Cplr.phi2_Symm
                            # log.info('add arr_phase[{}]: {}'.format(qubit, num_iswap))
                        # elif qubit == 2:

                    if isinstance(gate_obj, gates.VirtualZGate):
                        arr_phase[qubit] += gate_obj.theta
                        # log.info('arr_phase[{}]: {}'.format(qubit,arr_phase[qubit]))
                    if (isinstance(gate_obj, gates.SingleQubitXYRotation)):
                            # and arr_phase[qubit] != 0):
                        gate.gate = copy.copy(gate_obj)
                        gate.gate.phi += arr_phase[qubit]
                        # asymm_part = 0
                        # if qubit == 0:
                        #     asymm_part = np.floor(num_iswap*0.5)*(gates.iSWAP_Cplr.phi1_Asymm + gates.iSWAP_Cplr.phi2_Asymm) + np.floor(num_iswap%2*0.5 + 0.5)*gates.iSWAP_Cplr.phi1_Asymm
                        # elif qubit == 2:
                        #     asymm_part = np.floor(num_iswap*0.5)*(gates.iSWAP_Cplr.phi1_Asymm + gates.iSWAP_Cplr.phi2_Asymm) + np.floor(num_iswap%2*0.5 + 0.5)*gates.iSWAP_Cplr.phi2_Asymm
                        # gate.gate.phi += asymm_part   
                        # log.info('Number of iSWAP: {}, QB{} -, symm_part: {}, asymm_part:{}'.format(num_iswap, qubit, arr_phase[qubit], asymm_part))
                        gate.pulse = self._get_pulse_for_gate(gate)
                        

    def _add_microwave_gate(self):
        """Create waveform for gating microwave switch."""
        n_wave = self.n_qubit if self.local_xy else 1
        # go through all waveforms
        for n, wave in enumerate(self._wave_xy[:n_wave]):
            if self.uniform_gate:
                # the uniform gate is all ones
                gate = np.ones_like(wave)
                # if creating readout trig, turn off gate during readout
                if self.readout_trig_generate:
                    gate[-int((self.readout_trig_duration - self.gate_overlap -
                               self.gate_delay) * self.sample_rate):] = 0.0
            else:
                # non-uniform gate, find non-zero elements
                gate = np.array(np.abs(wave) > 0.0, dtype=float)
                # fix gate overlap
                n_overlap = int(np.round(self.gate_overlap * self.sample_rate))
                diff_gate = np.diff(gate)
                indx_up = np.nonzero(diff_gate > 0.0)[0]
                indx_down = np.nonzero(diff_gate < 0.0)[0]
                # add extra elements to left and right for overlap
                for indx in indx_up:
                    gate[max(0, indx - n_overlap):(indx + 1)] = 1.0
                for indx in indx_down:
                    gate[indx:(indx + n_overlap + 1)] = 1.0

                # fix gaps in gate shorter than min (look for 1>0)
                diff_gate = np.diff(gate)
                indx_up = np.nonzero(diff_gate > 0.0)[0]
                indx_down = np.nonzero(diff_gate < 0.0)[0]
                # ignore first transition if starting in zero
                if gate[0] == 0:
                    indx_up = indx_up[1:]
                n_down_up = min(len(indx_down), len(indx_up))
                len_down = indx_up[:n_down_up] - indx_down[:n_down_up]
                # find short gaps
                short_gaps = np.nonzero(
                    len_down < (self.minimal_gate_time * self.sample_rate))[0]
                for indx in short_gaps:
                    gate[indx_down[indx]:(1 + indx_up[indx])] = 1.0

                # shift gate in time
                n_shift = int(np.round(self.gate_delay * self.sample_rate))
                if n_shift < 0:
                    n_shift = abs(n_shift)
                    gate = np.r_[gate[n_shift:], np.zeros((n_shift, ))]
                elif n_shift > 0:
                    gate = np.r_[np.zeros((n_shift, )), gate[:(-n_shift)]]
            # make sure gate starts/ends in 0
            gate[0] = 0.0
            gate[-1] = 0.0
            # store results
            self._wave_gate[n] = gate

    def _filter_output_waveforms(self):
        """Filter output waveforms"""
        # start with gate
        if self.use_gate_filter and self.gate_filter_size > 1:
            # prepare filter
            window = self._get_filter_window(
                self.gate_filter_size, self.gate_filter,
                self.gate_filter_kaiser_beta)
            # apply filter to all output waveforms
            n_wave = self.n_qubit if self.local_xy else 1
            for n in range(n_wave):
                self._wave_gate[n] = self._apply_window_filter(
                    self._wave_gate[n], window)
                # make sure gate starts/ends in 0
                self._wave_gate[n][0] = 0.0
                self._wave_gate[n][-1] = 0.0

        # same for z waveforms
        if self.use_z_filter and self.z_filter_size > 1:
            # prepare filter
            window = self._get_filter_window(
                self.z_filter_size, self.z_filter, self.z_filter_kaiser_beta)
            # apply filter to all output waveforms
            for n in range(self.n_qubit):
                self._wave_z[n] = self._apply_window_filter(
                    self._wave_z[n], window)

    def _zero_last_z_point(self):
        """Make sure last point in z waveforms is always zero, since this is 
           the value output by the AWG between sequences.
        """
        for n in range(self.n_qubit):
            self._wave_z[n][-1]=0

    def _get_filter_window(self, size=11, window='Kaiser', kaiser_beta=14.0):
        """Get filter for waveform convolution"""
        if window == 'Rectangular':
            w = np.ones(size)
        elif window == 'Bartlett':
            # for filters that start/end in zeros, add 2 points and truncate
            w = np.bartlett(max(1, size+2))
            w = w[1:-1]
        elif window == 'Blackman':
            w = np.blackman(size + 2)
            w = w[1:-1]
        elif window == 'Hamming':
            w = np.hamming(size)
        elif window == 'Hanning':
            w = np.hanning(size + 2)
            w = w[1:-1]
        elif window == 'Kaiser':
            w = np.kaiser(size, kaiser_beta)
        else:
            raise('Unknown filter windows function %s.' % str(window))
        return w/w.sum()

    def _apply_window_filter(self, x, window):
        """Apply window filter to input waveform

        Parameters
        ----------
        x: np.array
            Input waveform.
        window: np.array
            Filter waveform.

        Returns
        -------
        np.array
            Filtered waveform.

        """
        # buffer waveform to avoid wrapping effects at boundaries
        n = len(window)
        s = np.r_[2*x[0] - x[n-1::-1], x, 2*x[-1] - x[-1:-n:-1]]
        # apply convolution
        y = np.convolve(s, window, mode='same')
        return y[n:-n+1]

    def _round(self, t, acc=1E-12):
        """Round the time `t` with a certain accuarcy `acc`.

        Parameters
        ----------
        t : float
            The time to be rounded.
        acc : float
            The accuarcy (the default is 1E-12).

        Returns
        -------
        float
            The rounded time.

        """
        return int(np.round(t / acc)) * acc

    def _add_readout_trig(self):
        """Create waveform for readout trigger."""
        trig = np.zeros_like(self.readout_iq)
        start = (np.abs(self.readout_iq) > 0.0).nonzero()[0][0]
        end = int(
            np.min((start + self.readout_trig_duration * self.sample_rate,
                    self.n_pts_readout)))
        trig[start:end] = self.readout_trig_amplitude

        # make sure trig starts and ends in 0.
        trig[0] = 0.0
        trig[-1] = 0.0
        self.readout_trig = trig

    def _init_waveforms(self):
        """Initialize waveforms according to sequence settings."""
        # To keep the first pulse delay, use the smallest delay as reference.
        min_delay = np.min([
            self.wave_xy_delays[:self.n_qubit],
            self.wave_z_delays[:self.n_qubit]
        ])
        self.wave_xy_delays -= min_delay
        self.wave_z_delays -= min_delay
        max_delay = np.max([
            self.wave_xy_delays[:self.n_qubit],
            self.wave_z_delays[:self.n_qubit]
        ])

        # find the end of the sequence
        # only include readout in size estimate if all waveforms have same size
        if self.readout_match_main_size:
            if len(self.sequence_list) == 0:
                end = max_delay
            else:
                end = np.max(
                    [s.t_end for s in self.sequence_list]) + max_delay
        else:
            if len(self.sequence_list) <= 1:
                end = max_delay
            else:
                end = np.max(
                    [s.t_end for s in self.sequence_list[0:-1]]) + max_delay

        # create empty waveforms of the correct size
        if self.trim_to_sequence:
            self.n_pts = int(np.ceil(end * self.sample_rate)) + 1
            if self.n_pts % 2 == 1:
                # Odd n_pts give spectral leakage in FFT
                self.n_pts += 1
        for n in range(self.n_qubit):
            self._wave_xy[n] = np.zeros(self.n_pts, dtype=np.complex)
            # log.info('wave z {} initiated to 0'.format(n))
            self._wave_z[n] = np.zeros(self.n_pts, dtype=float)
            self._wave_gate[n] = np.zeros(self.n_pts, dtype=float)

        # Waveform time vector
        self.t = np.arange(self.n_pts) / self.sample_rate

        # readout trig and i/q waveforms
        if self.readout_match_main_size:
            # same number of points for readout and main waveform
            self.n_pts_readout = self.n_pts
        else:
            # different number of points for readout and main waveform
            self.n_pts_readout = 1 + int(
                np.ceil(self.sample_rate * (self.sequence_list[-1].t_end -
                                            self.sequence_list[-1].t_start)))
            if self.n_pts_readout % 2 == 1:
                # Odd n_pts give spectral leakage in FFT
                self.n_pts_readout += 1

        self.readout_trig = np.zeros(self.n_pts_readout, dtype=float)
        self.readout_iq = np.zeros(self.n_pts_readout, dtype=np.complex)

    def _generate_waveforms(self):
        """Generate the waveforms corresponding to the sequence."""
        # log.info('generating waveform from sequence. Len is {}'.format(len(self.sequence_list)))
        for step in self.sequence_list:
            # log.info('Generating gates {}'.format(step.gates))
            for gate in step.gates:
                qubit = gate.qubit
                # log.info('-- qubit: ' + str(qubit))
                if isinstance(qubit, list):
                    qubit = qubit[0]
                gate_obj = gate.gate
                # log.info('-- gate_obj: ' + str(gate_obj))
                if isinstance(gate_obj, gates.SingleQubitXYRotation):
                    waveform = self._wave_xy[qubit]
                    delay = self.wave_xy_delays[qubit]
                    if self.compensate_microwave_crosstalk:
                        mw_crosstalk = self._microwave_crosstalk.compensation_matrix[:,
                                                                        qubit]

                elif isinstance(gate_obj,
                              (gates.IdentityGate, gates.VirtualZGate)):
                    continue
                elif isinstance(gate_obj, gates.SingleQubitZRotation):
                    waveform = self._wave_z[qubit]
                    delay = self.wave_z_delays[qubit]
                    if self.compensate_crosstalk:
                        crosstalk = self._crosstalk.compensation_matrix[:,
                                                                        qubit]
                elif isinstance(gate_obj, gates.ZGate_Cplr_iSWAP):
                    waveform = self._wave_z[qubit]
                    delay = self.wave_z_delays[qubit]
                    if self.compensate_crosstalk:
                        crosstalk = self._crosstalk.compensation_matrix[:,
                                                                        qubit]
                elif isinstance(gate_obj, gates.ZGate_TQB_iSWAP):
                    waveform = self._wave_z[qubit]
                    delay = self.wave_z_delays[qubit]
                    if self.compensate_crosstalk:
                        crosstalk = self._crosstalk.compensation_matrix[:,
                                                                        qubit]
                elif isinstance(gate_obj, gates.ZGate_Cplr_CZ):
                    waveform = self._wave_z[qubit]
                    delay = self.wave_z_delays[qubit]
                    if self.compensate_crosstalk:
                        crosstalk = self._crosstalk.compensation_matrix[:,
                                                                        qubit]
                elif isinstance(gate_obj, gates.ZGate_TQB_CZ):
                    waveform = self._wave_z[qubit]
                    delay = self.wave_z_delays[qubit]
                    if self.compensate_crosstalk:
                        crosstalk = self._crosstalk.compensation_matrix[:,
                                                                        qubit]
                elif isinstance(gate_obj, gates.ZGate_Cplr_CZ_opposite):
                    waveform = self._wave_z[qubit]
                    delay = self.wave_z_delays[qubit]
                    if self.compensate_crosstalk:
                        crosstalk = self._crosstalk.compensation_matrix[:,
                                                                        qubit]
                elif isinstance(gate_obj, gates.ZGate_TQB_CZ_opposite):
                    waveform = self._wave_z[qubit]
                    delay = self.wave_z_delays[qubit]
                    if self.compensate_crosstalk:
                        crosstalk = self._crosstalk.compensation_matrix[:,
                                                                        qubit]
                elif isinstance(gate_obj, gates.TwoQubitGate):
                    # log.info('adding 2qb gate waveforms')
                    waveform = self._wave_z[qubit]
                    delay = self.wave_z_delays[qubit]
                    if self.compensate_crosstalk:
                        crosstalk = self._crosstalk.compensation_matrix[:,
                                                                        qubit]
                elif isinstance(gate_obj, 
                    (gates.SingleQubitXYRotation, gates.SingleQubitXYRotation_12)):
                    waveform = self._wave_xy[qubit]
                    delay = self.wave_xy_delays[qubit]
                elif isinstance(gate_obj, gates.ReadoutGate):
                    waveform = self.readout_iq
                    delay = 0
                else:
                    raise ValueError(
                        "Don't know which waveform to add {} to.".format(
                            gate_obj))

                # get the range of indices in use
                if (isinstance(gate_obj, gates.ReadoutGate)
                        and not self.readout_match_main_size):
                    # special case for readout if not matching main wave size
                    start = 0.0
                    end = self._round(step.t_end - step.t_start)
                else:
                    start = self._round(step.t_start + delay)
                    end = self._round(step.t_end + delay)

                if (self.compensate_crosstalk and
                    isinstance(gate_obj,
                               (gates.SingleQubitZRotation,
                                gates.ZGate_Cplr_iSWAP,
                                gates.ZGate_TQB_iSWAP,
                                gates.ZGate_Cplr_CZ,
                                gates.ZGate_Cplr_CZ_opposite,
                                gates.ZGate_TQB_CZ,
                                gates.ZGate_TQB_CZ_opposite,
                                gates.TwoQubitGate))):
                    for q in range(self.n_qubit):
                        waveform = self._wave_z[q]
                        delay = self.wave_z_delays[q]
                        start = self._round(step.t_start + delay)
                        end = self._round(step.t_end + delay)
                        indices = np.arange(
                            max(np.floor(start * self.sample_rate), 0),
                            min(
                                np.ceil(end * self.sample_rate),
                                len(waveform)),
                            dtype=int)

                        # return directly if no indices
                        if len(indices) == 0:
                            continue

                        # calculate time values for the pulse indices
                        t = indices / self.sample_rate
                        max_duration = end - start
                        middle = end - max_duration / 2
                        if step.align == 'center':
                            t0 = middle
                        elif step.align == 'left':
                            t0 = middle - (max_duration - gate.duration) / 2
                        elif step.align == 'right':
                            t0 = middle + (max_duration - gate.duration) / 2

                        scaling_factor = float(crosstalk[q, 0])
                        if q != qubit:
                            scaling_factor = -scaling_factor
                        waveform[indices] += (
                            scaling_factor
                            * gate.pulse.calculate_waveform(t0, t))

                elif(self.compensate_microwave_crosstalk and
                    isinstance(gate_obj, 
                            (gates.SingleQubitXYRotation))):
                    for q in range(self.n_qubit):
                        waveform = self._wave_xy[q]
                        delay = self.wave_xy_delays[q]
                        start = self._round(step.t_start + delay)
                        end = self._round(step.t_end + delay)
                        indices = np.arange(
                            max(np.floor(start * self.sample_rate), 0),
                            min(
                                np.ceil(end * self.sample_rate),
                                len(waveform)),
                            dtype=int)

                        # return directly if no indices
                        if len(indices) == 0:
                            continue

                        # calculate time values for the pulse indices
                        t = indices / self.sample_rate
                        max_duration = end - start
                        middle = end - max_duration / 2
                        if step.align == 'center':
                            t0 = middle
                        elif step.align == 'left':
                            t0 = middle - (max_duration - gate.duration) / 2
                        elif step.align == 'right':
                            t0 = middle + (max_duration - gate.duration) / 2

                        # log.info('mw_crosstalk: {}'.format(mw_crosstalk))
                        scaling_factor = (mw_crosstalk[q])
                        if q != qubit:
                            scaling_factor = scaling_factor
                        # log.info('mw crosstalk scaling_factor: {}'.format(scaling_factor))
                        waveform[indices] += (
                            scaling_factor
                            * gate.pulse.calculate_waveform(t0, t))

                else:
                    # calculate the pulse waveform for the selected indices
                    indices = np.arange(
                        max(np.floor(start * self.sample_rate), 0),
                        min(np.ceil(end * self.sample_rate), len(waveform)),
                        dtype=int)

                    # return directly if no indices
                    if len(indices) == 0:
                        continue

                    # calculate time values for the pulse indices
                    t = indices / self.sample_rate
                    max_duration = end - start
                    middle = end - max_duration / 2
                    if step.align == 'center':
                        t0 = middle
                    elif step.align == 'left':
                        t0 = middle - (max_duration - gate.duration) / 2
                    elif step.align == 'right':
                        t0 = middle + (max_duration - gate.duration) / 2
                    waveform[indices] += gate.pulse.calculate_waveform(t0, t)
                    # log.info('gate_obj: ' + str(gate_obj))

    def set_parameters(self, config={}):
        """Set base parameters using config from from Labber driver.

        Parameters
        ----------
        config : dict
            Configuration as defined by Labber driver configuration window

        """
        # sequence parameters
        d = dict(
            Zero=0,
            One=1,
            Two=2,
            Three=3,
            Four=4,
            Five=5,
            Six=6,
            Seven=7,
            Eight=8,
            Nine=9)

        # If the number of qubits changed, re-init to update pulses etc
        if self.n_qubit != d[config.get('Number of qubits')]:
            self.__init__(d[config.get('Number of qubits')])

        self.dt = config.get('Pulse spacing')
        self.local_xy = config.get('Local XY control')
        # default for simultaneous pulses is true, only option for benchmarking
        self.simultaneous_pulses = config.get('Simultaneous pulses', True)

        # waveform parameters
        self.sample_rate = config.get('Sample rate')
        self.n_pts = int(config.get('Number of points', 0))
        self.first_delay = config.get('First pulse delay')
        self.trim_to_sequence = config.get('Trim waveform to sequence')
        self.trim_start = config.get('Trim both start and end')
        self.align_to_end = config.get('Align pulses to end of waveform')
        self.apply_euler_thm = config.get('Apply Euler Angle Thm')

        for n in range(self.n_qubit):
            m = n + 1
            self.phase_offsets[n] = config.get('Phase Offset #{}'.format(m), 0)
        # log.info('self.phase_offsets = {}'.format(self.phase_offsets))
        # qubit spectra
        for n in range(self.n_qubit):
            m = n + 1  # pulses are indexed from 1 in Labber
            qubit = qubits.Transmon(
                config.get('f01 max #{}'.format(m)),
                config.get('f01 min #{}'.format(m)),
                config.get('Ec #{}'.format(m)),
                config.get('Vperiod #{}'.format(m)),
                config.get('Voffset #{}'.format(m)),
                config.get('V0 #{}'.format(m)),
            )
            self.qubits[n] = qubit

        # single-qubit pulses XY
        for n, pulse in enumerate(self.pulses_1qb_xy):
            m = n + 1  # pulses are indexed from 1 in Labber
            pulse = (getattr(pulses, config.get('Pulse type'))(complex=True))
            # global parameters
            pulse.truncation_range = config.get('Truncation range')
            pulse.start_at_zero = config.get('Start at zero')
            pulse.use_drag = config.get('Use DRAG')
            # pulse shape
            if config.get('Uniform pulse shape'):
                pulse.width = config.get('Width')
                pulse.plateau = config.get('Plateau')
            else:
                pulse.width = config.get('Width #%d' % m)
                pulse.plateau = config.get('Plateau #%d' % m)

            if config.get('Uniform amplitude'):
                pulse.amplitude = config.get('Amplitude')
            else:
                pulse.amplitude = config.get('Amplitude #%d' % m)

            # pulse-specific parameters
            pulse.frequency = config.get('Frequency #%d' % m)
            pulse.drag_coefficient = config.get('DRAG scaling #%d' % m)
            pulse.drag_detuning = config.get('DRAG frequency detuning #%d' % m)

            self.pulses_1qb_xy[n] = pulse

        # single-qubit pulses XY (1-2)
        for n, pulse in enumerate(self.pulses_1qb_xy_12):
            m = n + 1  # pulses are indexed from 1 in Labber
            pulse = (getattr(pulses, config.get('Pulse type (1-2)'))(complex=True))
            # global parameters
            pulse.truncation_range = config.get('Truncation range (1-2)')
            pulse.start_at_zero = config.get('Start at zero (1-2)')
            pulse.use_drag = config.get('Use DRAG (1-2)')
            # pulse shape
            if config.get('Uniform pulse shape (1-2)'):
                pulse.width = config.get('Width (1-2)')
                pulse.plateau = config.get('Plateau (1-2)')
            else:
                pulse.width = config.get('Width #%d (1-2)' % m)
                pulse.plateau = config.get('Plateau #%d (1-2)' % m)

            if config.get('Uniform amplitude (1-2)'):
                pulse.amplitude = config.get('Amplitude (1-2)')
            else:
                pulse.amplitude = config.get('Amplitude #%d (1-2)' % m)

            # pulse-specific parameters
            pulse.frequency = config.get('Frequency #%d (1-2)' % m)
            pulse.drag_coefficient = config.get('DRAG scaling #%d (1-2)' % m)
            pulse.drag_detuning = config.get('DRAG frequency detuning #%d (1-2)' % m)

            self.pulses_1qb_xy_12[n] = pulse

        # single-qubit pulses Z
        for n, pulse in enumerate(self.pulses_1qb_z):
            # pulses are indexed from 1 in Labber
            m = n + 1
            # global parameters
            pulse = (getattr(pulses,
                             config.get('Pulse type, Z'))(complex=False))
            pulse.truncation_range = config.get('Truncation range, Z')
            pulse.start_at_zero = config.get('Start at zero, Z')
            # pulse shape
            if config.get('Uniform pulse shape, Z'):
                pulse.width = config.get('Width, Z')
                pulse.plateau = config.get('Plateau, Z')
            else:
                pulse.width = config.get('Width #%d, Z' % m)
                pulse.plateau = config.get('Plateau #%d, Z' % m)

            if config.get('Uniform amplitude, Z'):
                pulse.amplitude = config.get('Amplitude, Z')
            else:
                pulse.amplitude = config.get('Amplitude #%d, Z' % m)

            self.pulses_1qb_z[n] = pulse

        # two-qubit pulses
        for n, pulse in enumerate(self.pulses_2qb):
            # pulses are indexed from 1 in Labber
            s = ' #%d%d' % (n + 1, n + 2)
            # global parameters
            pulse = (getattr(pulses,
                             config.get('Pulse type, 2QB'))(complex=False))

            if config.get('Pulse type, 2QB') in ['CZ', 'NetZero']:
                pulse.F_Terms = d[config.get('Fourier terms, 2QB')]
                if config.get('Uniform 2QB pulses'):
                    pulse.width = config.get('Width, 2QB')
                    pulse.plateau = config.get('Plateau, 2QB')
                else:
                    pulse.width = config.get('Width, 2QB' + s)
                    pulse.plateau = config.get('Plateau, 2QB')

                # spectra
                if config.get('Assume linear dependence' + s, True):
                    pulse.qubit = None
                else:
                    pulse.qubit = self.qubits[n]

                # Get Fourier values
                if d[config.get('Fourier terms, 2QB')] == 4:
                    pulse.Lcoeff = np.array([
                        config.get('L1, 2QB' + s),
                        config.get('L2, 2QB' + s),
                        config.get('L3, 2QB' + s),
                        config.get('L4, 2QB' + s)
                    ])
                elif d[config.get('Fourier terms, 2QB')] == 3:
                    pulse.Lcoeff = np.array([
                        config.get('L1, 2QB' + s),
                        config.get('L2, 2QB' + s),
                        config.get('L3, 2QB' + s)
                    ])
                elif d[config.get('Fourier terms, 2QB')] == 2:
                    pulse.Lcoeff = np.array(
                        [config.get('L1, 2QB' + s),
                         config.get('L2, 2QB' + s)])
                elif d[config.get('Fourier terms, 2QB')] == 1:
                    pulse.Lcoeff = np.array([config.get('L1, 2QB' + s)])

                pulse.Coupling = config.get('Coupling, 2QB' + s)
                pulse.Offset = config.get('f11-f20 initial, 2QB' + s)
                pulse.amplitude = config.get('f11-f20 final, 2QB' + s)
                pulse.dfdV = config.get('df/dV, 2QB' + s)
                pulse.negative_amplitude = config.get('Negative amplitude' + s)

                pulse.calculate_cz_waveform()

            else:
                pulse.truncation_range = config.get('Truncation range, 2QB')
                pulse.start_at_zero = config.get('Start at zero, 2QB')
                # pulse shape
                if config.get('Uniform 2QB pulses'):
                    pulse.width = config.get('Width, 2QB')
                    pulse.plateau = config.get('Plateau, 2QB')
                else:
                    pulse.width = config.get('Width, 2QB' + s)
                    pulse.plateau = config.get('Plateau, 2QB' + s)
                # pulse-specific parameters
                pulse.amplitude = config.get('Amplitude, 2QB' + s)

            gates.CZ.new_angles(
                config.get('QB1 Phi 2QB #12'), config.get('QB2 Phi 2QB #12'))

            self.pulses_2qb[n] = pulse

        # two-qubit (coupler) pulses: composite pulses applied to both coupler (Cplr) & tuanble qubit (TQB).
        _gates = ['iSWAP', 'CZ']
        _qubits = ['Cplr', 'TQB']
        for _gate in _gates:
            for _qubit in _qubits:
                if (_gate == 'iSWAP' and _qubit == 'Cplr'):
                    _pulses = self.pulses_iSWAP_cplr
                elif (_gate == 'CZ' and _qubit == 'Cplr'):
                    _pulses = self.pulses_CZ_cplr
                elif (_gate == 'iSWAP' and _qubit == 'TQB'):
                    _pulses = self.pulses_iSWAP_tqb
                elif (_gate == 'CZ' and _qubit == 'TQB'):
                    _pulses = self.pulses_CZ_tqb
                else:
                    raise ValueError('invalid %s, %s'%(_gate, _qubit))

                for n, pulse in enumerate(_pulses):
                    if (n>0):
                        break
                    # pulses are indexed from 1 in Labber
                    _str = '%d-%d' % (n + 1, n + 2)
                    num_pulses = int(config.get('Number of pulses, 2QB (%s, %s, %s)'%(_gate, _qubit, _str)))
                    list_pulses = []
                    list_delays = []
                    for i in range(num_pulses):
                        # global parameters
                        pulse_type = config.get('Pulse type #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))
                        if pulse_type == 'Parametric Gate':
                            pulse = getattr(pulses, 'Cosine')(complex=False)
                        else:
                            pulse = getattr(pulses, pulse_type)(complex=False)

                        if config.get('Pulse type #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str)) in ['Optimal']:
                            # spectra
                            if config.get('Assume linear dependence #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str), True):
                                pulse.qubit = None
                            else:
                                pulse.qubit = self.qubits[1]
                            pulse.dfdV = config.get('df/dV #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))
                            pulse.negative_amplitude =  config.get('Negative Amplitude #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))

                            pulse.width = config.get('Width #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))
                            pulse.plateau = config.get('Plateau #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))

                            pulse.F_terms = d[config.get('Fourier terms #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))]
                            # Get Fourier values
                            if d[config.get('Fourier terms #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))] == 4:
                                pulse.F_coeffs = np.array([
                                    config.get('L1 #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str)),
                                    config.get('L2 #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str)),
                                    config.get('L3 #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str)),
                                    config.get('L4 #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))
                                ])
                            elif d[config.get('Fourier terms #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))] == 3:
                                pulse.F_coeffs = np.array([
                                    config.get('L1 #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str)),
                                    config.get('L2 #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str)),
                                    config.get('L3 #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))
                                ])
                            elif d[config.get('Fourier terms #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))] == 2:
                                pulse.F_coeffs = np.array([
                                    config.get('L1 #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str)),
                                    config.get('L2 #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))
                                ])
                            elif d[config.get('Fourier terms #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))] == 1:
                                pulse.F_coeffs = np.array([config.get('L1 #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))])


                            pulse.Hx = config.get('Hx #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))
                            pulse.init_Hz = config.get('Init Hz #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))
                            pulse.final_Hz = config.get('Final Hz #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))
                            pulse.calculate_optimal_waveform()
                        else:
                            # spectra
                            if config.get('Convert from freq to amp #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str), False):
                                pulse.qubit = self.qubits[1] #QB2 = CPLR
                                pulse.init_freq = config.get('Init freq #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))
                                pulse.final_freq = config.get('Final freq #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))
                            else:
                                pulse.qubit = None
                                pulse.init_freq = None
                                pulse.final_freq = None
                            pulse.transduce_pulse = config.get('Transduce Pulse #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str), False)
                            pulse.transduce_file_path = config.get('File Path of Transduce Function #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str),"")
                            if len(pulse.transduce_file_path) == 0:
                                pulse.transduce_file_path = os.path.join(path_currentdir, 'pulse_transduce_func.pickle')
                            
                            if (pulse.transduce_pulse == True):
                                amp_offset = config.get('Amplitude Offset #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str), 0)
                                swap_amplitude = config.get('Parametric Driving Amplitude #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str), 0)
                                data = pickle.load(open(pulse.transduce_file_path, 'rb'))
                                # remove nan data
                                ind_nan = np.argwhere(np.isnan(data['list_swap_freq']))
                                data['cplr_amp'] = np.delete(data['cplr_amp'], ind_nan)
                                data['list_swap_freq'] = np.delete(data['list_swap_freq'], ind_nan)
                                swap_freq_offset =  np.interp(amp_offset, xp = data['cplr_amp'], fp = data['list_swap_freq'])
                                x_range = np.linspace(-1,1,1001)
                                y_range = np.interp(np.linspace(-swap_amplitude, swap_amplitude, 1001), data['list_swap_freq']-swap_freq_offset, data['cplr_amp']-amp_offset)
                                pulse.transduce_func = lambda x: np.interp(x, xp = x_range, fp =  y_range)
                                
                            pulse.truncation_range = config.get('Truncation range #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))
                            pulse.start_at_zero = config.get('Start at zero #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))
                            # pulse-specific parameters
                            pulse.amplitude = config.get('Amplitude #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))
                            pulse.amplitude_opposite = config.get('Opposite Amplitude #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))
                            pulse.width = config.get('Width #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))
                            pulse.width_opposite = config.get('Opposite Width #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))
                            pulse.plateau = config.get('Plateau #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))
                            pulse.plateau_opposite = config.get('Opposite Plateau #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))
                            pulse.frequency = config.get('Frequency #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))
                            pulse.phase = config.get('Phase #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str))
                        list_pulses.append(pulse)

                        if i > 0:
                            list_delays.append(config.get('Pulse delay #%d, 2QB (%s, %s, %s)'%(i+1, _gate, _qubit, _str)))

                    # composite_pulse = pulses.CompositePulse(list_pulses = list_pulses, list_delays = list_delays)
                    make_net_zero = bool(config.get('Make Net-Zero, 2QB (%s, %s, %s)'%(_gate, _qubit, _str), False))
                    net_zero_spacing = config.get('Net-Zero Spacing, 2QB (%s, %s, %s)'%(_gate, _qubit, _str), 0.0)
                    
                    if (make_net_zero == True):
                        composite_pulse = pulses.NetZero_CompositePulse(pulses.CompositePulse(list_pulses = list_pulses, list_delays = list_delays), net_zero_spacing = net_zero_spacing)
                    else:
                        composite_pulse = pulses.CompositePulse(list_pulses = list_pulses, list_delays = list_delays)
                    _pulses[n] = composite_pulse

        for i in [0]:
            self.pulses_CZ_cplr_opposite[i] = self._swap_pulse_polarity(self.pulses_CZ_cplr[i])
            self.pulses_CZ_tqb_opposite[i] = self._swap_pulse_polarity(self.pulses_CZ_tqb[i])

        gates.iSWAP_Cplr.new_angles(
            config.get('FQB Phi Symm, 2QB (iSWAP, Cplr, 1-2)'), config.get('TQB Phi Symm, 2QB (iSWAP, Cplr, 1-2)'),
                config.get('FQB Phi Asymm, 2QB (iSWAP, Cplr, 1-2)'), config.get('TQB Phi Asymm, 2QB (iSWAP, Cplr, 1-2)'),
                        polarity = 'positive')

        # To reduce unnecessary into X&Y Gates as much as possible, we use two different versions of iSWAP gate.
        gates.iSWAP_Cplr_Z_behind.new_angles(
            config.get('FQB Phi, 2QB (iSWAP, Cplr, 1-2)'), config.get('TQB Phi, 2QB (iSWAP, Cplr, 1-2)'))
        gates.iSWAP_Cplr_Z_ahead.new_angles(
            config.get('TQB Phi, 2QB (iSWAP, Cplr, 1-2)'), config.get('FQB Phi, 2QB (iSWAP, Cplr, 1-2)'))


        gates.CZ_Cplr.new_angles(
            config.get('FQB Phi, 2QB (CZ, Cplr, 1-2)'), config.get('TQB Phi, 2QB (CZ, Cplr, 1-2)'), polarity = 'positive')

        gates.CZ_Cplr_opposite.new_angles(
            config.get('FQB Phi, 2QB (CZ, Cplr, 1-2)'), config.get('TQB Phi, 2QB (CZ, Cplr, 1-2)'), polarity = 'negative')
        gates.CZ_Cplr.set_phase_offsets(
            [config.get('FQB Phi Offset, 2QB (CZ, Cplr, 1-2)'), 
            0, 
            config.get('TQB Phi Offset, 2QB (CZ, Cplr, 1-2)')])
        

        # predistortion
        self.perform_predistortion = config.get('Predistort waveforms', False)
        # update all predistorting objects
        for p in self._predistortions:
            p.set_parameters(config)


        # Z predistortion
        self.perform_predistortion_z = config.get('Predistort Z')
        for p in self._predistortions_z:
            p.set_parameters(config)

        # predistortion for Z crosstalk-compensation
        self._predistortions_z2_z3.set_parameters(config)

        # crosstalk
        self.compensate_crosstalk = config.get('Compensate cross-talk', False)
        self._crosstalk.set_parameters(config)

        # microwave-crosstalk
        self.compensate_microwave_crosstalk = config.get('Compensate microwave cross-talk', False)
        self._microwave_crosstalk.set_parameters(config)

        # gate switch waveform
        self.generate_gate_switch = config.get('Generate gate')
        self.uniform_gate = config.get('Uniform gate')
        self.gate_delay = config.get('Gate delay')
        self.gate_overlap = config.get('Gate overlap')
        self.minimal_gate_time = config.get('Minimal gate time')

        # filters
        self.use_gate_filter = config.get('Filter gate waveforms', False)
        self.gate_filter = config.get('Gate filter', 'Kaiser')
        self.gate_filter_size = int(config.get('Gate - Filter size', 5))
        self.gate_filter_kaiser_beta = config.get(
            'Gate - Kaiser beta', 14.0)
        self.use_z_filter = config.get('Filter Z waveforms', False)
        self.z_filter = config.get('Z filter', 'Kaiser')
        self.z_filter_size = int(config.get('Z - Filter size', 5))
        self.z_filter_kaiser_beta = config.get(
            'Z - Kaiser beta', 14.0)

        # readout
        self.readout_match_main_size = config.get(
            'Match main sequence waveform size')
        self.readout_i_offset = config.get('Readout offset - I')
        self.readout_q_offset = config.get('Readout offset - Q')
        self.readout_trig_generate = config.get('Generate readout trig')
        self.readout_trig_amplitude = config.get('Readout trig amplitude')
        self.readout_trig_duration = config.get('Readout trig duration')
        self.readout_predistort = config.get('Predistort readout waveform')
        self.readout.set_parameters(config)

        # get readout pulse parameters
        phases = 2 * np.pi * np.array([
            0.8847060, 0.2043214, 0.9426104, 0.6947334, 0.8752361, 0.2246747,
            0.6503154, 0.7305004, 0.1309068
        ])
        for n, pulse in enumerate(self.pulses_readout):
            # pulses are indexed from 1 in Labber
            m = n + 1
            pulse = (getattr(pulses,
                             config.get('Readout pulse type'))(complex=True))
            pulse.truncation_range = config.get('Readout truncation range')
            pulse.start_at_zero = config.get('Readout start at zero')
            pulse.iq_skew = config.get('Readout IQ skew') * np.pi / 180
            pulse.iq_ratio = config.get('Readout I/Q ratio')

            if config.get('Distribute readout phases'):
                pulse.phase = phases[n]
            else:
                pulse.phase = 0

            if config.get('Uniform readout pulse shape'):
                pulse.width = config.get('Readout width')
                pulse.plateau = config.get('Readout duration')
            else:
                pulse.width = config.get('Readout width #%d' % m)
                pulse.plateau = config.get('Readout duration #%d' % m)

            if config.get('Uniform readout amplitude') is True:
                pulse.amplitude = config.get('Readout amplitude')
            else:
                pulse.amplitude = config.get('Readout amplitude #%d' % (n + 1))

            pulse.frequency = config.get('Readout frequency #%d' % m)
            self.pulses_readout[n] = pulse

        # Delays
        self.wave_xy_delays = np.zeros(self.n_qubit)
        self.wave_z_delays = np.zeros(self.n_qubit)
        for n in range(self.n_qubit):
            m = n + 1
            self.wave_xy_delays[n] = config.get('Qubit %d XY Delay' % m)
            self.wave_z_delays[n] = config.get('Qubit %d Z Delay' % m)


if __name__ == '__main__':
    pass
