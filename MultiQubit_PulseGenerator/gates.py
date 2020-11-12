#!/usr/bin/env python3
from copy import copy
import numpy as np
import logging
from sequence import Step
log = logging.getLogger('LabberDriver')

# TODO remove Step dep from CompositeGate


class BaseGate:
    """Base class for a qubit gate.

    """

    def get_adjusted_pulse(self, pulse):
        pulse = copy(pulse)
        return pulse

    def __repr__(self):
        return self.__str__()


class OneQubitGate(BaseGate):
    def number_of_qubits(self):
        return 1


class TwoQubitGate(BaseGate):
    def number_of_qubits(self):
        return 2



class SingleQubitXYRotation(OneQubitGate):
    """Single qubit rotations around the XY axes.

    Angles defined as in https://en.wikipedia.org/wiki/Bloch_sphere.

    Parameters
    ----------
    phi : float
        Rotation axis.
    theta : float
        Roation angle.

    """

    def __init__(self, phi, theta, name=None):
        self.phi = phi
        self.theta = theta
        self.name = name

    def get_adjusted_pulse(self, pulse):
        pulse = copy(pulse)
        pulse.phase = self.phi
        # pi pulse correspond to the full amplitude
        # if self.theta == 0.5 * np.pi:
        #     pulse.amplitude = pulse.amplitude_half # This is to fine-tune pi/2 pulses..
        # elif self.theta == -0.5 * np.pi:
        #     pulse.amplitude = -pulse.amplitude_half # This is to fine-tune pi/2 pulses..
        # else:
        pulse.amplitude *= self.theta / np.pi
        return pulse

    def __str__(self):
        if self.name is None:
            return "XYPhi={:+.6f}theta={:+.6f}".format(self.phi, self.theta)
        else:
            return self.name

    def __eq__(self, other):
        threshold = 1e-10
        if not isinstance(other, SingleQubitXYRotation):
            return False
        if np.abs(self.phi - other.phi) > threshold:
            return False
        if np.abs(self.theta - other.theta) > threshold:
            return False
        return True

# class SingleQubitXYRotation_AUX(OneQubitGate):
#     """Single qubit rotations around the XY axes.

#     Angles defined as in https://en.wikipedia.org/wiki/Bloch_sphere.

#     Parameters
#     ----------
#     phi : float
#         Rotation axis.
#     theta : float
#         Roation angle.

#     """

#     def __init__(self, phi, theta, name=None):
#         self.phi = phi
#         self.theta = theta
#         self.name = name

#     def get_adjusted_pulse(self, pulse):
#         pulse = copy(pulse)
#         pulse.phase = self.phi
#         # pi pulse correspond to the full amplitude
#         pulse.amplitude *= self.theta / np.pi
#         return pulse

#     def __str__(self):
#         if self.name is None:
#             return "XYPhi={:+.6f}theta={:+.6f}".format(self.phi, self.theta)
#         else:
#             return self.name

#     def __eq__(self, other):
#         threshold = 1e-10
#         if not isinstance(other, SingleQubitXYRotation_AUX):
#             return False
#         if np.abs(self.phi - other.phi) > threshold:
#             return False
#         if np.abs(self.theta - other.theta) > threshold:
#             return False
#         return True


class SingleQubitXYRotation_12(OneQubitGate):
    """Single qubit rotations around the XY axes for (1-2) transition.

    Angles defined as in https://en.wikipedia.org/wiki/Bloch_sphere.

    Parameters
    ----------
    phi : float
        Rotation axis.
    theta : float
        Roation angle.

    """

    def __init__(self, phi, theta, name=None):
        self.phi = phi
        self.theta = theta
        self.name = name

    def get_adjusted_pulse(self, pulse):
        pulse = copy(pulse)
        pulse.phase = self.phi
        # pi pulse correspond to the full amplitude
        pulse.amplitude *= self.theta / np.pi
        return pulse

    def __str__(self):
        if self.name is None:
            return "XYPhi={:+.6f}theta={:+.6f}".format(self.phi, self.theta)
        else:
            return self.name

    def __eq__(self, other):
        threshold = 1e-10
        if not isinstance(other, SingleQubitXYRotation_12):
            return False
        if np.abs(self.phi - other.phi) > threshold:
            return False
        if np.abs(self.theta - other.theta) > threshold:
            return False
        return True

class SingleQubitZRotation(OneQubitGate):
    """Single qubit rotation around the Z axis.

    Parameters
    ----------
    theta : float
        Roation angle.

    """

    def __init__(self, theta, name=None):
        self.theta = theta
        self.name = name

    def get_adjusted_pulse(self, pulse):
        pulse = copy(pulse)
        # pi pulse correspond to the full amplitude
        pulse.amplitude *= self.theta / np.pi
        return pulse

    def __str__(self):
        if self.name is None:
            return "Ztheta={:+.2f}".format(self.theta)
        else:
            return self.name

    def __eq__(self, other):
        threshold = 1e-10
        if not isinstance(other, SingleQubitZRotation):
            return False
        if np.abs(self.theta - other.theta) > threshold:
            return False
        return True


class Separator(OneQubitGate):
    """Separator.

    Does nothing to the qubit. This is to separate cliffords in iSWAP RB

    Parameters
    ----------

    """

    def __init__(self,  name=None):
        self.name = name
    def __str__(self):
        if self.name is None:
            return "Separator"
        else:
            return self.name
    def __str__(self):
        return "Separator"

class IdentityGate(OneQubitGate):
    """Identity gate.

    Does nothing to the qubit. The width can be specififed to
    implement a delay in the sequence. If no width is given, the identity gate
    inherits the width of the given pulse.

    Parameters
    ----------
    width : float
        Width of the I gate in seconds,
        None uses the XY width (the default is None).

    """

    def __init__(self, width=None):
        self.width = width

    def get_adjusted_pulse(self, pulse):
        pulse = copy(pulse)
        pulse.amplitude = 0
        pulse.use_drag = False  # Avoids bug
        if self.width is not None:
            pulse.width = 0
            pulse.plateau = self.width
        return pulse

    def __str__(self):
        return "I"


class VirtualZGate(OneQubitGate):
    """Virtual Z Gate."""

    def __init__(self, theta, name=None):
        self.theta = theta
        self.name = name

    def __eq__(self, other):
        threshold = 1e-10
        if not isinstance(other, VirtualZGate):
            return False
        if np.abs(self.theta - other.theta) > threshold:
            return False
        return True

    def __str__(self):
        if self.name is None:
            return "VZtheta={:+.2f}".format(self.theta)
        else:
            return self.name


class EulerZGate(OneQubitGate):
    """Euler Z Gate."""

    def __init__(self, theta, name=None):
        self.theta = theta
        self.name = name

    def __eq__(self, other):
        threshold = 1e-10
        if not isinstance(other, VirtualZGate):
            return False
        if np.abs(self.theta - other.theta) > threshold:
            return False
        return True

    def __str__(self):
        if self.name is None:
            return "EulerZ={:+.2f}".format(self.theta)
        else:
            return self.name
class CPHASE(TwoQubitGate):
    """ CPHASE gate. """


class ReadoutGate(OneQubitGate):
    """Readouts the qubit state."""
    def __str__(self):
        return 'Readout'

class CustomGate(BaseGate):
    """A gate using a given :obj:`Pulse`.

    Parameters
    ----------
    pulse : :obj:`Pulse`
        The corresponding pulse.

    """

    def __init__(self, pulse):
        self.pulse = pulse


class RabiGate(SingleQubitXYRotation):
    """Creates the Rabi gate used in the spin-locking sequence.

    Parameters
    ----------
    amplitude : Amplitude of the pulse
    plateau : The duration of the pulse.
    phase : Phase of the Rabi gate. 0 corresponds to rotation around X axis.
    """

    def __init__(self, amplitude, plateau, phase):
        self.amplitude = amplitude
        self.plateau = plateau
        self.phase = phase

    def get_adjusted_pulse(self, pulse):
        pulse = copy(pulse)
        pulse.amplitude = self.amplitude
        pulse.plateau = self.plateau
        pulse.phase = self.phase
        return pulse


class ZGate_Cplr_iSWAP(OneQubitGate):
    """Z pulses applied to a Coupler, to implement a iSWAP gate"""
    def __init__(self, name='ZGate_Cplr_iSWAP'):
        self.name = name

    def get_adjusted_pulse(self, pulse):
        self.pulse = copy(pulse)
        return self.pulse

    def __str__(self):
        return "ZGate_Cplr_iSWAP"

    def __repr__(self):
        return self.__str__()


class ZGate_TQB_iSWAP(OneQubitGate):
    """Z pulses applied to a tunable qubit, to implement a iSWAP gate"""
    def __init__(self, name='ZGate_TQB_iSWAP'):
        self.name = name

    def get_adjusted_pulse(self, pulse):
        pulse = copy(pulse)
        return pulse
    """Tunable Qubit (2QB gate using Cplr)"""

    def __str__(self):
        return "ZGate_TQB_iSWAP"
    def __repr__(self):
        return self.__str__()



class ZGate_Cplr_CZ(OneQubitGate):
    """Z pulses applied to a Coupler, to implement a CZ gate"""
    def __init__(self, name='ZGate_Cplr_CZ'):
        # super().__init__(theta=np.pi)
        self.name = name

    def get_adjusted_pulse(self, pulse):
        self.pulse = copy(pulse)
        return self.pulse

    def __str__(self):
        return "ZGate_Cplr_CZ"
    def __repr__(self):
        return self.__str__()


class ZGate_Cplr_CZ_opposite(OneQubitGate):
    """Z pulses applied to a Coupler, to implement a CZ gate"""
    def __init__(self, name='ZGate_Cplr_CZ_opposite'):
        # super().__init__(theta=np.pi)
        self.name = name

    def get_adjusted_pulse(self, pulse):
        self.pulse = copy(pulse)
        return self.pulse

    def __str__(self):
        return "ZGate_Cplr_CZ_opposite"
    def __repr__(self):
        return self.__str__()

class ZGate_TQB_CZ(OneQubitGate):
    """Z pulses applied to a tunable qubit, to implement a CZ gate"""
    def __init__(self, name='ZGate_TQB_CZ'):
        # super().__init__(theta=np.pi)
        self.name = name

    def get_adjusted_pulse(self, pulse):
        pulse = copy(pulse)
        return pulse

    def __str__(self):
        return "ZGate_TQB_CZ"
    def __repr__(self):
        return self.__str__()

class ZGate_TQB_CZ_opposite(OneQubitGate):
    """Z pulses applied to a tunable qubit, to implement a CZ gate"""
    def __init__(self, name='ZGate_TQB_CZ_opposite'):
        # super().__init__(theta=np.pi)
        self.name = name

    def get_adjusted_pulse(self, pulse):
        pulse = copy(pulse)
        return pulse

    def __str__(self):
        return "ZGate_TQB_CZ_opposite"
    def __repr__(self):
        return self.__str__()

class CompositeGate:
    """Multiple gates in one object.

    Parameters
    ----------
    n_qubit : int
        Number of qubits involved in the composite gate.

    Attributes
    ----------
    sequence : list of :Step:
        Holds the gates involved.

    """

    def __init__(self, n_qubit, name='None'):
        self.n_qubit = n_qubit
        self.sequence = []
        self.name = name

    def add_gate(self, gate, qubit=None):
        """Add a set of gates to the given qubit.

        For the qubits with no specificied gate, an IdentityGate will be given.
        The length of the step is given by the longest pulse.

        Parameters
        ----------
        qubit : int or list of int
            The qubit(s) to add the gate(s) to.
        gate : :obj:`BaseGate` or list of :obj:`BaseGate`
            The gate(s) to add.
        """
        if qubit is None:
            if self.n_qubit == 1:
                qubit = 0
            else:
                qubit = [n for n in range(self.n_qubit)]

        step = Step()
        if isinstance(gate, list):
            if len(gate) == 1:
                raise ValueError(
                    "For single gates, don't provide gate as a list.")
            if not isinstance(qubit, list):
                raise ValueError(
                    """Please provide qubit indices as a list when adding more
                    than one gate.""")
            if len(gate) != len(qubit):
                raise ValueError(
                    "Length of gate list must equal length of qubit list.")

            for q, g in zip(qubit, gate):
                step.add_gate(q, g)

        else:
            if gate.number_of_qubits() > 1:
                if not isinstance(qubit, list):
                    raise ValueError(
                        """Please provide qubit list for gates with more than
                        one qubit.""")
            else:
                if not isinstance(qubit, int):
                    raise ValueError(
                        "For single gates, give qubit as int (not list).")
            step.add_gate(qubit, gate)

        self.sequence.append(step)

    def number_of_qubits(self):
        return self.n_qubit

    def __len__(self):
        return len(self.sequence)

    def __str__(self):
        if self.name is not None:
            return self.name
        else:
            super().__str__()

    def __repr__(self):
        return self.__str__()


class CPHASE_with_1qb_phases(CompositeGate):
    """CPHASE gate followed by single qubit Z rotations.

    Parameters
    ----------
    phi1 : float
        Z rotation angle for qubit 1.
    phi2 : float
        Z rotation angle for qubit 2.

    """

    def __init__(self, phi1, phi2):
        super().__init__(n_qubit=2)
        self.add_gate(CPHASE())
        self.add_gate([VirtualZGate(phi1), VirtualZGate(phi2)])

    def new_angles(self, phi1, phi2):
        """Update the angles of the single qubit rotations.

        Parameters
        ----------
        phi1 : float
            Z rotation angle for qubit 1.
        phi2 : float
            Z rotation angle for qubit 2.

        """
        self.__init__(phi1, phi2)

    def __str__(self):
        return "CZ"

class iSWAP_Cplr_with_Z_behind(CompositeGate):
    """
        iSWAP gate - single qubit Z rotations.

        Parameters
        ----------
        phi1 : float
            Z rotation angle for qubit 1.
        phi2 : float
            Z rotation angle for qubit 2.
        
    """
    def __init__(self, phi1, phi2):
        super().__init__(n_qubit=3, name = 'iSWAP_Cplr')
        self.phi1 = phi1
        self.phi2 = phi2
        self.add_gate([IdentityGate(width = 0), ZGate_Cplr_iSWAP(), ZGate_TQB_iSWAP()])
        if self.phi1 == 0 and self.phi2 == 0:
            pass
        else:
            self.add_gate([EulerZGate(self.phi1), IdentityGate(width = 0), EulerZGate(self.phi2)])


    def new_angles(self, phi1, phi2):
        """Update the angles of the single qubit rotations.

        Parameters
        ----------
        phi1 : float
            Z rotation angle for qubit 1.
        phi2 : float
            Z rotation angle for qubit 2.

        """
        self.__init__(phi1, phi2)

    def __str__(self):
        return "iSWAP_Cplr"
    def __repr__(self):
        return self.__str__()

class iSWAP_Cplr_with_Z_ahead(CompositeGate):
    """
        single qubit Z rotations - iSWAP

        Parameters
        ----------
        phi1 : float
            Z rotation angle for qubit 1.
        phi2 : float
            Z rotation angle for qubit 2.

    """
    def __init__(self, phi1, phi2):
        super().__init__(n_qubit=3, name = 'iSWAP_Cplr')
        self.phi1 = phi1
        self.phi2 = phi2
        self.add_gate([EulerZGate(self.phi1), IdentityGate(width = 0), EulerZGate(self.phi2)])
        if self.phi1 == 0 and self.phi2 == 0:
            pass
        else:
            self.add_gate([IdentityGate(width = 0), ZGate_Cplr_iSWAP(), ZGate_TQB_iSWAP()])


    def new_angles(self, phi1, phi2):
        """Update the angles of the single qubit rotations.

        Parameters
        ----------
        phi1 : float
            Z rotation angle for qubit 1.
        phi2 : float
            Z rotation angle for qubit 2.

        """
        self.__init__(phi1, phi2)

    def __str__(self):
        return "iSWAP_Cplr"
    def __repr__(self):
        return self.__str__()


class iSWAP_Cplr_with_1qb_phases(CompositeGate):
    """iSWAP Coupler gate followed by single qubit Z rotations.

    Parameters
    ----------
    phi1 : float
        Z rotation angle for qubit 1.
    phi2 : float
        Z rotation angle for qubit 2.

    """
    def __init__(self, phi1_Symm, phi2_Symm, phi1_Asymm, phi2_Asymm, polarity):
        super().__init__(n_qubit=3, name = 'iSWAP_Cplr')
        # log.info('with virtual z')
        # self.add_gate([VirtualZGate(np.pi), IdentityGate(width = 0), VirtualZGate(np.pi)]) # due to negative g
        self.phi1_Symm = phi1_Symm
        self.phi2_Symm = phi2_Symm
        self.phi1_Asymm = phi1_Asymm
        self.phi2_Asymm = phi2_Asymm
        
        self.add_gate([IdentityGate(width = 0), ZGate_Cplr_iSWAP(), ZGate_TQB_iSWAP()])
        # self.add_gate([VirtualZGate(phi1_Symm), IdentityGate(width = 0), VirtualZGate(phi2_Symm)])
        # self.add_gate([VirtualZGate(phi1_Asymm), IdentityGate(width = 0), VirtualZGate(phi2_Asymm)])


    def new_angles(self, phi1_Symm, phi2_Symm, phi1_Asymm, phi2_Asymm, polarity):
        """Update the angles of the single qubit rotations.

        Parameters
        ----------
        phi1 : float
            Z rotation angle for qubit 1.
        phi2 : float
            Z rotation angle for qubit 2.

        """
        self.__init__(phi1_Symm, phi2_Symm, phi1_Asymm, phi2_Asymm, polarity)

    def __str__(self):
        return "iSWAP_Cplr"
    def __repr__(self):
        return self.__str__()

class CZ_Cplr_with_1qb_phases(CompositeGate):
    """iSWAP Coupler gate followed by single qubit Z rotations.

    Parameters
    ----------
    phi1 : float
        Z rotation angle for qubit 1.
    phi2 : float
        Z rotation angle for qubit 2.

    """

    def __init__(self, phi1, phi2, polarity):
        super().__init__(n_qubit=3, name = 'CZ_Cplr')
        if polarity == 'positive':
            self.add_gate([IdentityGate(width = 0), ZGate_Cplr_CZ(), ZGate_TQB_CZ()])
        else:
            self.add_gate([IdentityGate(width = 0), ZGate_Cplr_CZ_opposite(), ZGate_TQB_CZ_opposite()])

        self.add_gate([VirtualZGate(phi1), IdentityGate(width = 0), VirtualZGate(phi2)])

    phi_offsets = [0,0,0]
    def set_phase_offsets(self, phi_offsets):
        self.phi_offsets = phi_offsets
    def new_angles(self, phi1, phi2, polarity):
        """Update the angles of the single qubit rotations.

        Parameters
        ----------
        phi1 : float
            Z rotation angle for qubit 1.
        phi2 : float
            Z rotation angle for qubit 2.

        """
        self.__init__(phi1, phi2, polarity)

    def __str__(self):
        return "CZ_Cplr"
    def __repr__(self):
        return self.__str__()

I = IdentityGate(width=None)
I0 = IdentityGate(width=0)
Ilong = IdentityGate(width=75e-9)

# X gates
Xp = SingleQubitXYRotation(phi=0, theta=np.pi, name='Xp')
Xm = SingleQubitXYRotation(phi=0, theta=-np.pi, name='Xm')
X2p = SingleQubitXYRotation(phi=0, theta=np.pi / 2, name='X2p')
X2m = SingleQubitXYRotation(phi=0, theta=-np.pi / 2, name='X2m')


# X2p_aux = SingleQubitXYRotation_AUX(phi=0, theta=np.pi / 2, name='X2p') #auxillary
# X2m_aux = SingleQubitXYRotation_AUX(phi=0, theta=-np.pi / 2, name='X2m') #auxillary

# Y gates
Yp = SingleQubitXYRotation(phi=np.pi / 2, theta=np.pi, name='Yp')
Ym = SingleQubitXYRotation(phi=np.pi / 2, theta=-np.pi, name='Ym')
Y2m = SingleQubitXYRotation(phi=np.pi / 2, theta=-np.pi / 2, name='Y2m')
Y2p = SingleQubitXYRotation(phi=np.pi / 2, theta=np.pi / 2, name='Y2p')

# Y2m_aux = SingleQubitXYRotation_AUX(phi=np.pi / 2, theta=-np.pi / 2, name='Y2m') #auxillary
# Y2p_aux = SingleQubitXYRotation_AUX(phi=np.pi / 2, theta=np.pi / 2, name='Y2p') #auxillary

# (X+Y)/2 gates (for cross-entropy benchmarking)
W2p = SingleQubitXYRotation(phi= np.pi / 4, theta = np.pi / 2, name='W2p')



# X gates (1-2)
Xp_12 = SingleQubitXYRotation_12(phi=0, theta=np.pi, name='Xp_12')
Xm_12 = SingleQubitXYRotation_12(phi=0, theta=-np.pi, name='Xm_12')
X2p_12 = SingleQubitXYRotation_12(phi=0, theta=np.pi / 2, name='X2p_12')
X2m_12 = SingleQubitXYRotation_12(phi=0, theta=-np.pi / 2, name='X2m_12')

# Y gates (1-2)
Yp_12 = SingleQubitXYRotation_12(phi=np.pi / 2, theta=np.pi, name='Yp_12')
Ym_12 = SingleQubitXYRotation_12(phi=np.pi / 2, theta=-np.pi, name='Ym_12')
Y2m_12 = SingleQubitXYRotation_12(phi=np.pi / 2, theta=-np.pi / 2, name='Y2m_12')
Y2p_12 = SingleQubitXYRotation_12(phi=np.pi / 2, theta=np.pi / 2, name='Y2p_12')

# Z gates
Zp = SingleQubitZRotation(np.pi, name='Zp')
Z2p = SingleQubitZRotation(np.pi / 2, name='Z2p')
Zm = SingleQubitZRotation(-np.pi, name='Zm')
Z2m = SingleQubitZRotation(-np.pi / 2, name='Z2m')

# Virtual Z gates
VZp = VirtualZGate(np.pi, name='VZp')
VZ2p = VirtualZGate(np.pi / 2, name='VZ2p')
VZm = VirtualZGate(-np.pi, name='VZm')
VZ2m = VirtualZGate(np.pi / 2, name='VZ2m')

# Euler-angle Z gates
EulerZ = EulerZGate(0, name='EulerZ') 
Sep = Separator(name='Separator')
# two-qubit gates
CPh = CPHASE()

# Composite gates
CZEcho = CompositeGate(n_qubit=2)
CZEcho.add_gate([X2p, I])
CZEcho.add_gate(CPh)
CZEcho.add_gate([Xp, Xp])
CZEcho.add_gate(CPh)
CZEcho.add_gate([X2p, Xp])

H = CompositeGate(n_qubit=1, name='H')
H.add_gate(VZp)
H.add_gate(Y2p)

CZ = CPHASE_with_1qb_phases(
    0, 0)  # Start with 0, 0 as the single qubit phase shifts.

CNOT = CompositeGate(n_qubit=2, name='CNOT')
CNOT.add_gate(H, 1)
CNOT.add_gate(CZ, [0, 1])
CNOT.add_gate(H, 1)



iSWAP_Cplr_Z_ahead = iSWAP_Cplr_with_Z_ahead(0, 0) # start with 0, 0 as the single qubit phase shifts
iSWAP_Cplr_Z_behind = iSWAP_Cplr_with_Z_behind(0, 0) # start with 0, 0 as the single qubit phase shifts

iSWAP_Cplr = iSWAP_Cplr_with_1qb_phases(0, 0, 0, 0, polarity = 'positive') # start with 0, 0 as the single qubit phase shifts
iSWAP_Cplr_opposite = iSWAP_Cplr_with_1qb_phases(0, 0, 0, 0, polarity = 'negative') # start with 0, 0 as the single qubit phase shifts

CZ_Cplr = CZ_Cplr_with_1qb_phases(0, 0, polarity = 'positive') # start with 0, 0 as the single qubit phase shifts
CZ_Cplr_opposite = CZ_Cplr_with_1qb_phases(0, 0, polarity = 'negative') # start with 0, 0 as the single qubit phase shifts
# CplrGate = CustomGate()
if __name__ == '__main__':
    pass
