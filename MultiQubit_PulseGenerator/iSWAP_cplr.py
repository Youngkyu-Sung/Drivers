from sequence import Sequence
import gates


class CustomSequence(Sequence):
    def generate_sequence(self, config):
        tunable_coupler_gate = gates.CompositeGate(n_qubit=3)
        tunable_coupler_gate.add_gate([gates.I, gates.Zp, gates.Zp])

        self.add_gate([0, 1, 2], tunable_coupler_gate)