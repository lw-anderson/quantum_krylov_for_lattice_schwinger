from operators.coupled_quartic_oscillators_hamiltonian import CoupledQuarticOscillatorsHamiltonian
from operators.hamiltonian_base_class import HamiltonianBaseClass
from operators.phi4_model_hamiltonian import Phi4ModelHamiltonian
from operators.schwinger_gauged_hamiltonian import SchwingerGaugedHamiltonian
from operators.schwinger_hamiltonian import SchwingerHamiltonian


class HamiltonianFactory:
    def __init__(self, **kwargs):
        self.args = kwargs

    def get(self) -> HamiltonianBaseClass:

        if self.args["type"] == "oscillators":
            return CoupledQuarticOscillatorsHamiltonian(self.args["n"], self.args["d"], self.args["couplings"],
                                                        self.args["anharmonicities"])
        elif self.args["type"] == "phi4-model":
            return Phi4ModelHamiltonian(self.args["dims"], self.args["n"], self.args["d"], self.args["a"],
                                        self.args["m"], self.args["lam"])
        elif self.args["type"] == "schwinger-model":
            return SchwingerHamiltonian(self.args["n"], self.args["d"], self.args["mu"], self.args["x"], self.args["k"])
        elif self.args["type"] == "schwinger-model-gauged":
            return SchwingerGaugedHamiltonian(self.args["n"], self.args["mu"], self.args["x"], self.args["k"])
        else:
            raise ValueError("Invalid Hamiltonian type.")
