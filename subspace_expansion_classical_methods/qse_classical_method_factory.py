from subspace_expansion_classical_methods.krylov_basis import QSEKrylovBasis
from subspace_expansion_classical_methods.qse_base_class import QSEBaseClass


class QSEClassicalMethodFactory:
    def __init__(self, **kwargs):
        self.args = kwargs

    def get(self) -> QSEBaseClass:

        if self.args["method"] == "krylov":
            return QSEKrylovBasis(self.args["num_qubits"], self.args["basis_size"], self.args["low_mem_mul"])

        else:
            raise ValueError("Invalid subspace method.")
