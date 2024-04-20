from unittest import TestCase

from subspace_expansion_classical_methods.krylov_basis import QSEKrylovBasis
from subspace_expansion_classical_methods.qse_classical_method_factory import QSEClassicalMethodFactory


class TestQSEClassicalMethodFactory(TestCase):
    def test_get_krylov_basis(self):
        fact = QSEClassicalMethodFactory(method="krylov", num_qubits=2, basis_size=2, low_mem_mul=False)
        qse = fact.get()
        self.assertIsInstance(qse, QSEKrylovBasis)

    def test_invalid_qse_method(self):
        fact = QSEClassicalMethodFactory(method="invalid qse method")
        self.assertRaises(ValueError, fact.get)
