from typing import List

import numpy as np
from scipy.sparse import kron, csr_matrix
from sympy import Matrix
from sympy.physics.quantum import TensorProduct

floattype = np.float64


def multi_tensor_product(matrices: Matrix) -> Matrix:
    """
    Calculate tensor product of arbitrary number of matrices.
    matrices: List of sympy matrices to tensor product together
    """
    prod = TensorProduct(matrices[0], matrices[1])
    for mat in matrices[2:]:
        prod = TensorProduct(prod, mat)
    return prod


def sparse_multi_tensor_product(matrices: List[csr_matrix]) -> csr_matrix:
    """
    Calculate tensor product of arbitrary number of ma.
    matrices: List of scipy matrices to tensor product together
    """
    if len(matrices) == 1:
        return matrices[0]
    prod = kron(matrices[0], matrices[1])
    prod.eliminate_zeros()
    if len(matrices) > 2:
        for mat in matrices[2:]:
            assert type(mat) is csr_matrix, Exception("matrices should be list of sparse matrices")
            prod = kron(prod, mat)
            prod.eliminate_zeros()
            prod = prod.astype(floattype)
    prod = csr_matrix(prod.astype(floattype))
    prod.eliminate_zeros()
    return prod
