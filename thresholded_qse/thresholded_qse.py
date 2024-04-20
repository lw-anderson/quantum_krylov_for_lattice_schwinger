import numpy as np


class ThresholdingError(Exception):
    pass


def project_onto_subspace(S: np.ndarray, H: np.ndarray, epsilon: float) -> (np.ndarray, np.ndarray):
    proj = projector_threshold_space(S, epsilon)

    projected_S = proj.T @ S @ proj
    projected_H = proj.T @ H @ proj

    # print(f"threshold = {epsilon}, shapes {S.shape}->{projected_S.shape}")
    return projected_S, projected_H


def projector_threshold_space(S: np.ndarray, epsilon: float) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    indices = np.where(np.abs(eigenvalues) > epsilon)[0]
    selected_eigenvectors = eigenvectors[:, indices]

    # Constructing full (normalised?) projector. Rather than what Kirby does.
    # v = selected_eigenvectors
    # proj = v @ np.linalg.inv(v.T.conj() @ v) @ v.T.conj()

    # TODO: Find better way of dealing with very small negative eigenvalues.
    if epsilon == 0.:
        selected_eigenvectors = eigenvectors

    proj = selected_eigenvectors

    return proj
