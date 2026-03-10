"""
Recurrence analysis for detecting hidden dynamical structure.

Constructs recurrence matrices and computes recurrence quantification
analysis (RQA) statistics from phase-space trajectories.
"""

import numpy as np


class RecurrenceAnalysis:
    """Recurrence-plot construction and quantification.

    Parameters
    ----------
    phase_space : np.ndarray
        Array of shape ``(N, d)`` — a phase-space trajectory
        (e.g. from :class:`~nonlinear.core.embedding.PhaseSpace`).
    """

    def __init__(self, phase_space):
        self.ps = np.atleast_2d(phase_space)

    def recurrence_matrix(self, epsilon):
        """Binary recurrence matrix *R[i,j] = 1* iff distance < *epsilon*.

        Parameters
        ----------
        epsilon : float
            Recurrence threshold.

        Returns
        -------
        np.ndarray
            Boolean matrix of shape ``(N, N)``.
        """
        N = len(self.ps)
        # Vectorised pairwise distances
        diff = self.ps[:, np.newaxis, :] - self.ps[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        return dist < epsilon

    def recurrence_rate(self, epsilon):
        """Fraction of recurrent points (excluding the main diagonal).

        Parameters
        ----------
        epsilon : float
            Recurrence threshold.

        Returns
        -------
        float
            Recurrence rate in [0, 1].
        """
        R = self.recurrence_matrix(epsilon)
        N = len(R)
        np.fill_diagonal(R, False)
        return R.sum() / (N * (N - 1))

    def determinism(self, epsilon, l_min=2):
        """Fraction of recurrent points forming diagonal lines >= *l_min*.

        A proxy for the predictability / determinism of the system.

        Parameters
        ----------
        epsilon : float
            Recurrence threshold.
        l_min : int
            Minimum diagonal line length (default 2).

        Returns
        -------
        float
            Determinism in [0, 1].
        """
        R = self.recurrence_matrix(epsilon)
        N = len(R)
        np.fill_diagonal(R, False)

        total_recurrence = R.sum()
        if total_recurrence == 0:
            return 0.0

        # Scan upper-triangle diagonals for line lengths
        diag_sum = 0
        for k in range(1, N):
            diag = np.diag(R, k)
            line_len = 0
            for val in diag:
                if val:
                    line_len += 1
                else:
                    if line_len >= l_min:
                        diag_sum += line_len
                    line_len = 0
            if line_len >= l_min:
                diag_sum += line_len

        # Count both upper and lower triangles
        return (2 * diag_sum) / total_recurrence
