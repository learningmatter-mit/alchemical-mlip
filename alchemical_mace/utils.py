import os
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout

from ase import Atoms


@contextmanager
def suppress_print(out: bool = True, err: bool = False):
    """Suppress stdout and/or stderr."""

    with ExitStack() as stack:
        devnull = stack.enter_context(open(os.devnull, "w"))
        if out:
            stack.enter_context(redirect_stdout(devnull))
        if err:
            stack.enter_context(redirect_stderr(devnull))
        yield


# From CHGNet
def upper_triangular_cell(atoms: Atoms):
    """Transform to upper-triangular cell."""
    import numpy as np
    from ase.md.npt import NPT

    if NPT._isuppertriangular(atoms.get_cell()):
        return

    a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
    angles = np.radians((alpha, beta, gamma))
    sin_a, sin_b, _sin_g = np.sin(angles)
    cos_a, cos_b, cos_g = np.cos(angles)
    cos_p = (cos_g - cos_a * cos_b) / (sin_a * sin_b)
    cos_p = np.clip(cos_p, -1, 1)
    sin_p = (1 - cos_p**2) ** 0.5

    new_basis = [
        (a * sin_b * sin_p, a * sin_b * cos_p, a * cos_b),
        (0, b * sin_a, b * cos_a),
        (0, 0, c),
    ]

    atoms.set_cell(new_basis, scale_atoms=True)
