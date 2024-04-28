from typing import Sequence, Tuple

import ase
import numpy as np
import torch
import torch.nn.functional as F
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import ExpCellFilter
from ase.optimize import FIRE
from ase.stress import full_3x3_to_voigt_6_stress
from mace import data
from mace.calculators import mace_mp
from mace.tools import torch_geometric

from alchemical_mace.model import (
    AlchemicalPair,
    AlchemyManager,
    alchemical_mace_mp,
    get_z_table_and_r_max,
)

################################################################################
# Alchemical MACE calculator
################################################################################


class AlchemicalMACECalculator(Calculator):
    """
    Alchemical MACE calculator for ASE.
    """

    def __init__(
        self,
        atoms: ase.Atoms,
        alchemical_pairs: Sequence[Sequence[Tuple[int, int]]],
        alchemical_weights: Sequence[float],
        device: str = "cpu",
        model: str = "medium",
    ):
        """
        Initialize the Alchemical MACE calculator.

        Args:
            atoms (ase.Atoms): Atoms object.
            alchemical_pairs (Sequence[Sequence[Tuple[int, int]]]): List of
                alchemical pairs. Each pair is a tuple of the atom index and
                atomic number of an alchemical atom.
            alchemical_weights (Sequence[float]): List of alchemical weights.
            device (str): Device to run the calculations on.
            model (str): Model to use for the MACE calculator.
        """
        Calculator.__init__(self)
        self.results = {}
        self.implemented_properties = ["energy", "free_energy", "forces", "stress"]

        # Build the alchemical MACE model
        self.device = device
        self.model = alchemical_mace_mp(
            model=model, device=device, default_dtype="float32"
        )
        for param in self.model.parameters():
            param.requires_grad = False

        # Set AlchemyManager
        z_table, r_max = get_z_table_and_r_max(self.model)
        alchemical_weights = torch.tensor(alchemical_weights, dtype=torch.float32)
        self.alchemy_manager = AlchemyManager(
            atoms=atoms,
            alchemical_pairs=alchemical_pairs,
            alchemical_weights=alchemical_weights,
            z_table=z_table,
            r_max=r_max,
        ).to(self.device)

        # Disable alchemical weights gradients by default
        self.alchemy_manager.alchemical_weights.requires_grad = False
        self.calculate_alchemical_grad = False

        self.num_atoms = len(atoms)

    def set_alchemical_weights(self, alchemical_weights: Sequence[float]):
        alchemical_weights = torch.tensor(
            alchemical_weights,
            dtype=torch.float32,
            device=self.device,
        )
        self.alchemy_manager.alchemical_weights.data = alchemical_weights

    def get_alchemical_atomic_masses(self) -> np.ndarray:
        # Get atomic masses for alchemical atoms
        node_masses = ase.data.atomic_masses[self.alchemy_manager.atomic_numbers]
        weights = self.alchemy_manager.alchemical_weights.data
        weights = F.pad(weights, (1, 0), "constant", 1.0).cpu().numpy()
        node_weights = weights[self.alchemy_manager.weight_indices]

        # Scatter sum to get the atomic masses
        atom_masses = np.zeros(self.num_atoms, dtype=np.float32)
        np.add.at(
            atom_masses, self.alchemy_manager.atom_indices, node_masses * node_weights
        )
        return atom_masses

    # pylint: disable=dangerous-default-value
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # prepare data
        tensor_kwargs = {"dtype": torch.float32, "device": self.device}
        positions = torch.tensor(atoms.get_positions(), **tensor_kwargs)
        cell = torch.tensor(atoms.get_cell().array, **tensor_kwargs)
        if self.calculate_alchemical_grad:
            self.alchemy_manager.alchemical_weights.requires_grad = True
        batch = self.alchemy_manager(positions, cell).to(self.device)

        # get outputs
        if self.calculate_alchemical_grad:
            out = self.model(batch, compute_stress=True, compute_alchemical_grad=True)
            (grad,) = torch.autograd.grad(
                outputs=[batch["node_weights"], batch["edge_weights"]],
                inputs=[self.alchemy_manager.alchemical_weights],
                grad_outputs=[out["node_grad"], out["edge_grad"]],
                retain_graph=False,
                create_graph=False,
            )
            grad = grad.cpu().numpy()
            self.alchemy_manager.alchemical_weights.requires_grad = False
        else:
            out = self.model(batch, retain_graph=False, compute_stress=True)
            grad = np.zeros(
                self.alchemy_manager.alchemical_weights.shape[0], dtype=np.float32
            )

        # store results
        self.results = {}
        self.results["energy"] = out["energy"][0].item()
        self.results["free_energy"] = self.results["energy"]
        self.results["forces"] = out["forces"].detach().cpu().numpy()
        self.results["stress"] = full_3x3_to_voigt_6_stress(
            out["stress"][0].detach().cpu().numpy()
        )
        self.results["alchemical_grad"] = grad


class NVTMACECalculator(Calculator):
    def __init__(self, model: str = "medium", device: str = "cuda"):
        Calculator.__init__(self)
        self.results = {}
        self.implemented_properties = ["energy", "free_energy", "forces", "stress"]
        self.device = device
        self.model = mace_mp(
            model=model, device=device, default_dtype="float32"
        ).models[0]
        self.z_table, self.r_max = get_z_table_and_r_max(self.model)
        for param in self.model.parameters():
            param.requires_grad = False

    # pylint: disable=dangerous-default-value
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # prepare data
        config = data.config_from_atoms(atoms)
        atomic_data = data.AtomicData.from_config(
            config, z_table=self.z_table, cutoff=self.r_max
        )
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[atomic_data],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader)).to(self.device)

        out = self.model(batch, compute_stress=False)
        self.results = {}
        self.results["energy"] = out["energy"][0].item()
        self.results["free_energy"] = self.results["energy"]
        self.results["forces"] = out["forces"].detach().cpu().numpy()


class FrenkelLaddCalculator(Calculator):
    """
    Frenkel-Ladd calculator for ASE.
    """

    def __init__(
        self,
        spring_constants: np.ndarray,
        initial_positions: np.ndarray,
        device: str,
        model: str = "medium",
    ):
        """
        Initialize the Frenkel-Ladd calculator.

        Args:
            spring_constants (np.ndarray): Spring constants for each atom.
            initial_positions (np.ndarray): Initial positions of the atoms.
            device (str): Device to run the calculations on.
            model (str): Model to use for the MACE calculator.
        """
        Calculator.__init__(self)
        self.results = {}
        self.implemented_properties = ["energy", "free_energy", "forces"]
        self.device = device
        self.model = mace_mp(
            model=model, device=device, default_dtype="float32"
        ).models[0]
        self.z_table, self.r_max = get_z_table_and_r_max(self.model)
        for param in self.model.parameters():
            param.requires_grad = False

        # Spring constants
        self.spring_constants = spring_constants
        self.initial_positions = initial_positions

        # Reversible scaling factor
        self.weights = [1.0, 0.0]
        self.compute_mace = True

    def set_weights(self, lambda_value: float):
        self.weights = [1.0 - lambda_value, lambda_value]

    # pylint: disable=dangerous-default-value
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # Get MACE results if needed
        if self.compute_mace:
            config = data.config_from_atoms(atoms)
            atomic_data = data.AtomicData.from_config(
                config, z_table=self.z_table, cutoff=self.r_max
            )
            data_loader = torch_geometric.dataloader.DataLoader(
                dataset=[atomic_data],
                batch_size=1,
                shuffle=False,
                drop_last=False,
            )
            batch = next(iter(data_loader)).to(self.device)
            out = self.model(batch, compute_stress=False)  # Frenkel-Ladd is NVT
            mace_energy = out["energy"][0].item()
            mace_forces = out["forces"].detach().cpu().numpy()
        else:
            mace_energy = 0.0
            mace_forces = np.zeros((len(atoms), 3), dtype=np.float32)

        # Get spring energy and forces
        displacement = atoms.get_positions() - self.initial_positions
        spring_energy = 0.5 * np.sum(
            self.spring_constants * np.sum(displacement**2, axis=1)
        )
        spring_forces = -self.spring_constants[:, None] * displacement

        # Combine energies and forces
        total_energy = self.weights[0] * spring_energy + self.weights[1] * mace_energy
        total_forces = self.weights[0] * spring_forces + self.weights[1] * mace_forces
        if self.compute_mace:
            energy_diff = mace_energy - spring_energy
        else:
            energy_diff = 0.0

        self.results = {}
        self.results["energy"] = total_energy
        self.results["free_energy"] = total_energy
        self.results["forces"] = total_forces
        self.results["energy_diff"] = energy_diff


class DefectFrenkelLaddCalculator(Calculator):
    """
    Frenkel-Ladd calculator for ASE, for a crystal with a defect.
    """

    def __init__(
        self,
        atoms: ase.Atoms,
        spring_constant: float,
        defect_index: int,
        device: str = "cpu",
        model: str = "medium",
    ):
        """
        Initialize the Frenkel-Ladd calculator.

        Args:
            atoms (ase.Atoms): Atoms object.
            spring_constant (float): Spring constant for the defect atom.
            defect_index (int): Index of the defect atom.
            device (str): Device to run the calculations on.
            model (str): Model to use for the MACE calculator.
        """
        Calculator.__init__(self)
        self.results = {}
        self.implemented_properties = ["energy", "free_energy", "forces", "stress"]

        # Build the alchemical MACE model
        self.device = device
        self.model = alchemical_mace_mp(
            model=model, device=device, default_dtype="float32"
        )
        for param in self.model.parameters():
            param.requires_grad = False

        # Set AlchemyManager
        z_table, r_max = get_z_table_and_r_max(self.model)
        alchemical_weights = torch.tensor([1.0], dtype=torch.float32)
        atomic_number = atoms.get_atomic_numbers()[defect_index]
        alchemical_pairs = [[AlchemicalPair(defect_index, atomic_number)]]
        self.alchemy_manager = AlchemyManager(
            atoms=atoms,
            alchemical_pairs=alchemical_pairs,
            alchemical_weights=alchemical_weights,
            z_table=z_table,
            r_max=r_max,
        ).to(self.device)

        # Disable alchemical weights gradients by default
        self.alchemy_manager.alchemical_weights.requires_grad = False
        self.calculate_alchemical_grad = False

        self.num_atoms = len(atoms)

        # Switching
        self.defect_index = defect_index
        self.spring_constant = spring_constant

    def set_alchemical_weight(self, alchemical_weight: float):
        # Set alchemical weights
        alchemical_weights = torch.tensor(
            [1.0 - alchemical_weight],  # initial = original atoms = 1 - 0
            dtype=torch.float32,
            device=self.device,
        )
        self.alchemy_manager.alchemical_weights.data = alchemical_weights

    # pylint: disable=dangerous-default-value
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # prepare data
        tensor_kwargs = {"dtype": torch.float32, "device": self.device}
        positions = torch.tensor(atoms.get_positions(), **tensor_kwargs)
        cell = torch.tensor(atoms.get_cell().array, **tensor_kwargs)
        if self.calculate_alchemical_grad:
            self.alchemy_manager.alchemical_weights.requires_grad = True
        batch = self.alchemy_manager(positions, cell).to(self.device)

        # get outputs
        if self.calculate_alchemical_grad:
            out = self.model(batch, retain_graph=True, compute_stress=True)
            out["energy"].backward()
            grad = self.alchemy_manager.alchemical_weights.grad.item()
            self.alchemy_manager.alchemical_weights.grad.zero_()
            self.alchemy_manager.alchemical_weights.requires_grad = False
        else:
            out = self.model(batch, retain_graph=False, compute_stress=True)
            grad = 0.0
        mace_energy = out["energy"][0].item()
        mace_forces = out["forces"].detach().cpu().numpy()
        mace_stress = out["stress"][0].detach().cpu().numpy()

        # Get spring energy and forces
        cell_center = np.array([0.5, 0.5, 0.5]) @ atoms.get_cell().array
        displacement = atoms.get_positions()[self.defect_index] - cell_center
        spring_energy = 0.5 * self.spring_constant * np.sum(displacement**2)
        spring_forces = -self.spring_constant * displacement

        # Combine energies and forces
        # Note: weight here is 1 - lambda, and we're not weighting the mace
        # energy because it's already weighted by the alchemical weight
        weight = self.alchemy_manager.alchemical_weights.item()
        total_energy = mace_energy + (1 - weight) * spring_energy
        total_forces = mace_forces
        total_forces[self.defect_index] += (1 - weight) * spring_forces
        if self.calculate_alchemical_grad:
            # H(lambda) = E(1 - lambda) + lambda * spring_energy
            # dH/d(lambda) = -dE/d(1 - lambda) + spring_energy
            grad = -grad + spring_energy

        # store results
        self.results = {}
        self.results["energy"] = total_energy
        self.results["free_energy"] = total_energy
        self.results["forces"] = total_forces
        self.results["stress"] = full_3x3_to_voigt_6_stress(mace_stress)
        self.results["alchemical_grad"] = grad


def get_alchemical_optimized_cellpar(
    atoms: ase.Atoms,
    alchemical_pairs: Sequence[Sequence[Tuple[int, int]]],
    alchemical_weights: Sequence[float],
    model: str = "medium",
    device: str = "cpu",
    **kwargs,
):
    """
    Optimize the cell parameters of a crystal with alchemical atoms using the
    Alchemical MACE calculator.

    Args:
        atoms (ase.Atoms): Atoms object.
        alchemical_pairs (Sequence[Sequence[Tuple[int, int]]]): List of
            alchemical pairs. Each pair is a tuple of the atom index and
            atomic number of an alchemical atom.
        alchemical_weights (Sequence[float]): List of alchemical weights.
        model (str): Model to use for the MACE calculator.
        device (str): Device to run the calculations on.

    Returns:
        np.ndarray: Optimized cell parameters.
    """
    # Make a copy of the atoms object
    atoms = atoms.copy()

    # Load Alchemical MACE calculator and relax the structure
    calc = AlchemicalMACECalculator(
        atoms, alchemical_pairs, alchemical_weights, device=device, model=model
    )
    atoms.set_calculator(calc)
    atoms.set_masses(calc.get_alchemical_atomic_masses())
    atoms = ExpCellFilter(atoms)
    optimizer = FIRE(atoms)
    optimizer.run(fmax=kwargs.get("fmax", 0.01), steps=kwargs.get("steps", 500))

    # Return the optimized cell parameters
    return atoms.atoms.get_cell().cellpar()
