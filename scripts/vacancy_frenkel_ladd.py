import argparse
from pathlib import Path

import ase
import numpy as np
import pandas as pd
from ase import units
from ase.build import make_supercell
from ase.constraints import ExpCellFilter
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.optimize import FIRE
from mace.calculators import mace_mp
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm

from alchemical_mace.calculator import (
    DefectFrenkelLaddCalculator,
    FrenkelLaddCalculator,
    NVTMACECalculator,
)
from alchemical_mace.utils import upper_triangular_cell


# Arguments
parser = argparse.ArgumentParser()

# Structure
parser.add_argument("--structure-file", type=str)
parser.add_argument("--supercell", type=int, nargs=3, default=[5, 5, 5])

# Molecular dynamics: general
parser.add_argument("--temperature", type=float, default=300.0)
parser.add_argument("--pressure", type=float, default=1.0)
parser.add_argument("--timestep", type=float, default=2.0)
parser.add_argument("--ttime", type=float, default=25.0)
parser.add_argument("--ptime", type=int, default=75.0)

# Molecular dynamics: timesteps
parser.add_argument("--npt-equil-stpes", type=int, default=10000)
parser.add_argument("--npt-prod-steps", type=int, default=20000)
parser.add_argument("--nvt-equil-steps", type=int, default=20000)
parser.add_argument("--nvt-prod-steps", type=int, default=30000)
parser.add_argument("--alchemy-equil-steps", type=int, default=20000)
parser.add_argument("--alchemy-switch-steps", type=int, default=30000)

# Molecular dynamics: output control
parser.add_argument("--output-dir", type=Path, default=Path("results"))
parser.add_argument("--log-interval", type=int, default=1)

# MACE model
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--model", type=str, default="small")

args = parser.parse_args()
args.output_dir.mkdir(exist_ok=True, parents=True)


################################################################################
# Energy minimization: defect-free structure
################################################################################

# Load structure
atoms = ase.io.read(args.structure_file)
atoms = make_supercell(atoms, np.diag(args.supercell))

# Load universal MACE calculator and relax the structure
mace_calc = mace_mp(model=args.model, device=args.device, default_dtype="float32")
atoms.calc = mace_calc
atoms = ExpCellFilter(atoms)
optimizer = FIRE(atoms)
optimizer.run(fmax=0.01, steps=500)
atoms = atoms.atoms  # get the relaxed structure
initial_atoms = atoms.copy()  # save the initial structure


################################################################################
# Cell volume equilibration: defect-free structure
################################################################################

atoms = initial_atoms.copy()
atoms.set_calculator(mace_calc)
bulk_modulus = 100.0 * units.GPa

# Equilibration and volume calculation
dyn = Inhomogeneous_NPTBerendsen(
    atoms,
    timestep=args.timestep * units.fs,
    temperature_K=args.temperature,
    pressure_au=args.pressure * 1.01325 * units.bar,
    taut=args.ttime * units.fs,
    taup=args.ptime * units.fs,
    compressibility_au=1.0 / bulk_modulus,
)
MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
Stationary(atoms)

# NPT equilibration and volume relaxation
cellpar_traj = []
for step in tqdm(range(args.npt_equil_stpes), desc="NPT equil"):
    dyn.run(steps=1)
for step in tqdm(range(args.npt_prod_steps), desc="NPT prod"):
    dyn.run(steps=1)
    if step % args.log_interval == 0:
        cellpar_traj.append(atoms.get_cell().cellpar())
abc_new = np.mean(cellpar_traj, axis=0)[:3]

# Scale the initial cell to match the average volume
atoms = initial_atoms
atoms.set_cell(np.diag(abc_new), scale_atoms=True)
atoms.set_calculator(mace_calc)

# Relax the atomic positions
optimizer = FIRE(atoms)
optimizer.run(fmax=0.01, steps=500)
initial_atoms = atoms.copy()  # save the initial structure


################################################################################
# MSD calculation: defect-free structure
################################################################################

initial_positions = atoms.get_positions()
# Using the reversible scaling MACE calculator with fixed scale of 1.0
# since we can turn off the stress calculation
calc = NVTMACECalculator(device=args.device, model=args.model)
atoms.set_calculator(calc)

# NVT MSD calculation
dyn = Langevin(
    atoms,
    timestep=args.timestep * units.fs,
    temperature_K=args.temperature,
    friction=1 / (args.ttime * units.fs),
)
MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
Stationary(atoms)

temperatures = []
for step in tqdm(range(args.nvt_equil_steps), desc="NVT equil"):
    dyn.run(steps=1)
squared_disp = np.zeros(len(atoms))
for step in tqdm(range(args.nvt_prod_steps), desc="NVT prod"):
    dyn.run(steps=1)
    squared_disp += np.sum((atoms.get_positions() - initial_positions) ** 2, axis=1)
mean_squared_disp = squared_disp / args.nvt_prod_steps

# Calculate spring constants and average over symmetrically equivalent atoms
spring_constants = 3.0 * units.kB * args.temperature / mean_squared_disp
structure = AseAtomsAdaptor.get_structure(initial_atoms)
sga = SpacegroupAnalyzer(structure)
equivalent_indices = sga.get_symmetrized_structure().equivalent_indices
for indices in equivalent_indices:
    spring_constants[indices] = np.mean(spring_constants[indices])

np.save(args.output_dir / "spring_constants.npy", spring_constants)
np.save(args.output_dir / "masses.npy", atoms.get_masses())


################################################################################
# Frenkel-Ladd calculation: defect-free structure
################################################################################

atoms = initial_atoms.copy()
calc = FrenkelLaddCalculator(
    spring_constants=spring_constants,
    initial_positions=initial_positions,
    device=args.device,
    model=args.model,
)
atoms.set_calculator(calc)

# NVT Frenkel-Ladd calculation
dyn = Langevin(
    atoms,
    timestep=args.timestep * units.fs,
    temperature_K=args.temperature,
    friction=1 / (args.ttime * units.fs),
)
MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
Stationary(atoms)

# Define Frenkel-Ladd path
t = np.linspace(0.0, 1.0, args.alchemy_switch_steps)
lambda_steps = t**5 * (70 * t**4 - 315 * t**3 + 540 * t**2 - 420 * t + 126)
lambda_values = [
    np.zeros(args.alchemy_equil_steps),
    lambda_steps,
    np.ones(args.alchemy_equil_steps),
    lambda_steps[::-1],
]
lambda_values = np.concatenate(lambda_values)


def get_observables(dynamics, time, lambda_value):
    num_atoms = len(dynamics.atoms)
    return {
        "time": time,
        "potential": dynamics.atoms.get_potential_energy() / num_atoms,
        "temperature": dynamics.atoms.get_temperature(),
        "volume": dynamics.atoms.get_volume() / num_atoms,
        "lambda": lambda_value,
        "lambda_grad": dynamics.atoms._calc.results["energy_diff"] / num_atoms,
    }


# Simulation loop
calc.compute_mace = False
total_steps = 2 * args.alchemy_equil_steps + 2 * args.alchemy_switch_steps

observables = []
for step in tqdm(range(total_steps), desc="Frenkel-Ladd"):
    if step == args.alchemy_equil_steps:  # turn on MACE after spring equilibration
        calc.compute_mace = True
    lambda_value = lambda_values[step]
    calc.set_weights(lambda_value)

    dyn.run(steps=1)
    if step % args.log_interval == 0:
        time = (step + 1) * args.timestep
        observables.append(get_observables(dyn, time, lambda_value))

# Save observables
df = pd.DataFrame(observables)
df.to_csv(args.output_dir / "observables.csv", index=False)


################################################################################
# Cell volume equilibration: structure with a defect
################################################################################

atoms = initial_atoms.copy()

# Create a vacancy at the center of the supercell
vacancy_index = len(atoms) // 2
atom_mask = np.ones(len(atoms), dtype=bool)
atom_mask[vacancy_index] = False
del atoms[vacancy_index]

atoms.set_calculator(mace_calc)

# Equilibration and volume calculation
dyn = Inhomogeneous_NPTBerendsen(
    atoms,
    timestep=args.timestep * units.fs,
    temperature_K=args.temperature,
    pressure_au=args.pressure * 1.01325 * units.bar,
    taut=args.ttime * units.fs,
    taup=args.ptime * units.fs,
    compressibility_au=1.0 / bulk_modulus,
)
MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
Stationary(atoms)

# NPT equilibration and volume relaxation
cellpar_traj = []
for step in tqdm(range(args.npt_equil_stpes), desc="NPT equil"):
    dyn.run(steps=1)
for step in tqdm(range(args.npt_prod_steps), desc="NPT prod"):
    dyn.run(steps=1)
    if step % args.log_interval == 0:
        cellpar_traj.append(atoms.get_cell().cellpar())
abc_new = np.mean(cellpar_traj, axis=0)[:3]

# Scale the initial cell to match the average volume
atoms = initial_atoms.copy()
atoms.set_cell(np.diag(abc_new), scale_atoms=True)
del atoms[vacancy_index]
atoms.set_calculator(mace_calc)

# Relax the atomic positions
optimizer = FIRE(atoms)
optimizer.run(fmax=0.01, steps=500)


################################################################################
# Frenkel-Ladd calculation: structure with a defect
################################################################################

calc = FrenkelLaddCalculator(
    spring_constants=spring_constants[atom_mask],
    initial_positions=initial_positions[atom_mask],
    device=args.device,
    model=args.model,
)
atoms.set_calculator(calc)

# NVT Frenkel-Ladd calculation
dyn = Langevin(
    atoms,
    timestep=args.timestep * units.fs,
    temperature_K=args.temperature,
    friction=1 / (args.ttime * units.fs),
)
MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
Stationary(atoms)

# Simulation loop
calc.compute_mace = False
total_steps = 2 * args.alchemy_equil_steps + 2 * args.alchemy_switch_steps

observables = []
for step in tqdm(range(total_steps), desc="Frenkel-Ladd"):
    if step == args.alchemy_equil_steps:  # turn on MACE after spring equilibration
        calc.compute_mace = True
    lambda_value = lambda_values[step]
    calc.set_weights(lambda_value)

    dyn.run(steps=1)
    if step % args.log_interval == 0:
        time = (step + 1) * args.timestep
        observables.append(get_observables(dyn, time, lambda_value))

# Save observables
df = pd.DataFrame(observables)
df.to_csv(args.output_dir / "observables_defect.csv", index=False)


################################################################################
# Cell volume equilibration: partial Frenkel-Ladd calculation
################################################################################

atoms = initial_atoms.copy()
atoms.set_calculator(mace_calc)

# Equilibration and volume calculation
dyn = Inhomogeneous_NPTBerendsen(
    atoms,
    timestep=args.timestep * units.fs,
    temperature_K=args.temperature,
    pressure_au=args.pressure * 1.01325 * units.bar,
    taut=args.ttime * units.fs,
    taup=args.ptime * units.fs,
    compressibility_au=1.0 / bulk_modulus,
)
MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
Stationary(atoms)

# NPT equilibration and volume relaxation
for step in tqdm(range(args.npt_equil_stpes), desc="NPT equil"):
    dyn.run(steps=1)


################################################################################
# Alchemical switching
################################################################################

# Set up the partial Frenkel-Ladd calculation
calc = DefectFrenkelLaddCalculator(
    atoms=atoms,
    spring_constant=spring_constants[vacancy_index],
    defect_index=vacancy_index,
    device=args.device,
    model=args.model,
)
atoms.set_calculator(calc)
upper_triangular_cell(atoms)  # for ASE NPT

# NPT alchemical switching
ptime = args.ptime * units.fs
pfactor = bulk_modulus * ptime * ptime

dyn = NPT(
    atoms,
    timestep=args.timestep * units.fs,
    temperature_K=args.temperature,
    externalstress=args.pressure * 1.01325 * units.bar,
    ttime=args.ttime * units.fs,
    pfactor=pfactor,
)

# Define alchemical path
t = np.linspace(0.0, 1.0, args.alchemy_switch_steps)
lambda_steps = t**5 * (70 * t**4 - 315 * t**3 + 540 * t**2 - 420 * t + 126)
lambda_values = [
    np.zeros(args.alchemy_equil_steps),
    lambda_steps,
    np.ones(args.alchemy_equil_steps),
    lambda_steps[::-1],
]
lambda_values = np.concatenate(lambda_values)

calculate_gradients = [
    np.zeros(args.alchemy_equil_steps, dtype=bool),
    np.ones(args.alchemy_switch_steps, dtype=bool),
    np.zeros(args.alchemy_equil_steps, dtype=bool),
    np.ones(args.alchemy_switch_steps, dtype=bool),
]
calculate_gradients = np.concatenate(calculate_gradients)


def get_observables(dynamics, time, lambda_value):
    num_atoms = len(dynamics.atoms)
    alchemical_grad = dynamics.atoms._calc.results["alchemical_grad"]
    return {
        "time": time,
        "potential": dynamics.atoms.get_potential_energy() / num_atoms,
        "temperature": dynamics.atoms.get_temperature(),
        "volume": dynamics.atoms.get_volume() / num_atoms,
        "lambda": lambda_value,
        "lambda_grad": alchemical_grad / num_atoms,
    }


# Simulation loop
total_steps = 2 * args.alchemy_equil_steps + 2 * args.alchemy_switch_steps

observables = []
for step in tqdm(range(total_steps), desc="Alchemical switching"):
    lambda_value = lambda_values[step]
    grad_enabled = calculate_gradients[step]

    # Set alchemical weights and atomic masses
    calc.set_alchemical_weight(lambda_value)
    calc.calculate_alchemical_grad = grad_enabled

    dyn.run(steps=1)
    if step % args.log_interval == 0:
        time = (step + 1) * args.timestep
        observables.append(get_observables(dyn, time, lambda_value))

# Save observables
df = pd.DataFrame(observables)
df.to_csv(args.output_dir / "observables_FL.csv", index=False)
