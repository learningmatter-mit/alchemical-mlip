import argparse
from pathlib import Path

import ase
import numpy as np
import pandas as pd
from ase import units
from ase.build import make_supercell
from ase.constraints import ExpCellFilter
from ase.md.npt import NPT
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.optimize import FIRE
from mace.calculators import mace_mp
from tqdm import tqdm

from alchemical_mace.calculator import AlchemicalMACECalculator
from alchemical_mace.model import AlchemicalPair
from alchemical_mace.utils import upper_triangular_cell


# Arguments
parser = argparse.ArgumentParser()

# Structure
parser.add_argument("--structure-file", type=str)
parser.add_argument("--supercell", type=int, nargs=3, default=[6, 6, 6])

# Alchemy
parser.add_argument("--switch-pair", type=str, nargs=2, default=["Pb", "Sn"])

# Molecular dynamics: general
parser.add_argument("--temperature", type=float, default=300.0)
parser.add_argument("--pressure", type=float, default=1.0)
parser.add_argument("--timestep", type=float, default=2.0)
parser.add_argument("--ttime", type=float, default=25.0)
parser.add_argument("--ptime", type=int, default=75.0)

# Molecular dynamics: timesteps
parser.add_argument("--npt-equil-stpes", type=int, default=10000)
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
# Cell volume equilibration
################################################################################

atoms = initial_atoms.copy()
atoms.set_calculator(mace_calc)
bulk_modulus = 100.0 * units.GPa

# NPT equilibration
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

# Define alchemical transformation
src_elem, dst_elem = args.switch_pair
src_Z, dst_Z = ase.data.atomic_numbers[src_elem], ase.data.atomic_numbers[dst_elem]
src_idx = np.where(atoms.get_atomic_numbers() == src_Z)[0]
alchemical_pairs = [
    [AlchemicalPair(atom_index=idx, atomic_number=Z) for idx in src_idx]
    for Z in [src_Z, dst_Z]
]

# Set up the alchemical MACE calculator
calc = AlchemicalMACECalculator(
    atoms=atoms,
    alchemical_pairs=alchemical_pairs,
    alchemical_weights=[1.0, 0.0],
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
lambda_steps = t ** 5 * (70 * t ** 4 - 315 * t ** 3 + 540 * t ** 2 - 420 * t + 126)
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
    lambda_grad = (alchemical_grad[1] - alchemical_grad[0]) / num_atoms
    return {
        "time": time,
        "potential": dynamics.atoms.get_potential_energy() / num_atoms,
        "temperature": dynamics.atoms.get_temperature(),
        "volume": dynamics.atoms.get_volume() / num_atoms,
        "lambda": lambda_value,
        "lambda_grad": lambda_grad,
    }


# Simulation loop
total_steps = 2 * args.alchemy_equil_steps + 2 * args.alchemy_switch_steps

observables = []
for step in (tqdm(range(total_steps), desc="Alchemical switching")):
    lambda_value = lambda_values[step]
    grad_enabled = calculate_gradients[step]

    # Set alchemical weights and atomic masses
    calc.set_alchemical_weights([1 - lambda_value, lambda_value])
    atoms.set_masses(calc.get_alchemical_atomic_masses())
    calc.calculate_alchemical_grad = grad_enabled

    dyn.run(steps=1)
    if step % args.log_interval == 0:
        time = (step + 1) * args.timestep
        observables.append(get_observables(dyn, time, lambda_value))

# Save observables
df = pd.DataFrame(observables)
df.to_csv(args.output_dir / "observables.csv", index=False)

# Save masses for post-processing
calc.set_alchemical_weights([1.0, 0.0])
np.save(args.output_dir / "masses_init.npy", calc.get_alchemical_atomic_masses())
calc.set_alchemical_weights([0.0, 1.0])
np.save(args.output_dir / "masses_final.npy", calc.get_alchemical_atomic_masses())
