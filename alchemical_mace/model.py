import ast
from collections import namedtuple
from typing import Dict, List, Optional, Sequence, Tuple

import ase
import numpy as np
import torch
import torch.nn.functional as F
from e3nn import o3
from e3nn.util.jit import compile_mode
from mace import modules, tools
from mace.calculators import mace_mp
from mace.data.neighborhood import get_neighborhood
from mace.modules import RealAgnosticResidualInteractionBlock, ScaleShiftMACE
from mace.modules.utils import get_edge_vectors_and_lengths, get_symmetric_displacement
from mace.tools import (
    AtomicNumberTable,
    atomic_numbers_to_indices,
    to_one_hot,
    torch_geometric,
    utils,
)
from mace.tools.scatter import scatter_sum

################################################################################
# Alchemy manager class for handling alchemical weights
################################################################################

AlchemicalPair = namedtuple("AlchemicalPair", ["atom_index", "atomic_number"])


class AlchemyManager(torch.nn.Module):
    """
    Class for managing alchemical weights and building alchemical graphs for MACE.
    """

    def __init__(
        self,
        atoms: ase.Atoms,
        alchemical_pairs: Sequence[Sequence[Tuple[int, int]]],
        alchemical_weights: torch.Tensor,
        z_table: AtomicNumberTable,
        r_max: float,
    ):
        """
        Initialize the alchemy manager.

        Args:
            atoms: ASE atoms object
            alchemical_pairs: List of lists of tuples, where each tuple contains
                the atom index and atomic number of an alchemical atom
            alchemical_weights: Tensor of alchemical weights
            z_table: Atomic number table
            r_max: Maximum cutoff radius for the alchemical graph
        """
        super().__init__()
        self.alchemical_weights = torch.nn.Parameter(alchemical_weights)
        self.r_max = r_max

        # Process alchemical pairs into atom indices and atomic numbers
        # Alchemical weights are 1-indexed, 0 is reserved for non-alchemical atoms
        alchemical_atom_indices = []
        alchemical_atomic_numbers = []
        alchemical_weight_indices = []

        for weight_idx, pairs in enumerate(alchemical_pairs):
            for pair in pairs:
                alchemical_atom_indices.append(pair.atom_index)
                alchemical_atomic_numbers.append(pair.atomic_number)
                alchemical_weight_indices.append(weight_idx + 1)

        non_alchemical_atom_indices = [
            i for i in range(len(atoms)) if i not in alchemical_atom_indices
        ]
        non_alchemical_atomic_numbers = atoms.get_atomic_numbers()[
            non_alchemical_atom_indices
        ].tolist()
        non_alchemical_weight_indices = [0] * len(non_alchemical_atom_indices)

        self.atom_indices = alchemical_atom_indices + non_alchemical_atom_indices
        self.atomic_numbers = alchemical_atomic_numbers + non_alchemical_atomic_numbers
        self.weight_indices = alchemical_weight_indices + non_alchemical_weight_indices

        self.atom_indices = np.array(self.atom_indices)
        self.atomic_numbers = np.array(self.atomic_numbers)
        self.weight_indices = np.array(self.weight_indices)

        sort_idx = np.argsort(self.atom_indices)
        self.atom_indices = self.atom_indices[sort_idx]
        self.atomic_numbers = self.atomic_numbers[sort_idx]
        self.weight_indices = self.weight_indices[sort_idx]

        # Array to map original atom indices to alchemical indices
        # -1 means the atom does not have a corresponding alchemical atom
        # [n_atoms, n_weights + 1]
        self.original_to_alchemical_index = np.full(
            (len(atoms), len(alchemical_pairs) + 1), -1
        )
        for i, (atom_idx, weight_idx) in enumerate(
            zip(self.atom_indices, self.weight_indices)
        ):
            self.original_to_alchemical_index[atom_idx, weight_idx] = i

        self.is_original_atom_alchemical = np.any(
            self.original_to_alchemical_index[:, 1:] != -1, axis=1
        )

        # Extract common node features
        z_indices = atomic_numbers_to_indices(self.atomic_numbers, z_table=z_table)
        node_attrs = to_one_hot(
            torch.tensor(z_indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(z_table),
        )
        self.register_buffer("node_attrs", node_attrs)
        self.pbc = atoms.get_pbc()

    def forward(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Build an alchemical graph for the given positions and cell.

        Args:
            positions: Tensor of atomic positions
            cell: Tensor of cell vectors

        Returns:
            Dictionary containing the alchemical graph data
        """

        # Build original atom graph
        orig_edge_index, shifts, unit_shifts = get_neighborhood(
            positions=positions.detach().cpu().numpy(),
            cutoff=self.r_max,
            pbc=self.pbc,
            cell=cell.detach().cpu().numpy(),
        )

        # Extend edges to alchemical pairs
        edge_index = []
        orig_edge_loc = []
        edge_weight_indices = []

        is_alchemical = self.is_original_atom_alchemical[orig_edge_index]
        src_non_dst_non = ~is_alchemical[0] & ~is_alchemical[1]
        src_non_dst_alch = ~is_alchemical[0] & is_alchemical[1]
        src_alch_dst_non = is_alchemical[0] & ~is_alchemical[1]
        src_alch_dst_alch = is_alchemical[0] & is_alchemical[1]

        # Both non-alchemical: keep as is
        _orig_edge_index = orig_edge_index[:, src_non_dst_non]
        edge_index.append(self.original_to_alchemical_index[_orig_edge_index, 0])
        orig_edge_loc.append(np.where(src_non_dst_non)[0])
        edge_weight_indices.append(np.zeros_like(_orig_edge_index[0]))

        # Source non-alchemical, destination alchemical: pair all, weights are 1
        _src, _dst = orig_edge_index[:, src_non_dst_alch]
        _orig_edge_loc = np.where(src_non_dst_alch)[0]
        _src = self.original_to_alchemical_index[_src, 0]
        _dst = self.original_to_alchemical_index[_dst, :]
        _dst_mask = _dst != -1
        _dst = _dst[_dst_mask]
        _repeat = _dst_mask.sum(axis=1)
        _src = np.repeat(_src, _repeat)
        edge_index.append(np.stack((_src, _dst), axis=0))
        orig_edge_loc.append(np.repeat(_orig_edge_loc, _repeat))
        edge_weight_indices.append(np.zeros_like(_src))

        # Source alchemical, destination non-alchemical: pair all, follow src weights
        _src, _dst = orig_edge_index[:, src_alch_dst_non]
        _orig_edge_loc = np.where(src_alch_dst_non)[0]
        _src = self.original_to_alchemical_index[_src, :]
        _dst = self.original_to_alchemical_index[_dst, 0]
        _src_mask = _src != -1
        _src = _src[_src_mask]
        _repeat = _src_mask.sum(axis=1)
        _dst = np.repeat(_dst, _repeat)
        edge_index.append(np.stack((_src, _dst), axis=0))
        orig_edge_loc.append(np.repeat(_orig_edge_loc, _repeat))
        edge_weight_indices.append(np.where(_src_mask)[1])

        # Both alchemical: pair according to alchemical indices, weights are 1
        _orig_edge_index = orig_edge_index[:, src_alch_dst_alch]
        _orig_edge_loc = np.where(src_alch_dst_alch)[0]
        _alch_edge_index = self.original_to_alchemical_index[_orig_edge_index, :]
        _idx = np.where((_alch_edge_index != -1).all(axis=0))
        edge_index.append(_alch_edge_index[:, _idx[0], _idx[1]])
        orig_edge_loc.append(_orig_edge_loc[_idx[0]])
        edge_weight_indices.append(np.zeros_like(_idx[0]))

        # Collect all edges
        edge_index = np.concatenate(edge_index, axis=1)
        orig_edge_loc = np.concatenate(orig_edge_loc)
        edge_weight_indices = np.concatenate(edge_weight_indices)

        # Convert to torch tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        shifts = torch.tensor(shifts[orig_edge_loc], dtype=torch.float32)
        unit_shifts = torch.tensor(unit_shifts[orig_edge_loc], dtype=torch.float32)

        # Alchemical weights for nodes and edges
        weights = F.pad(self.alchemical_weights, (1, 0), "constant", 1.0)
        node_weights = weights[self.weight_indices]
        edge_weights = weights[edge_weight_indices]

        # Build data batch
        atomic_data = torch_geometric.data.Data(
            num_nodes=len(self.atom_indices),
            edge_index=edge_index,
            node_attrs=self.node_attrs,
            positions=positions[self.atom_indices],
            shifts=shifts,
            unit_shifts=unit_shifts,
            cell=cell,
            node_weights=node_weights,
            edge_weights=edge_weights,
            node_atom_indices=torch.tensor(self.atom_indices, dtype=torch.long),
        )
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[atomic_data],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader))

        return batch


################################################################################
# Alchemical MACE model
################################################################################

# get_outputs function from mace.modules.utils is modified to calculate also
# the alchemical gradients


def get_outputs(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: torch.Tensor,
    cell: torch.Tensor,
    node_weights: torch.Tensor,
    edge_weights: torch.Tensor,
    retain_graph: bool = False,
    create_graph: bool = False,
    compute_force: bool = True,
    compute_stress: bool = False,
    compute_alchemical_grad: bool = False,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    if not compute_force:
        return None, None, None, None, None
    inputs = [positions]
    if compute_stress:
        inputs.append(displacement)
    if compute_alchemical_grad:
        inputs.extend([node_weights, edge_weights])
    gradients = torch.autograd.grad(
        outputs=[energy],
        inputs=inputs,
        grad_outputs=grad_outputs,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
    )

    forces = gradients[0]
    stress = torch.zeros_like(displacement)
    virials = gradients[1] if compute_stress else None
    if compute_alchemical_grad:
        node_grad, edge_grad = gradients[-2], gradients[-1]
    else:
        node_grad, edge_grad = None, None
    if compute_stress and virials is not None:
        cell = cell.view(-1, 3, 3)
        volume = torch.einsum(
            "zi,zi->z",
            cell[:, 0, :],
            torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
        ).unsqueeze(-1)
        stress = virials / volume.view(-1, 1, 1)

    if forces is not None:
        forces = -1 * forces
    if virials is not None:
        virials = -1 * virials
    return forces, virials, stress, node_grad, edge_grad


@compile_mode("script")
class AlchemicalResidualInteractionBlock(RealAgnosticResidualInteractionBlock):
    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor,  # alchemy
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        tp_weights = tp_weights * edge_weights[:, None]  # alchemy
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class AlchemicalMACE(ScaleShiftMACE):
    def forward(
        self,
        data: Dict[str, torch.Tensor],
        retain_graph: bool = False,  # alchemy
        create_graph: bool = False,  # alchemy
        compute_force: bool = True,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_alchemical_grad: bool = False,  # alchemy
        map_to_original_atoms: bool = True,  # alchemy
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["positions"].requires_grad_(True)
        data["node_attrs"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        node_e0 = node_e0 * data["node_weights"]  # alchemy
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        node_es_list = []
        node_feats_list = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                edge_weights=data["edge_weights"],  # alchemy
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"]
            )
            node_feats_list.append(node_feats)
            node_es_list.append(readout(node_feats).squeeze(-1))  # {[n_nodes, ], }

        # Concatenate node features
        # node_feats_out = torch.cat(node_feats_list, dim=-1)

        # Sum over interactions
        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0), dim=0
        )  # [n_nodes, ]
        node_inter_es = self.scale_shift(node_inter_es)
        node_inter_es = node_inter_es * data["node_weights"]  # alchemy

        # Sum over nodes in graph
        inter_e = scatter_sum(
            src=node_inter_es, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Add E_0 and (scaled) interaction energy
        total_energy = e0 + inter_e
        node_energy = node_e0 + node_inter_es

        forces, virials, stress, node_grad, edge_grad = get_outputs(
            energy=total_energy,  # alchemy
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            node_weights=data["node_weights"],  # alchemy
            edge_weights=data["edge_weights"],  # alchemy
            retain_graph=retain_graph,  # alchemy
            create_graph=create_graph,  # alchemy
            compute_force=compute_force,
            # compute_virials=compute_virials,  # alchemy
            compute_stress=compute_stress,
            compute_alchemical_grad=compute_alchemical_grad,  # alchemy
        )

        # Map to original atoms (node energies and forces): alchemy
        if map_to_original_atoms:
            # Note: we're not giving the dim_size, as we assume that all
            # original atoms are present in the batch
            node_index = data["node_atom_indices"]
            node_energy = scatter_sum(src=node_energy, dim=0, index=node_index)
            if compute_force:
                forces = scatter_sum(src=forces, dim=0, index=node_index)

        output = {
            "energy": total_energy,
            "node_energy": node_energy,
            "interaction_energy": inter_e,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "node_grad": node_grad,
            "edge_grad": edge_grad,
        }

        return output


################################################################################
# Alchemical MACE universal model
################################################################################


def alchemical_mace_mp(
    model: str,
    device: str,
    default_dtype: str = "float32",
):
    """
    Load a pre-trained alchemical MACE model.

    Args:
        model: Model size (small, medium)
        device: Device to load the model onto
        default_dtype: Default data type for the model

    Returns:
        Alchemical MACE model
    """

    # Load foundation MACE model and extract initial parameters
    assert model in ("small", "medium")  # TODO: support large model
    mace = mace_mp(model=model, device=device, default_dtype=default_dtype).models[0]
    atomic_energies = mace.atomic_energies_fn.atomic_energies.detach().clone()
    z_table = utils.AtomicNumberTable([int(z) for z in mace.atomic_numbers])
    atomic_inter_scale = mace.scale_shift.scale.detach().clone()
    atomic_inter_shift = mace.scale_shift.shift.detach().clone()

    # Prepare arguments for building the model
    placeholder_args = ["--name", "None", "--train_file", "None"]
    args = tools.build_default_arg_parser().parse_args(placeholder_args)
    args.max_L = {"small": 0, "medium": 1, "large": 2}[model]
    args.num_channels = 128
    args.hidden_irreps = o3.Irreps(
        (args.num_channels * o3.Irreps.spherical_harmonics(args.max_L))
        .sort()
        .irreps.simplify()
    )

    # Build the alchemical MACE model
    model = AlchemicalMACE(
        r_max=6.0,
        num_bessel=10,
        num_polynomial_cutoff=5,
        max_ell=3,
        interaction_cls=AlchemicalResidualInteractionBlock,
        interaction_cls_first=AlchemicalResidualInteractionBlock,
        num_interactions=2,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps(args.hidden_irreps),
        MLP_irreps=o3.Irreps(args.MLP_irreps),
        atomic_energies=atomic_energies,
        avg_num_neighbors=args.avg_num_neighbors,
        atomic_numbers=z_table.zs,
        correlation=args.correlation,
        gate=modules.gate_dict[args.gate],
        radial_MLP=ast.literal_eval(args.radial_MLP),
        radial_type=args.radial_type,
        atomic_inter_scale=atomic_inter_scale,
        atomic_inter_shift=atomic_inter_shift,
    )

    # Load foundation model parameters
    model.load_state_dict(mace.state_dict(), strict=True)
    for i in range(int(model.num_interactions)):
        model.interactions[i].avg_num_neighbors = mace.interactions[i].avg_num_neighbors
    model = model.to(device)
    return model


def get_z_table_and_r_max(model: ScaleShiftMACE) -> Tuple[AtomicNumberTable, float]:
    """Extract the atomic number table and maximum cutoff radius from a MACE model."""
    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    r_max = model.r_max.item()
    return z_table, r_max
