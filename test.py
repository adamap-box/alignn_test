from jarvis.core.atoms import Atoms
from alignn.graphs import Graph
import torch
from alignn.models.alignn_atomwise import ALIGNNAtomWise
from typing import Literal
from alignn.utils import BaseSettings
from graph import Graph

def compute_bond_cosines(edges):
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # use law of cosines to compute angles cosines
    # negate src bond so displacements are like `a <- b -> c`
    # cos(theta) = ba \dot bc / (||ba|| ||bc||)
    r1 = -edges.src["r"]
    r2 = edges.dst["r"]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    # bond_cosine = torch.arccos((torch.clamp(bond_cosine, -1, 1)))
    # print (r1,r1.shape)
    # print (r2,r2.shape)
    # print (bond_cosine,bond_cosine.shape)
    return {"h": bond_cosine}


class ALIGNNAtomWiseConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.alignn."""

    name: Literal["alignn_atomwise"]
    alignn_layers: int = 2
    gcn_layers: int = 2
    atom_input_features: int = 1
    atom_input_features: int = 92
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 64
    hidden_features: int = 256
    # fc_layers: int = 1
    # fc_features: int = 64
    output_features: int = 1
    grad_multiplier: int = -1
    calculate_gradient: bool = True
    atomwise_output_features: int = 0
    graphwise_weight: float = 1.0
    gradwise_weight: float = 1.0
    stresswise_weight: float = 0.0
    atomwise_weight: float = 0.0
    # if link == log, apply `exp` to final outputs
    # to constrain predictions to be positive
    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    classification: bool = False
    force_mult_natoms: bool = False
    energy_mult_natoms: bool = True
    include_pos_deriv: bool = False
    use_cutoff_function: bool = False
    inner_cutoff: float = 3  # Ansgtrom
    stress_multiplier: float = 1
    add_reverse_forces: bool = True  # will make True as default soon
    lg_on_fly: bool = True  # will make True as default soon
    batch_stress: bool = True
    multiply_cutoff: bool = False
    use_penalty: bool = True
    extra_features: int = 0
    exponent: int = 5
    penalty_factor: float = 0.1
    penalty_threshold: float = 1
    additional_output_features: int = 0
    additional_output_weight: float = 0

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"

FIXTURES = {
        "lattice_mat": [
            [2.715, 2.715, 0],
            [0, 2.715, 2.715],
            [2.715, 0, 2.715],
        ],
        "coords": [[0, 0, 0], [0.25, 0.25, 0.25]],
        "elements": ["Si", "Si"],
}

Si = Atoms(
        lattice_mat=FIXTURES["lattice_mat"],
        coords=FIXTURES["coords"],
        elements=FIXTURES["elements"],
)

g, _ = Graph.atom_dgl_multigraph(
        atoms=Si, neighbor_strategy="radius_graph", cutoff=5
)
lg = g.line_graph(shared=True)
lg.apply_edges(compute_bond_cosines)
device = "cpu"

model = ALIGNNAtomWise(ALIGNNAtomWiseConfig(name="alignn_atomwise"))
model.to(device)
model.eval()
out = model([g, lg])
print(out)