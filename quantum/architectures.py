"""
Variational quantum circuit for HOMO-LUMO gap regression on QM9.

Encoding: atom coordinates mapped to RY/RZ rotations (Bloch sphere direction).
Entanglement: IsingXX (distance-based) + IsingYY (bond-type-based) for all
              heavy-atom pairs.
Trainable: num_layers x per-qubit RY rotations.
Output: Z expectation value on qubit 0.

Author: Dr. Aritra Bal (ETP)
Date: March 03, 2026
"""

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from itertools import combinations
from loguru import logger
from typing import Any, Tuple


# Hartree to eV conversion factor
_HARTREE_TO_EV: float = 1.


class QuantumCircuit:
    """
    Variational quantum circuit for molecular property regression.

    Node feature column layout expected (from qm9_dataloader):
        col 0: atomic number
        col 1: aromatic flag
        col 2: hybridisation scalar
        col 3: hydrogen count
        col 4: x (Angstrom)
        col 5: y (Angstrom)
        col 6: z (Angstrom)

    Edge feature: (max_nodes, max_nodes) bond-type matrix {0,1,2,3,4}.
    """

    def __init__(self, config: Any, device: qml.Device) -> None:
        """
        Args:
            config : Config namespace object.
            device : Initialised PennyLane device.
        """
        self.n_qubits   = config.model.n_qubits
        self.num_layers = config.model.num_layers
        self.operations_per_layer = config.model.operations_per_layer
        self.backend    = config.model.backend
        self.device     = device
        self.all_wires  = list(range(self.n_qubits))
        self.two_comb_wires: list = list(combinations(self.all_wires, 2))
        self.num_extra_weights = config.model.extra_weights
        if self.n_qubits < 7:
            logger.warning(
                f"n_qubits={self.n_qubits} < 7: the filtered QM9 subset contains "
                f"7-8 heavy atoms; some heavy atoms will be excluded from encoding."
            )
        if self.n_qubits > 8:
            logger.warning(
                f"n_qubits={self.n_qubits} > 8: more qubits than heavy atoms; "
                f"hydrogen atom rows may be included in encoding."
            )

        # Trainable weights: (num_layers, n_qubits) RY rotation angles.
        self.weights: pnp.ndarray = pnp.array(
            np.random.uniform(-np.pi/6., np.pi/6., (self.num_layers, self.operations_per_layer, self.n_qubits)),
            requires_grad=True,
        )
        self.extra_weights: pnp.ndarray = pnp.array(
            np.random.uniform(0.,5., self.num_extra_weights),
            requires_grad=True
        )
        self.qnode = qml.QNode(self._circuit, self.device, interface=self.backend)
        logger.info(
            f"QuantumCircuit ready | qubits={self.n_qubits} | "
            f"layers={self.num_layers} | pairs={len(self.two_comb_wires)} | "
            f"weights shape={self.weights.shape}"
        )

    # ------------------------------------------------------------------
    # Internal circuit components
    # ------------------------------------------------------------------

    def _encode(self, node_feat: np.ndarray, batched: bool) -> None:
        """
        Encode each atom's 3D coordinate as a direction on the Bloch sphere.

        For atom i, compute the polar (theta) and azimuthal (phi) angles of
        the vector (x, y, z) and apply RY(theta) followed by RZ(phi).
        Magnitude information is discarded.

        Args:
            node_feat : (n_qubits, 7) or (B, n_qubits, 7)
            batched   : True if a batch dimension is present.
        """
        for i in self.all_wires:
            if batched:
                x = node_feat[:, i, 4]
                y = node_feat[:, i, 5]
                z = node_feat[:, i, 6]
            else:
                x = node_feat[i, 4]
                y = node_feat[i, 5]
                z = node_feat[i, 6]

            r     = np.sqrt(x**2 + y**2 + z**2 + 1e-8)
            theta = np.arccos(np.clip(z / r, -1.0, 1.0))  # polar angle [0, pi]
            phi   = np.arctan2(y, x)                        # azimuthal [-pi, pi]

            qml.RX(theta, wires=i)
            qml.RY(phi,   wires=i)
            qml.RX(np.pi/2 * r/6., wires=i)  
            
    def _entangle(self, node_feat: np.ndarray, edge_feat: np.ndarray, batched: bool) -> None:
        """
        Apply pairwise IsingXX and IsingYY gates over all heavy-atom pairs.

        IsingXX angle: (mean_squared_distance / 10) * (pi/2)
        IsingYY angle: (pi/6) * bond_type

        Args:
            node_feat : (n_qubits, 7) or (B, n_qubits, 7)
            edge_feat : (max_nodes, max_nodes) or (B, max_nodes, max_nodes)
            batched   : True if a batch dimension is present.
        """
        for (i, j) in self.two_comb_wires:
            if batched:
                xi, yi, zi = node_feat[:, i, 4], node_feat[:, i, 5], node_feat[:, i, 6]
                xj, yj, zj = node_feat[:, j, 4], node_feat[:, j, 5], node_feat[:, j, 6]
                bond_type   = edge_feat[:, i, j]
            else:
                xi, yi, zi = node_feat[i, 4], node_feat[i, 5], node_feat[i, 6]
                xj, yj, zj = node_feat[j, 4], node_feat[j, 5], node_feat[j, 6]
                bond_type   = edge_feat[i, j]

            # Mean squared distance over x, y, z components.
            #msd       = np.sqrt((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2)
            #angle_xx  = (1. - msd / 10.) * (np.pi / 2.0)
            # Angular separation
            norm_i    = np.sqrt(xi**2 + yi**2 + zi**2) + 1e-8
            norm_j    = np.sqrt(xj**2 + yj**2 + zj**2) + 1e-8
            cos_X     = (xi*xj + yi*yj + zi*zj) / (norm_i * norm_j)
            X         = np.arccos(np.clip(cos_X, -1.0, 1.0))
            angle_xx  = np.where(X > np.pi / 2.0, 0.0, np.pi / 2.0 - X)
            
            angle_yy  = (np.pi / 12.0) * (bond_type+1.)

            qml.IsingXX(angle_xx, wires=[i, j])
            qml.IsingYY(angle_yy, wires=[i, j])

    def _trainable_layers(self, weights: pnp.ndarray, layer: int) -> None:
        """
        Apply num_layers of per-qubit trainable RY rotations.

        Args:
            weights : (num_layers, n_qubits) trainable parameter array.
        """
    
        for i in self.all_wires:
            qml.RY(weights[layer, 0, i], wires=i)  # RZ-RY-RZ rotation
            qml.RX(weights[layer, 1, i], wires=i)  # RX rotation
            #qml.RX(weights[layer, 2, i], wires=i)  # RX rotation
    # ------------------------------------------------------------------
    # QNode circuit function
    # ------------------------------------------------------------------

    def _circuit(
        self,
        weights: pnp.ndarray,
        node_feat: np.ndarray,
        edge_feat: np.ndarray,
    ) -> Any:
        """
        Full variational circuit: encode -> entangle -> trainable -> measure.

        Args:
            weights   : (num_layers, n_qubits) trainable parameters.
            node_feat : (max_nodes, 7) or (B, max_nodes, 7)
            edge_feat : (max_nodes, max_nodes) or (B, max_nodes, max_nodes)

        Returns:
            Expectation value of PauliZ on wire 0. Scalar or (B,) if batched.
        """
        # Squeeze trivial batch dimension; PennyLane broadcasting handles B>1.
        if node_feat.ndim == 3 and node_feat.shape[0] == 1:
            node_feat = node_feat[0]
            edge_feat = edge_feat[0]

        batched = node_feat.ndim == 3

        # Use only the first n_qubits atom rows (heavy atoms).
        if batched:
            node_feat = node_feat[:, :self.n_qubits, :]
        else:
            node_feat = node_feat[:self.n_qubits, :]
        for layer in range(self.num_layers):
            self._entangle(node_feat, edge_feat, batched)
            self._encode(node_feat, batched)
            self._trainable_layers(weights,layer)

        return qml.expval(qml.PauliZ(0))

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(
        self,
        weights: pnp.ndarray,
        node_feat: np.ndarray,
        edge_feat: np.ndarray,
        target: np.ndarray,
    ) -> pnp.ndarray:
        """
        MSE loss between normalised HOMO-LUMO gap and Z measurement.

        QM9 stores the gap in Hartree; it is converted to eV here.
        
        Args:
            weights   : (num_layers, n_qubits) trainable parameters.
            node_feat : (max_nodes, 7) or (B, max_nodes, 7)
            edge_feat : (max_nodes, max_nodes) or (B, max_nodes, max_nodes)
            target    : Scalar or (B,) HOMO-LUMO gap values in Hartree.

        Returns:
            Scalar MSE loss.
        """
        gap_ev       = target * _HARTREE_TO_EV
        #norm_target  = gap_ev / 10.0 - 1.0
        pred         = self.qnode(weights, node_feat, edge_feat)
        pred_ev = (1-pred) * 10.0
        return pnp.mean(pnp.abs((pred_ev - gap_ev)))