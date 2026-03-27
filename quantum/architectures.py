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
from typing import TYPE_CHECKING, Any, Tuple
from tqdm import tqdm
if TYPE_CHECKING:
    from torch.utils.data import DataLoader


# Hartree to eV conversion factor
_HARTREE_TO_EV: float = 1.
_MEV = 1000
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
        self.read_qubits = config.model.read_qubits if hasattr(config.model, 'read_qubits') else min(self.n_qubits, 3)  # max 9 heavy atoms in filtered QM9 subset
        self.dmax = 1.7  # Angstrom; neighborhood cutoff
        self.num_layers = config.model.num_layers
        self.operations_per_layer = config.model.operations_per_layer
        self.backend    = config.model.backend
        self.device_name = config.model.device
        self.device     = device
        self.all_wires  = list(range(self.n_qubits))
        self.coeffs = []
        self.two_comb_wires: list = list(combinations(self.all_wires, 2))
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
        self.extra_weights = pnp.array(
            list(vars(config.model.extra_weights).values()), requires_grad=True
        )
        self.extra_weights_IDX = {k: i for i, k in enumerate(vars(config.model.extra_weights).keys())}
        self.qnode = qml.QNode(self._circuit, self.device, interface=self.backend, diff_method="parameter-shift")
        logger.info(
            f"QuantumCircuit ready | qubits={self.n_qubits} | "
            f"layers={self.num_layers} | pairs={len(self.two_comb_wires)} | "
            f"weights shape={self.weights.shape}"
        )

    # ------------------------------------------------------------------
    # Internal circuit components
    # ------------------------------------------------------------------

    def _encode(self, node_feat: np.ndarray, extra_weights: pnp.ndarray, batched: bool) -> None:
        """
        Encode each atom's 3D coordinate as a direction on the Bloch sphere.

        For atom i, compute the polar (theta) and azimuthal (phi) angles of
        the vector (x, y, z) and apply RY(theta) followed by RZ(phi).
        Magnitude information is discarded.

        Args:
            node_feat : (n_qubits, 7) or (B, n_qubits, 7)
            batched   : True if a batch dimension is present.
        """
        for i in range(self.n_qubits):
            if batched:
                x = node_feat[:, i, 4]
                y = node_feat[:, i, 5]
                z = node_feat[:, i, 6]
                a = np.int32(node_feat[:, i, 0])  # atomic number
            else:
                x = node_feat[i, 4]
                y = node_feat[i, 5]
                z = node_feat[i, 6]
                a = np.int32(node_feat[i, 0])  # atomic number

            elem_idx = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}[a]
            r     = np.sqrt(x**2 + y**2 + z**2 + 1e-8)
            theta = np.arccos(np.clip(z / r, -1.0, 1.0))  # polar angle [0, pi]
            phi   = np.arctan2(y, x)                        # azimuthal [-pi, pi]
            qml.RX(r * extra_weights[self.extra_weights_IDX['radius']], wires=i)
            qml.RY(theta,   wires=i)
            phi_offset = phi + extra_weights[self.extra_weights_IDX['elem_' + str(a)]]
            qml.RZ(phi_offset, wires=i)
            
            
            # qml.RX(np.pi/6,wires=i)
            # qml.RY(np.pi/4,wires=i)
            # qml.RZ(np.pi/3,wires=i)

    def _embed(self, node_feat: np.ndarray, extra_weights: pnp.ndarray, batched: bool) -> None:
        """
        Encode each atom's 3D coordinate as a direction on the Bloch sphere.

        For atom i, compute the polar (theta) and azimuthal (phi) angles of
        the vector (x, y, z) and apply RY(theta) followed by RZ(phi).
        Magnitude information is discarded.

        Args:
            node_feat : (n_qubits, 7) or (B, n_qubits, 7)
            batched   : True if a batch dimension is present.
        """
        if batched:
            x = np.float32(node_feat[:, :, 4]) - np.mean(node_feat[:, :, 4], axis=1, keepdims=True)
            y = np.float32(node_feat[:, :, 5]) - np.mean(node_feat[:, :, 5], axis=1, keepdims=True)
            z = np.float32(node_feat[:, :, 6]) - np.mean(node_feat[:, :, 6], axis=1, keepdims=True)
            a = np.int32(node_feat[:, :, 0])  # atomic number
        else:
            x = np.float32(node_feat[:, 4]) - np.mean(node_feat[:, 4])
            y = np.float32(node_feat[:, 5]) - np.mean(node_feat[:, 5])
            z = np.float32(node_feat[:, 6]) - np.mean(node_feat[:, 6])
            a = np.int32(node_feat[:, 0])  # atomic number

        r     = np.sqrt(x**2 + y**2 + z**2 + 1e-8)
        theta = np.arccos(np.clip(z / r, -1.0, 1.0))  # polar angle [0, pi]
        phi   = np.arctan2(y, x)                        # azimuthal [-pi, pi]
        #phi = np.pi/6 * np.ones_like(theta)
        # qml.RX(np.pi *extra_weights[self.extra_weights_IDX['elem_' + str(a)]], wires=i)
        # qml.RY(phi,   wires=i)
        # qml.RX(theta, wires=i)
        atom_angle=[]
        # First the pre-processing bla bla
        r_scaled=r * extra_weights[self.extra_weights_IDX['radius']]
        for z in a:
            atom_angle.append(np.pi*2*extra_weights[self.extra_weights_IDX['elem_' + str(z)]])
        # Now perform the embedding
        atom_angle = pnp.array(atom_angle, requires_grad=True)
        qml.AngleEmbedding(r_scaled+atom_angle, wires=self.all_wires, rotation='X')
        qml.AngleEmbedding(theta, wires=self.all_wires, rotation='Y')
        qml.AngleEmbedding(phi, wires=self.all_wires, rotation='Z')
    
    def _invariant_embed(self,extra_weights: pnp.ndarray, d_local: np.ndarray, theta_local: np.ndarray, phi_local: np.ndarray, atom_angle: pnp.ndarray, neighbor_id: int) -> None:
        """
        Placeholder for a more complex embedding operation that could include
        additional features or non-linear transformations.
        """
        # qml.AngleEmbedding(atom_angle, wires=self.all_wires, rotation='X')
        #for i in range(valid.shape[-1]):
        d_scaled = (1.0 - d_local[:, neighbor_id]) * np.pi
        qml.AngleEmbedding(atom_angle, wires=self.all_wires, rotation='Y')
        qml.AngleEmbedding(d_scaled, wires=self.all_wires, rotation='X')
        #qml.AngleEmbedding(theta_local[:, neighbor_id], wires=self.all_wires, rotation='Y')
        #qml.AngleEmbedding(phi_local[:, neighbor_id], wires=self.all_wires, rotation='X')
    
    def _encode_atomID(self, node_feat: np.ndarray, extra_weights: pnp.ndarray, batched: bool) -> None:
        """
        Minimal additional encoding: apply a small RX rotation proportional to the radial distance.
        """
        if batched:
            a = np.int32(node_feat[:, :, 0])  # atomic number
        else:
            a = np.int32(node_feat[:, 0])  # atomic number
        atom_angle=[]
        for z in a:
            atom_angle.append(np.pi*2*extra_weights[self.extra_weights_IDX['elem_' + str(z)]])
        atom_angle = pnp.array(atom_angle, requires_grad=True)
        qml.AngleEmbedding(atom_angle, wires=self.all_wires, rotation='Y')
        
    def _entangle(self, extra_weights: pnp.ndarray, node_feat: np.ndarray, edge_feat: np.ndarray, batched: bool, layer_id: int) -> None:
        """
        Apply pairwise entanglement gates over all bond pairs using distance and angular measures.

        Args:
            extra_weights : Extra trainable parameters (dist_coupling, sigmoid_k, sigmoid_mid, bond_*, out_scale, out_bias).
            node_feat     : (n_qubits, 7) or (B, n_qubits, 7)
            edge_feat     : (max_nodes, max_nodes, 4) or (B, max_nodes, max_nodes, 4)
            batched       : True if a batch dimension is present.
        """
        for (i, j) in self.two_comb_wires:
            if node_feat[...,i,0] < 0.5 or node_feat[...,j,0] < 0.5:  # Skip if either qubit corresponds to a non-existent atom (atomic number ~0)
                continue 
            if i>node_feat[...,i,-2] or j>node_feat[...,j,-2]:
                continue
            if batched: 
                bond_ij = edge_feat[:, i, j,0]
                theta_ij = edge_feat[:, i, j,1]
                phi_ij = edge_feat[:, i, j,2]
                d_ij = edge_feat[:, i, j,3]/self.dmax
            else:
                bond_ij = edge_feat[i, j,0]
                theta_ij = edge_feat[i, j,1]
                phi_ij = edge_feat[i, j,2]
                d_ij = edge_feat[i, j,3]/self.dmax
            
            distX_coupling = extra_weights[self.extra_weights_IDX['dist_coupling']]
            distY_coupling = extra_weights[self.extra_weights_IDX['dist_coupling_2']]
            bond_coupling = extra_weights[self.extra_weights_IDX['bond_' + str(int(bond_ij))]]
            if bond_ij > 0.:  # Only entangle if there's a bond and it's within the cutoff
                qml.IsingXX(distX_coupling * (1-d_ij)*np.cos(theta_ij), wires=[i, j])
                qml.IsingZZ(bond_coupling * np.pi, wires=[j, i])
                qml.IsingYY(distY_coupling * (1-d_ij)*np.cos(phi_ij), wires=[i, j])
                
    def _simple_entangle(self, edge_feat: np.ndarray = None) -> None:
        """
        Simple entanglement layer: apply CNOTs over all heavy-atom pairs.
        """
        for (i, j) in self.two_comb_wires:
            if edge_feat[i, j] > 0:  # Only entangle if there's a bond
                qml.CNOT(wires=[j, i])
    def _trainable_layers(self, weights: pnp.ndarray, layer: int) -> None:
        """
        Apply num_layers of per-qubit trainable RY rotations.

        Args:
            weights : (num_layers, n_qubits) trainable parameter array.
        """
  
        for i in self.all_wires:
            qml.Rot(weights[layer, 0, i],weights[layer, 1, i], weights[layer, 2, i], wires=i)  # RZ-RY-RZ rotation

    #------------------------------------------------------------------
    # QNode circuit function
    # ------------------------------------------------------------------
    def _trainable_measurement(self, extra_weights: pnp.ndarray) -> None:
        """
        Apply trainable axis rotations.

        Args:
            extra_weights : Extra trainable parameters (dist_coupling, sigmoid_k, sigmoid_mid, bond_*, out_scale, out_bias).
        """
        for i in range(self.read_qubits):
            qml.RY(extra_weights[self.extra_weights_IDX[f'meas_theta_{i}']], wires=i)
            qml.RZ(extra_weights[self.extra_weights_IDX[f'meas_phi_{i}']], wires=i)
        
    def _circuit(
        self,
        weights: pnp.ndarray,
        extra_weights: pnp.ndarray,
        node_feat: np.ndarray,
        edge_feat: np.ndarray,
    ) -> Any:
        """
        Full variational circuit: encode -> entangle -> trainable -> measure.

        Args:
            weights   : (num_layers, n_qubits) trainable parameters.
            node_feat : (max_nodes, 7) or (B, max_nodes, 7)
            edge_feat : (max_nodes, max_nodes, 4) or (B, max_nodes, max_nodes, 4)

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
        #theta_local, phi_local, d_local, atom_angle = self._invariant_features(node_feat, extra_weights, edge_feat, batched)
        self._encode_atomID(node_feat, extra_weights, batched)
        for layer in range(self.num_layers):
            self._entangle(extra_weights, node_feat, edge_feat, batched,layer)
            self._trainable_layers(weights,layer)
        self._trainable_measurement(extra_weights)    
        # fetch coeff1, coeff2, coeff3 from extra_weights and #construct the Hamiltonian as:
        self.coeffs = []
        for j in range(self.read_qubits):
            self.coeffs.append(extra_weights[self.extra_weights_IDX[f'coeff{j+1}']])
        self.coeffs = pnp.array(self.coeffs, requires_grad=True)
        obs = [qml.PauliZ(i) for i in range(self.read_qubits)]
        H = qml.Hamiltonian(self.coeffs, obs)
        return qml.expval(H)
    
    def _state_circuit(
        self,
        weights: pnp.ndarray,
        extra_weights: pnp.ndarray,
        node_feat: np.ndarray,
        edge_feat: np.ndarray,
    ) -> Any:
        """
        Full variational circuit: encode -> entangle -> trainable -> measure.

        Args:
            weights   : (num_layers, n_qubits) trainable parameters.
            node_feat : (max_nodes, 7) or (B, max_nodes, 7)
            edge_feat : (max_nodes, max_nodes, 4) or (B, max_nodes, max_nodes, 4)

        Returns:
            Full statevector of the circuit (2**n_qubits complex amplitudes).
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
        self._encode_atomID(node_feat, extra_weights, batched)
        for layer in range(self.num_layers):
            self._entangle(extra_weights, node_feat, edge_feat, batched, layer)
            self._trainable_layers(weights, layer)
        self._trainable_measurement(extra_weights)
        return qml.state()

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(
        self,
        weights: pnp.ndarray,
        extra_weights: pnp.ndarray,
        node_feat: np.ndarray,
        edge_feat: np.ndarray,
        target: np.ndarray,
    ) -> pnp.ndarray:
        """
        MAE loss between HOMO-LUMO gap and Z measurement.

        QM9 stores the gap in eV; it is (not really) converted to eV here, so the constant is 1.
        
        Args:
            weights   : (num_layers, n_qubits) trainable parameters.
            node_feat : (max_nodes, 7) or (B, max_nodes, 7)
            edge_feat : (max_nodes, max_nodes) or (B, max_nodes, max_nodes)
            target    : Scalar or (B,) HOMO-LUMO gap values in eV

        Returns:
            Scalar MAE loss.
        """
        gap_ev       = target * _HARTREE_TO_EV
        pred         = self.qnode(weights, extra_weights, node_feat, edge_feat)
        # add the bias term from extra weights
        bias = sum(pnp.abs(self.coeffs))
        pred = bias + pred
        return pnp.mean(pnp.abs(pred - gap_ev))
    
    def loss(
        self,
        weights: pnp.ndarray,
        extra_weights: pnp.ndarray,
        node_feat: np.ndarray,
        edge_feat: np.ndarray,
        target: np.ndarray,
    ) -> pnp.ndarray:
        """
        MSE loss between HOMO-LUMO gap and Z measurement.

        QM9 stores the gap in eV; it is (not really) converted to eV here, so the constant is 1.
        
        Args:
            weights   : (num_layers, n_qubits) trainable parameters.
            node_feat : (max_nodes, 7) or (B, max_nodes, 7)
            edge_feat : (max_nodes, max_nodes) or (B, max_nodes, max_nodes)
            target    : Scalar or (B,) HOMO-LUMO gap values in eV

        Returns:
            Scalar MAE loss.
        """
        gap_ev       = target * _HARTREE_TO_EV
        pred         = self.qnode(weights, extra_weights, node_feat, edge_feat)
        # add the bias term from extra weights
        bias = sum(pnp.abs(self.coeffs))
        pred = bias + pred
        return pnp.mean((pred - gap_ev)**2)
    
    def huber_loss(
            self,
            weights: pnp.ndarray,
            extra_weights: pnp.ndarray,
            node_feat: np.ndarray,
            edge_feat: np.ndarray,
            target: np.ndarray
    ):
        """
        Computes Huber Loss between target HOMO-LUMO gap and Z measurement
        """
        gap_ev       = target * _HARTREE_TO_EV
        pred         = self.qnode(weights, extra_weights, node_feat, edge_feat)
        # add the bias term from extra weights
        bias = sum(pnp.abs(self.coeffs))
        affine_bias = extra_weights[self.extra_weights_IDX['out_bias']]
        pred = bias + affine_bias - pred
        delta = 0.1#extra_weights[self.extra_weights_IDX['huber_delta']]
        residual = pred - gap_ev
        huber_loss = pnp.where(pnp.abs(residual) <= delta, 0.5 * residual**2, delta * pnp.abs(residual) - 0.5 * delta**2)
        return pnp.mean(huber_loss)
    
    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + qml.math.exp(-x))
    
    @staticmethod
    def _relu(x):
        return pnp.maximum(0, x)

    def _print_weight_summary(self):
        """
        Print extra weights along with key 
        """ 
        logger.info("Extra Weights:")
        for key, value in zip(self.extra_weights_IDX.keys(), self.extra_weights):
            logger.info(f"  {key}: {value}")

    # ------------------------------------------------------------------
    # Quantum Fisher Information computation
    # ------------------------------------------------------------------
    def quantum_fisher(
        self,
        node_feat: np.ndarray,
        edge_feat: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the Quantum Fisher Information Matrix (QFIM) for a single molecule.

        Evaluates the metric tensor of a state-returning QNode with respect to
        the trainable Rot-gate weights at fixed inputs (node_feat, edge_feat).
        Returns the submatrix corresponding to the rotation parameters only
        (shape: num_layers * 3 * n_qubits square).

        Args:
            node_feat : (n_qubits, 7) node feature array for a single molecule.
            edge_feat : (max_nodes, max_nodes, 4) edge feature array
                        (bond type, theta_ij, phi_ij, distance).

        Returns:
            np.ndarray of shape (num_layers * 3 * n_qubits, num_layers * 3 * n_qubits)
            — the QFIM restricted to the Rot-gate parameters.
        """
        # The HDF5 dataloader can yield torch tensors when convert_pnp=False.
        # Convert inputs here so the metric-tensor path uses the same interface
        # as the trainable weights and does not mix torch and autograd tensors.
        node_feat = pnp.array(np.asarray(node_feat), requires_grad=False)
        edge_feat = pnp.array(np.asarray(edge_feat), requires_grad=False)
        # The full metric tensor needs one auxiliary wire for Hadamard tests.
        # Keep the configured shot count so QFIM estimation follows the same
        # finite-shot execution mode as inference.
        metric_device = qml.device(
            self.device_name,
            wires=self.n_qubits + 1,
            shots=self.device.shots,
        )
        state_qnode = qml.QNode(
            self._state_circuit,
            metric_device,
            interface=self.backend,
            diff_method="best",
        )
        metric_fn = qml.metric_tensor(state_qnode, hybrid=False)
        full_qfi = metric_fn(self.weights, self.extra_weights, node_feat, edge_feat)
        # Extract submatrix for trainable Rot weights only (first num_layers*operations_per_layer*n_qubits params)
        n_rot = self.num_layers * self.operations_per_layer * self.n_qubits
        return full_qfi[:n_rot, :n_rot]
        
    def run_fisher_computation(
        self,
        dataloader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Fisher Information Matrix for each input point in the dataloader.
        
        Args:
            dataloader: DataLoader containing input data and labels (batch_size=1)
            
        Returns:
            Tuple of (fisher_matrices, labels) where:
            - fisher_matrices: (N, Np, Np) array of Fisher Information Matrices
            - labels: (N,) array of truth labels
        """
        fisher_matrices = []
        all_labels = []

        for node_feat, edge_feat, target, _ in tqdm(dataloader, desc="Computing Fisher Information"):
            qfi = self.quantum_fisher(node_feat[0], edge_feat[0])
            fisher_matrices.append(qfi)
            all_labels.append(target[0])

        fisher_matrices = np.array(fisher_matrices)  # (N, Np, Np)
        all_labels = np.array(all_labels)            # (N,)

        return fisher_matrices, all_labels
