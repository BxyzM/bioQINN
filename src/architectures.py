"""
Modular Quantum Classifier for arbitrary quantum circuit architectures.

Improved version of original 1P1Q architecture so that the user can supply custom encoding, entanglement and trainable operations.

Author: Dr. Aritra Bal (ETP)
Date: December 01, 2025
"""

import pennylane as qml
import numpy as np
from typing import Callable, Optional, Tuple, Any, List
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
import time

class QuantumClassifier:
    """
    Modular quantum classifier with user-defined circuit components.
    
    This class implements a flexible quantum circuit where the encoding,
    entanglement, and trainable operations are provided by the user as
    callable functions. This allows for arbitrary circuit architectures
    while maintaining consistent interfaces for training and inference.
    
    Args:
        wires: Number of qubits in the circuit
        device_name: PennyLane device name
        backend: Backend interface ('autograd', 'torch', 'jax', 'tf')
        shots: Number of measurement shots (None for exact statevector)
        encoding: Callable for data encoding that accepts (inputs, wires)
        entanglement: Callable for entanglement layer that accepts (wires,)
        operation: Callable for trainable operations that accepts (weights, wires)
        initial_weights: Initial parameter values for the circuit
        approx: Approximation method for metric tensor ('block-diag', None)
        use_aux_wire: Whether to use auxiliary wire for metric tensor computation
    """
    
    def __init__(
        self,
        wires: int = 5,
        device_name: str = 'default.qubit',
        backend: str = 'autograd',
        shots: Optional[int] = None,
        encoding: Optional[Callable] = None,
        entanglement: Optional[Callable] = None,
        operation: Optional[Callable] = None,
        initial_weights: Optional[np.ndarray] = None,
        approx: Optional[str] = None,
        use_aux_wire: bool = False,
        **kwargs: Any
    ) -> None:
        """Initialize the quantum classifier with user-defined components."""
        
        # Circuit configuration
        self.n_qubits = wires
        self.backend = backend
        self.all_wires = list(range(self.n_qubits))
        self.approx = approx
        self.use_aux_wire = use_aux_wire
        if self.approx is not None:
            logger.info(f"Using approximation method for metric tensor: {self.approx}")
            logger.warning("Approximation will absolutely affect accuracy of Fisher information calculations, there is no point of this calculation if only diagonal (self-correlation) elements are computed. Press Ctrl+C to abort now if you did not intend this.")
            time.sleep(5)
        # Auxiliary wire for metric tensor computation
        if self.use_aux_wire:
            self.aux_wire = self.n_qubits
            device_wires = self.n_qubits + 1
        else:
            self.aux_wire = None
            device_wires = self.n_qubits
        
        # User-supplied circuit components
        self.encoding = encoding
        self.entanglement = entanglement
        self.operation = operation
        
        # Validate that required components are provided
        if self.encoding is None:
            logger.warning("No encoding function provided - circuit will have no data encoding")
        if self.operation is None:
            logger.warning("No operation function provided - circuit will have no trainable parameters")
        
        # Initialize device
        self.device = qml.device(device_name, wires=device_wires, shots=shots)
        logger.info(f"Device initialized: {self.device}")
        
        # Initialize weights
        self.current_weights = initial_weights
        if self.current_weights is not None:
            logger.info(f"Weights initialized with shape {self.current_weights.shape}")
        
        # QNodes (to be set up later)
        self.circuit_qnode: Optional[qml.QNode] = None
        self.state_circuit_qnode: Optional[qml.QNode] = None
        
        # Optional batch tracking for Fisher computation
        self.total_batches = kwargs.get('total_batches', None)
    
    def _circuit(
        self,
        weights: np.ndarray,
        inputs: np.ndarray
    ) -> Any:
        """
        Core quantum circuit for classification.
        
        Applies encoding, entanglement, and trainable operations in sequence.
        Returns the expectation value of Pauli-Z on the first qubit.
        
        Args:
            weights: Circuit parameters for trainable operations
            inputs: Input data with shape (N,N,F) or (B,N,N,F)
                   If B=1, the batch dimension is removed automatically
        
        Returns:
            Expectation value of Pauli-Z measurement on wire 0
        """
        # Handle batch dimension
        if inputs.ndim == 4:
            if inputs.shape[0] == 1:
                # Remove batch dimension: (1,N,N,F) -> (N,N,F)
                inputs = np.squeeze(inputs, axis=0)
            else:
                # For batch_size > 1, process first sample
                # TODO: Extend to full batch processing, PennyLane should be able to handle this shit nowadays 
                logger.warning(f"Batch size > 1 detected ({inputs.shape[0]}), processing first sample only")
                inputs = inputs[0]
        
        # Apply encoding (data -> quantum state)
        if self.encoding is not None:
            self.encoding(inputs=inputs, wires=self.all_wires)
        
        # Apply entanglement layer
        if self.entanglement is not None:
            self.entanglement(wires=self.all_wires)
        
        # Apply trainable operations
        if self.operation is not None:
            self.operation(weights=weights, wires=self.all_wires)
        
        # Measure Z expectation on first qubit
        return qml.expval(qml.PauliZ(0))
    
    def _state_circuit(
        self,
        weights: np.ndarray,
        inputs: np.ndarray
    ) -> Any:
        """
        Quantum circuit that returns the full quantum state.
        
        Identical to _circuit but returns the complete state vector
        instead of a measurement. Used for quantum Fisher information
        calculations and state analysis.
        
        Args:
            weights: Circuit parameters for trainable operations
            inputs: Input data with shape (N,N,F) or (B,N,N,F)
        
        Returns:
            Full quantum state vector
        """
        # Handle batch dimension
        if inputs.ndim == 4:
            if inputs.shape[0] == 1:
                inputs = np.squeeze(inputs, axis=0)
            else:
                logger.warning(f"Batch size > 1 detected ({inputs.shape[0]}), processing first sample only")
                inputs = inputs[0]
        
        # Apply encoding
        if self.encoding is not None:
            self.encoding(inputs=inputs, wires=self.all_wires)
        
        # Apply entanglement
        if self.entanglement is not None:
            self.entanglement(wires=self.all_wires)
        
        # Apply trainable operations
        if self.operation is not None:
            self.operation(weights=weights, wires=self.all_wires)
        
        # Return full quantum state
        return qml.state()
    
    def set_circuit(self) -> None:
        """
        Initialize the QNode circuits for measurement and state retrieval.
        
        Creates two QNodes:
        1. circuit_qnode: For measurements (expectation values)
        2. state_circuit_qnode: For full state vector retrieval
        """
        self.circuit_qnode = qml.QNode(
            self._circuit,
            self.device,
            interface=self.backend
        )
        
        self.state_circuit_qnode = qml.QNode(
            self._state_circuit,
            self.device,
            interface=self.backend
        )
        
        logger.info(f"Circuit QNodes initialized with backend: {self.backend}")
    
    def fetch_circuit(self) -> qml.QNode:
        """
        Get the measurement circuit QNode.
        
        Initializes the circuit if not already set.
        
        Returns:
            Configured QNode for circuit execution
        """
        if self.circuit_qnode is None:
            self.set_circuit()
        return self.circuit_qnode
    
    def state_circuit(
        self,
        weights: np.ndarray,
        inputs: np.ndarray
    ) -> np.ndarray:
        """
        Execute the state circuit for a given input.
        
        Args:
            weights: Circuit parameters
            inputs: Input data (will be reshaped if needed)
        
        Returns:
            Quantum state vector
        """
        if self.state_circuit_qnode is None:
            self.set_circuit()
        
        # Ensure inputs have batch dimension
        if inputs.ndim == 3:
            inputs = inputs[np.newaxis, ...]  # Add batch dimension
        
        return self.state_circuit_qnode(weights, inputs)
    
    def quantum_fisher(
        self,
        input_point: np.ndarray,
        use_adjoint: bool = False
    ) -> np.ndarray:
        """
        Compute the Quantum Fisher Information Matrix (QFIM).
        
        The QFIM characterizes the distinguishability of quantum states
        parameterized by the circuit weights. It is computed using either
        the standard metric tensor or the adjoint method.
        
        Args:
            input_point: Single input data point
            use_adjoint: If True, use adjoint_metric_tensor (faster but less general)
        
        Returns:
            Quantum Fisher Information Matrix with shape (n_params, n_params)
        """
        if self.state_circuit_qnode is None:
            self.set_circuit()
        
        if self.current_weights is None:
            raise ValueError("Weights not initialized. Set weights before computing Fisher information.")
        
        # Ensure input has correct shape
        if input_point.ndim == 3:
            input_point = input_point[np.newaxis, ...]  # Add batch dimension
        elif input_point.ndim == 2:
            logger.warning("Input is 2D, reshaping by adding batch and channel dimensions")
            # Assume it's (N,N) and add batch and channel dims
            input_point = input_point[np.newaxis, ..., np.newaxis]
        
        # Define metric tensor function
        if use_adjoint:
            metric_fn = lambda w: qml.adjoint_metric_tensor(self.state_circuit_qnode)(w, input_point)
        else:
            if self.use_aux_wire:
                metric_fn = lambda w: qml.metric_tensor(
                    self.state_circuit_qnode,
                    hybrid=True,
                    aux_wire=self.aux_wire,
                    approx=self.approx
                )(w, input_point)
            else:
                metric_fn = lambda w: qml.metric_tensor(
                    self.state_circuit_qnode,
                    approx=self.approx
                )(w, input_point)
        
        # Compute QFIM
        qfim = metric_fn(self.current_weights)
        
        return qfim
    
    def run_fisher_computation(
        self,
        dataloader: DataLoader,
        use_adjoint: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute QFIM for all samples in a dataloader.
        
        Args:
            dataloader: DataLoader with batch_size=1
            use_adjoint: Whether to use adjoint method for metric tensor
        
        Returns:
            Tuple of (fisher_matrices, labels) where:
            - fisher_matrices: Array of shape (N, n_params, n_params)
            - labels: Array of shape (N,) containing ground truth labels
        """
        fisher_matrices = []
        all_labels = []
        
        total = self.total_batches if self.total_batches is not None else len(dataloader)
        
        for batch_inputs, batch_labels in tqdm(dataloader, desc="Computing QFIM", total=total):
            # Extract single sample (batch_size should be 1)
            input_point = batch_inputs[0] if batch_inputs.shape[0] == 1 else batch_inputs
            
            # Compute QFIM
            qfim = self.quantum_fisher(input_point, use_adjoint=use_adjoint)
            fisher_matrices.append(qfim)
            
            # Extract label
            label = batch_labels[0] if batch_labels.ndim > 0 else batch_labels
            all_labels.append(label)
        
        # Convert to numpy arrays
        fisher_matrices = np.array(fisher_matrices)
        all_labels = np.array(all_labels)
        
        logger.info(f"Fisher computation complete. Shape: {fisher_matrices.shape}")
        return fisher_matrices, all_labels
    
    def load_weights(
        self,
        weights: np.ndarray,
        requires_grad: bool = False
    ) -> None:
        """
        Load circuit parameters.
        
        Args:
            weights: Parameter array to load
            requires_grad: Whether to enable gradient computation
        """
        self.current_weights = np.array(weights, requires_grad=requires_grad)
        logger.info(f"Weights loaded with shape {self.current_weights.shape}")
    
    def load_weights_from_file(
        self,
        filepath: str,
        weight_key: str = 'weights',
        requires_grad: bool = False
    ) -> None:
        """
        Load weights from a saved file.
        
        Args:
            filepath: Path to file containing weights
            weight_key: Key in the dictionary/file containing weights
            requires_grad: Whether to enable gradient computation
        """
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        weights = data[weight_key] if isinstance(data, dict) else data
        self.load_weights(weights, requires_grad=requires_grad)
        logger.info(f"Weights loaded from {filepath}")
    
    def run_inference(
        self,
        dataloader: DataLoader,
        loss_fn: Callable,
        return_scores: bool = True,
        **loss_kwargs
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Run inference on dataset.
        
        Args:
            dataloader: DataLoader containing input data and labels
            loss_fn: Loss function with signature (weights, inputs, labels, circuit, ...)
            return_scores: Whether to return circuit output scores
            **loss_kwargs: Additional arguments for loss function
        
        Returns:
            Tuple of (costs, scores) if return_scores=True, else (costs,)
        """
        if self.current_weights is None:
            raise ValueError("Weights not initialized. Load weights before running inference.")
        
        if self.circuit_qnode is None:
            self.set_circuit()
        
        all_costs = []
        all_scores = [] if return_scores else None
        
        total = self.total_batches if self.total_batches is not None else len(dataloader)
        
        for batch_inputs, batch_labels in tqdm(dataloader, desc="Running inference", total=total):
            # Compute loss
            result = loss_fn(
                self.current_weights,
                inputs=batch_inputs,
                labels=batch_labels,
                quantum_circuit=self.circuit_qnode,
                return_scores=return_scores,
                **loss_kwargs
            )
            
            if return_scores:
                cost, score = result
                all_costs.append(float(cost))
                all_scores.append(float(score))
            else:
                all_costs.append(float(result))
        
        costs = np.array(all_costs)
        
        if return_scores:
            scores = np.array(all_scores)
            logger.info(f"Inference complete. Mean cost: {costs.mean():.4f}")
            return costs, scores
        else:
            logger.info(f"Inference complete. Mean cost: {costs.mean():.4f}")
            return costs
    
    def get_circuit_info(self) -> dict:
        """
        Get information about the current circuit configuration.
        
        Returns:
            Dictionary containing circuit configuration details
        """
        return {
            'n_qubits': self.n_qubits,
            'backend': self.backend,
            'device': str(self.device),
            'wires': self.all_wires,
            'has_encoding': self.encoding is not None,
            'has_entanglement': self.entanglement is not None,
            'has_operation': self.operation is not None,
            'weights_shape': self.current_weights.shape if self.current_weights is not None else None,
            'use_aux_wire': self.use_aux_wire,
            'approx': self.approx
        }
    
    def print_circuit_info(self) -> None:
        """Print current circuit configuration."""
        info = self.get_circuit_info()
        logger.info("Circuit Configuration:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")