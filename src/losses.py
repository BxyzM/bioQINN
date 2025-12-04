import pennylane.numpy as np

def quantum_loss(weights, inputs, labels, quantum_circuit, return_scores=False, loss_type='MSE'):
    """
    Quantum circuit loss function.
    
    Args:
        weights: Circuit parameters
        inputs: Input data batch
        labels: Ground truth labels
        quantum_circuit: QNode for circuit execution
        return_scores: Whether to return predictions
        loss_type: Loss function type
        
    Returns:
        Loss value or (loss, scores) tuple
    """
    # Handle batch dimension
    # Execute circuit for each sample
    predictions = np.array(quantum_circuit(weights, inputs),requires_grad=True)
    # Compute loss
    if loss_type == 'MSE':
        loss = np.mean((predictions - labels) ** 2)
    elif loss_type == 'BCE':
        # Binary cross-entropy with clipping for stability
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        loss = -np.mean(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))
    elif loss_type == 'MAE':
        loss = np.mean(np.abs(predictions - labels))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    if return_scores:
        return loss, predictions
    return loss