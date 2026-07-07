import torch
import torch.nn as nn

class PopulationCELoss(nn.Module):
    """Cross-entropy loss and accuracy for a whole population at once.

    Every individual classifies the same batch of targets independently, so the
    loss is summed across the population (giving each network its own gradient)
    while accuracy is reported per individual. Batch items that produce NaN/Inf
    logits for any individual are masked out so a single unstable network cannot
    corrupt the shared loss.
    """

    def __init__(self, batch_size, pop_size, n_classes):
        super(PopulationCELoss, self).__init__()
        self.batch_size = batch_size
        self.pop_size = pop_size
        self.n_classes = n_classes
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, output, target):
        """Computes the summed loss and per-individual accuracy.

        Args:
            output: Logits of shape ``[batch_size, pop_size, n_classes]``.
            target: Class labels of shape ``[batch_size]`` (shared across the
                population).

        Returns:
            Tuple ``(summed_loss, accuracies)`` where ``summed_loss`` is a
            scalar summed over the population and ``accuracies`` has shape
            ``[pop_size]``.
        """
        # Reshape output to [batch_size * pop_size, n_classes]
        output_reshaped = output.reshape(-1, self.n_classes)

        # Repeat target for each individual in the population
        target_repeated = target.unsqueeze(1).repeat(1, self.pop_size).view(-1)

        loss = self.loss(output_reshaped, target_repeated)
        loss = loss.view(self.batch_size, self.pop_size)
 
        # Check for NaNs and Infs in the original output tensor
        invalid_mask_elements = torch.isnan(output) | torch.isinf(output) # shape [batch_size, pop_size, n_classes]
        invalid_mask_individual_batch = invalid_mask_elements.any(dim=2) # Shape: [batch_size, pop_size]
        invalid_batch_items_mask = invalid_mask_individual_batch.any(dim=1) # Shape: [batch_size]

        # Set loss to 0 where NaNs or Infs are present in the corresponding batch item
        loss[invalid_batch_items_mask, :] = 0

        # Compute mean loss over the batch for each individual
        cleaned_loss_vector = loss.mean(dim=0) # Now mean over batch_size, result shape [pop_size]
        summed_loss = cleaned_loss_vector.sum()

        # Calculate the accuracy for each individual
        _, predicted = output.max(2) # Max over n_classes, result shape [batch_size, pop_size]
        target_for_accuracy = target.unsqueeze(1).repeat(1, self.pop_size)
        accuracies = predicted.eq(target_for_accuracy).float().mean(dim=0) # Mean over batch_size, result shape [pop_size]

        return summed_loss, accuracies
    

def accumulate_spike_loss(spike_count_tensor: torch.Tensor):
    """Sums spike counts into a scalar spike-cost term, ignoring NaN/Inf.

    Used as the energy/sparsity penalty that is added to the classification
    loss during training.
    """
    invalid_mask = torch.isnan(spike_count_tensor) | torch.isinf(spike_count_tensor)
    # Replace NaN/Inf with 0
    cleaned_spike_count = torch.where(
        invalid_mask,
        torch.tensor(0.0, device=spike_count_tensor.device, dtype=spike_count_tensor.dtype),
        spike_count_tensor
    )
    return cleaned_spike_count.sum()