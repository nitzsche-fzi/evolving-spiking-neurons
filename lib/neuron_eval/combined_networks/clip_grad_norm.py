import torch

def clip_population_grad_norm(model: torch.nn.Module, max_norm: float, eps: float = 1e-6):
    """
    Per-population gradient clipping.
    Never reduces over population dimension:
        grad[p, ...] is treated as the gradient of individual network p.
    """
    sq_norms = None

    # ------------- accumulate squared norms per population -------------
    for layer in model.dense_layers:
        grad = layer.weight.grad
        if grad is None:
            continue

        # shape: [pop, in, out] -> [pop, D]
        g = grad.flatten(1)
        layer_sq = (g * g).sum(dim=1)   # [pop]

        sq_norms = layer_sq if sq_norms is None else sq_norms + layer_sq

    if sq_norms is None:
        return  # no grads present

    norms = (sq_norms + eps).sqrt()                   # [pop]
    scale = (max_norm / norms).clamp(max=1.0)         # [pop]

    # ------------- apply scaling -------------
    for layer in model.dense_layers:
        grad = layer.weight.grad
        if grad is None:
            continue

        g = grad.flatten(1)                           # still writable
        g.mul_(scale.unsqueeze(1))
