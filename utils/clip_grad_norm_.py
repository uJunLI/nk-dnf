import jittor as jt
import math
from typing import Iterable, Optional, List, Union, Tuple

def clip_grad_norm_(
    parameters: Union[jt.Var, Iterable[jt.Var]],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> float:
    """
    Clips gradient norm of an iterable of parameters in Jittor.

    Args:
        parameters (Iterable[jt.Var] or jt.Var): Parameters with gradients to normalize.
        max_norm (float): Maximum norm of the gradients.
        norm_type (float): Type of the used p-norm. Defaults to 2.0.
        error_if_nonfinite (bool): If True, raises an error if the total norm is NaN or Inf.

    Returns:
        float: Total norm of the gradients.
    """
    # Ensure parameters is an iterable
    if isinstance(parameters, jt.Var):
        parameters = [parameters]

    # Filter parameters with valid gradients
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return 0.0

    max_norm = float(max_norm)
    norm_type = float(norm_type)

    # Compute the total norm
    if norm_type == math.inf:
        total_norm = max(g.abs().max().item() for g in grads)
    else:
        total_norm = sum((g.norm(norm_type).item() ** norm_type for g in grads)) ** (1.0 / norm_type)

    # Check for non-finite norm
    if error_if_nonfinite and (math.isnan(total_norm) or math.isinf(total_norm)):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients is non-finite. "
            "Cannot clip gradients. To ignore this error, set `error_if_nonfinite=False`."
        )

    # Compute clipping coefficient
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = min(1.0, clip_coef)

    # Scale gradients in-place
    for g in grads:
        g.update(g * clip_coef)

    return total_norm
