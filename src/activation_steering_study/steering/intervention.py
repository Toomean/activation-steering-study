"""Register a fixed direction at one block's final token on each forward pass."""

from collections.abc import Callable

import torch


def register_intervention(
    block: torch.nn.Module, direction: torch.Tensor, alpha: float
) -> Callable[[], None]:
    """Add alpha * direction to the first sequence's final block output.

    The caller must call the returned function after the intervention.
    """
    # A forward hook can return a replacement output:
    # https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
    def add_direction(_module, _inputs, output):
        modified_output = output.clone()
        modified_output[0, -1] += alpha * direction
        return modified_output

    handle = block.register_forward_hook(add_direction)
    return handle.remove
