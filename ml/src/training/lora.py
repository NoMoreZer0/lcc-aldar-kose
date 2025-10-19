"""
LoRA (Low-Rank Adaptation) implementation for efficient finetuning.
"""
import logging
from typing import Dict, List, Optional
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """
    LoRA layer that adds low-rank adaptation to a linear layer.

    Implements the LoRA method from "LoRA: Low-Rank Adaptation of Large Language Models"
    https://arxiv.org/abs/2106.09685
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: Rank of the low-rank matrices
            alpha: Scaling factor
            dropout: Dropout probability
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA layers."""
        # Low-rank path: B @ A @ x
        result = self.lora_B(self.lora_A(self.dropout(x)))
        return result * self.scaling


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.

    Combines a frozen pretrained linear layer with a trainable LoRA adaptation.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        """
        Args:
            base_layer: Pretrained linear layer to adapt
            rank: LoRA rank
            alpha: LoRA alpha
            dropout: LoRA dropout
        """
        super().__init__()

        # Store base layer (frozen)
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)

        # Create LoRA adaptation
        self.lora = LoRALayer(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: base output + LoRA adaptation."""
        base_output = self.base_layer(x)
        lora_output = self.lora(x)
        return base_output + lora_output

    def merge_weights(self):
        """Merge LoRA weights into base layer for inference."""
        if self.lora is None:
            return

        # Compute merged weight: W + (B @ A) * scaling
        with torch.no_grad():
            delta_w = self.lora.lora_B.weight @ self.lora.lora_A.weight
            delta_w = delta_w * self.lora.scaling
            self.base_layer.weight.data += delta_w

        # Remove LoRA layers
        self.lora = None

    def unmerge_weights(self):
        """Separate LoRA weights from base layer."""
        raise NotImplementedError("Unmerging not implemented")


def find_target_modules(
    model: nn.Module,
    target_modules: List[str],
) -> Dict[str, nn.Module]:
    """
    Find all modules matching target names.

    Args:
        model: Model to search
        target_modules: List of target module names (can include wildcards)

    Returns:
        Dict mapping module names to modules
    """
    target_modules_dict = {}

    for name, module in model.named_modules():
        # Check if this module matches any target pattern
        for target in target_modules:
            # Support wildcard matching
            pattern = target.replace("*", ".*")
            if re.search(pattern, name):
                if isinstance(module, nn.Linear):
                    target_modules_dict[name] = module
                    break

    logger.info(f"Found {len(target_modules_dict)} target modules for LoRA")
    return target_modules_dict


def apply_lora(
    model: nn.Module,
    lora_config: Dict,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Apply LoRA to a model.

    Args:
        model: Model to apply LoRA to
        lora_config: LoRA configuration dict
        target_modules: List of target module names (if None, use config)

    Returns:
        Model with LoRA applied
    """
    if target_modules is None:
        target_modules = lora_config.get("target_modules", [
            "to_q", "to_k", "to_v", "to_out.0",
            "add_k_proj", "add_v_proj",
        ])

    rank = lora_config.get("rank", 4)
    alpha = lora_config.get("alpha", 1.0)
    dropout = lora_config.get("dropout", 0.0)

    # Find all target modules
    modules_to_replace = find_target_modules(model, target_modules)

    # Replace with LoRA versions
    for name, module in modules_to_replace.items():
        # Get parent module and attribute name
        *parent_path, attr_name = name.split(".")

        parent = model
        for p in parent_path:
            parent = getattr(parent, p)

        # Create LoRA linear layer
        lora_linear = LoRALinear(
            base_layer=module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        # Replace module
        setattr(parent, attr_name, lora_linear)
        logger.debug(f"Replaced {name} with LoRA layer (rank={rank})")

    # Freeze all parameters except LoRA
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable LoRA parameters: {trainable_params:,}")
    logger.info(f"LoRA parameters: {trainable_params / total_params * 100:.2f}%")

    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge all LoRA weights into base layers for inference.

    Args:
        model: Model with LoRA layers

    Returns:
        Model with merged weights
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()

    logger.info("Merged LoRA weights into base model")
    return model


def save_lora_weights(model: nn.Module, save_path: str):
    """
    Save only LoRA weights to a file.

    Args:
        model: Model with LoRA layers
        save_path: Path to save weights
    """
    lora_state_dict = {}

    for name, param in model.named_parameters():
        if "lora" in name and param.requires_grad:
            lora_state_dict[name] = param.cpu()

    torch.save(lora_state_dict, save_path)
    logger.info(f"Saved {len(lora_state_dict)} LoRA parameters to {save_path}")


def load_lora_weights(model: nn.Module, load_path: str):
    """
    Load LoRA weights from a file.

    Args:
        model: Model with LoRA layers
        load_path: Path to load weights from
    """
    lora_state_dict = torch.load(load_path)

    # Load state dict
    missing, unexpected = model.load_state_dict(lora_state_dict, strict=False)

    logger.info(f"Loaded {len(lora_state_dict)} LoRA parameters from {load_path}")
    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")


class LoRAConfig:
    """Configuration for LoRA."""

    def __init__(
        self,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        bias: str = "none",
    ):
        """
        Args:
            rank: LoRA rank
            alpha: LoRA alpha (scaling factor)
            dropout: LoRA dropout
            target_modules: List of module names to apply LoRA to
            bias: Bias configuration ("none", "all", "lora_only")
        """
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or [
            "to_q", "to_k", "to_v", "to_out.0",
        ]
        self.bias = bias

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
        }


# PEFT (Parameter-Efficient Fine-Tuning) integration
try:
    from peft import LoraConfig, get_peft_model, PeftModel

    def apply_peft_lora(model: nn.Module, lora_config: Dict) -> nn.Module:
        """
        Apply LoRA using the PEFT library (if available).

        Args:
            model: Model to apply LoRA to
            lora_config: LoRA configuration dict

        Returns:
            PEFT model with LoRA
        """
        peft_config = LoraConfig(
            r=lora_config.get("rank", 4),
            lora_alpha=lora_config.get("alpha", 4),
            target_modules=lora_config.get("target_modules"),
            lora_dropout=lora_config.get("dropout", 0.0),
            bias=lora_config.get("bias", "none"),
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        return model

except ImportError:
    logger.info("PEFT library not available, using custom LoRA implementation")

    def apply_peft_lora(model: nn.Module, lora_config: Dict) -> nn.Module:
        """Fallback to custom implementation."""
        return apply_lora(model, lora_config)
