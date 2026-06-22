"""Paper-facing PrefMLLM module aliases.

The original research branch used the internal "name_memory" label for the
closed-user personalization state.  The release keeps those checkpoint keys for
compatibility, while exposing the paper terminology here.
"""

from llava.model.language_model.llava_llama_name_memory import LlavaLlamaNameMemoryWrapper
from llava.model.memory import (
    PREFMLLM_FACTORS,
    FactorizedUserAwareHierarchicalMoE,
    FactorizedUserStateMemory,
)

PrefMLLMWrapper = LlavaLlamaNameMemoryWrapper

__all__ = [
    "PREFMLLM_FACTORS",
    "FactorizedUserStateMemory",
    "FactorizedUserAwareHierarchicalMoE",
    "PrefMLLMWrapper",
]
