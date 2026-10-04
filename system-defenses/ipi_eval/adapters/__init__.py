"""Defense adapters. Each maps a system-defenses/<Name>/impl/defense.py onto DefenseAdapter."""
from .baseline import BaselineDefense
from .firewalls import FirewallsDefense
from .fides import FidesDefense
from .melon import MelonDefense
from .causalarmor import CausalArmorDefense
from .ipiguard import IPIGuardDefense
from .camel import CamelDefense

REGISTRY = {
    "baseline": BaselineDefense,
    "firewalls": FirewallsDefense,
    "fides": FidesDefense,
    "melon": MelonDefense,                                            # DEFAULT: OpenAI text-embedding-3-large (θ=0.8, paper); key from secrets.md
    "melon-aug": lambda **k: MelonDefense(augment=True, **k),         # MELON-Aug (repeat-prompt), openai embeddings
    "melon-bge": lambda **k: MelonDefense(embed_backend="bge", **k),  # local BGE substitute (θ=0.9); no key needed
    "melon-aug-bge": lambda **k: MelonDefense(augment=True, embed_backend="bge", **k),
    "melon-openai": lambda **k: MelonDefense(embed_backend="openai", **k),       # explicit alias of the default
    "melon-aug-openai": lambda **k: MelonDefense(augment=True, embed_backend="openai", **k),
    "causalarmor": CausalArmorDefense,                                # CoT masking ON (default)
    "causalarmor-nocot": lambda **k: CausalArmorDefense(cot_masking=False, **k),  # CoT masking OFF
    "ipiguard": IPIGuardDefense,
    "camel": CamelDefense,
}


def get_defense(name: str, **kwargs):
    if name not in REGISTRY:
        raise KeyError(f"unknown defense '{name}'. available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)
