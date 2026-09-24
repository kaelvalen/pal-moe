"""
The S1 architecture contract (docs/ARCHITECTURE_CONTRACT.md).

    X --Backbone--> Z --Expert--> Z' --Readout--> Y
                          ^
                          |
                        Router

Importing this package registers every implementation, so a config string can
select one without the training loop importing it:

    from pal_moe.arch import describe, build_readout, build_expert

Nothing here changes v1 behaviour. The v1 modules (`SharedEncoder`,
`MLPExpert`, the three routers, `DynamicMoE`) are wrapped or re-exposed, never
edited, and `pal_moe.factory` keeps building them exactly as before.
"""

# Import for the side effect of registering every implementation.
from . import backbones as _backbones  # noqa: F401,E402
from . import experts as _experts  # noqa: F401,E402
from . import readouts as _readouts  # noqa: F401,E402
from . import routers as _routers  # noqa: F401,E402
from .backbones import (
    CachedBackbone,
    ProjectedBackbone,
    RandomProjectionBackbone,
    SharedEncoderBackbone,
    build_projected_backbone,
    wrap_encoder,
)
from .experts import (
    IdentityExpert,
    LegacyClassificationExpert,
    ResidualAdapter,
    ResidualMLPExpert,
)
from .protocols import (
    Backbone,
    ClassificationExpert,
    Readout,
    RepresentationExpert,
    Router,
    identity_check,
    is_classification_expert,
    is_representation_expert,
)
from .readouts import (
    CosineReadout,
    LinearReadout,
    LogisticReadout,
    MLPReadout,
    NCMReadout,
    RidgeReadout,
    mask_unseen,
    trainable_hook,
)
from .registry import (
    BACKBONES,
    CLASSIFICATION_EXPERTS,
    EXPERTS,
    READOUTS,
    ROUTERS,
    build_backbone,
    build_expert,
    build_readout,
    build_router,
    describe,
)
from .routers import LegacyRouter, PrototypeRouter, build_legacy_router

__all__ = [
    # protocols
    "Backbone",
    "RepresentationExpert",
    "ClassificationExpert",
    "Readout",
    "Router",
    "is_representation_expert",
    "is_classification_expert",
    "identity_check",
    # registries
    "BACKBONES",
    "EXPERTS",
    "CLASSIFICATION_EXPERTS",
    "READOUTS",
    "ROUTERS",
    "build_backbone",
    "build_expert",
    "build_readout",
    "build_router",
    "build_legacy_router",
    "describe",
    # implementations
    "SharedEncoderBackbone",
    "CachedBackbone",
    "RandomProjectionBackbone",
    "ProjectedBackbone",
    "build_projected_backbone",
    "wrap_encoder",
    "IdentityExpert",
    "ResidualAdapter",
    "ResidualMLPExpert",
    "LegacyClassificationExpert",
    "NCMReadout",
    "CosineReadout",
    "LinearReadout",
    "LogisticReadout",
    "RidgeReadout",
    "MLPReadout",
    "mask_unseen",
    "trainable_hook",
    "PrototypeRouter",
    "LegacyRouter",
]
