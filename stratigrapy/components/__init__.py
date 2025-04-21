from .sea_level import SeaLevelCalculator
from .weathering import BedrockWeatherer
from .diffusion import GravityDrivenDiffuser
from .stream_power import WaterDrivenDiffuser
from .landsliding import SimpleSedimentLandslider
from .compaction import SedimentCompactor

COMPONENTS = [
    SeaLevelCalculator,
    BedrockWeatherer,
    GravityDrivenDiffuser,
    WaterDrivenDiffuser,
    SimpleSedimentLandslider,
    SedimentCompactor,
]

__all__ = [cls.__name__ for cls in COMPONENTS]
