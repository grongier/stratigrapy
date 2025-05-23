from .sea_level import SeaLevelCalculator
from .weathering import BedrockWeatherer
from .diffusion import GravityDrivenDiffuser
from .stream_power import WaterDrivenDiffuser
from .stream_power import WaterDrivenDisplacer
from .landsliding import SimpleSedimentLandslider
from .compaction import SedimentCompactor

COMPONENTS = [
    SeaLevelCalculator,
    BedrockWeatherer,
    GravityDrivenDiffuser,
    WaterDrivenDiffuser,
    WaterDrivenDisplacer,
    SimpleSedimentLandslider,
    SedimentCompactor,
]

__all__ = [cls.__name__ for cls in COMPONENTS]
