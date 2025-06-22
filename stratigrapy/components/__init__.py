from .sea_level import SeaLevelCalculator
from .weathering import BedrockWeatherer
from .diffusion import GravityDrivenRouter
from .stream_power import WaterDrivenRouter
from .stream_power import FluxDrivenRouter
from .landsliding import SimpleSedimentLandslider
from .compaction import SedimentCompactor

COMPONENTS = [
    SeaLevelCalculator,
    BedrockWeatherer,
    GravityDrivenRouter,
    WaterDrivenRouter,
    FluxDrivenRouter,
    SimpleSedimentLandslider,
    SedimentCompactor,
]

__all__ = [cls.__name__ for cls in COMPONENTS]
