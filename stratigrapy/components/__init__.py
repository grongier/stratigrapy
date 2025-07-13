from .sea_level import SeaLevelCalculator
from .sea_level import CyclicSeaLevelCalculator
from .source import SourceProducer
from .weathering import BedrockWeatherer
from .diffusion import GravityDrivenDiffuser
from .diffusion import GravityDrivenRouter
from .stream_power import WaterDrivenDiffuser
from .stream_power import WaterDrivenRouter
from .stream_power import FluxDrivenRouter
from .landsliding import SimpleSedimentLandslider
from .compaction import SedimentCompactor

COMPONENTS = [
    SeaLevelCalculator,
    SourceProducer,
    CyclicSeaLevelCalculator,
    BedrockWeatherer,
    GravityDrivenDiffuser,
    GravityDrivenRouter,
    WaterDrivenDiffuser,
    WaterDrivenRouter,
    FluxDrivenRouter,
    SimpleSedimentLandslider,
    SedimentCompactor,
]

__all__ = [cls.__name__ for cls in COMPONENTS]
