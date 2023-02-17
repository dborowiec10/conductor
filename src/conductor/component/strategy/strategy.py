from conductor.component.strategy.gradient import GradientStrategy
from conductor.component.strategy.round_robin import RoundRobinStrategy
from conductor.component.strategy.sequential import SequentialStrategy
from conductor.component.strategy.trimmer import TrimmerStrategy
from conductor.component.strategy.one_shot import OneShotStrategy

strategies = {
    "gradient": GradientStrategy,
    "round_robin": RoundRobinStrategy,
    "sequential": SequentialStrategy,
    "trimmer": TrimmerStrategy,
    "one_shot": OneShotStrategy
}
