from .SMO import SMO
from .StochasticSMO import StochasticSMO


def get_new_instance_of_optimizer(optimizer):
    if isinstance(optimizer, (SMO, StochasticSMO)):
        return type(optimizer)(**optimizer.get_config())
    elif optimizer.lower() == 'stochasticsmo':
        return StochasticSMO()
    elif optimizer.lower() == 'smo':
        return SMO()
    else:
        raise ValueError(f"{optimizer} is not a valid optimizer.")
