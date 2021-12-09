from .SMO import SMO
from .StochasticSMO import StochasticSMO
from .CythonSMO import CythonSMO


def get_new_instance_of_optimizer(optimizer):
    if isinstance(optimizer, (SMO, StochasticSMO, CythonSMO)):
        return type(optimizer)(**optimizer.get_config())
    elif optimizer.lower() == 'stochasticsmo':
        return StochasticSMO()
    elif optimizer.lower() == 'smo':
        return SMO()
    elif optimizer.lower() == 'cythonsmo':
        return CythonSMO(1.0, 1e-2, 1e-2)
    else:
        raise ValueError(f"{optimizer} is not a valid optimizer.")
