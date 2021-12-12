import pyximport
pyximport.install(language_level=3)

from .SMO import SMO
from .StochasticSMO import StochasticSMO
from .CythonSMO import CythonSMO
from .CythonStochasticSMO import CythonStochasticSMO

from .get_optimizer import get_new_instance_of_optimizer
