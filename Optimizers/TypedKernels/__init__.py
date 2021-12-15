from .DiskKernel import DiskKernel
from .LazyKernel import LazyKernel
from .StockKernel import StockKernel

from .DiskKernelV2 import DiskKernel as DiskKernelV2
from .LazyKernelV2 import LazyKernel as LazyKernelV2
from .AdaptiveKernel import AdaptiveKernel


kernels = {"stock": StockKernel, "lazy": LazyKernel, "disk": DiskKernel,
           "adaptive": AdaptiveKernel, "lazyv2": LazyKernelV2, "diskv2": DiskKernelV2}


def get_kernel(kernel_type):
    val = kernels.get(kernel_type, None)
    if val is None:
        raise ValueError(f"{kernel_type} is not a valid kernel. Only {list(kernels.keys())} are valid kernel types.")
    else:
        return val
