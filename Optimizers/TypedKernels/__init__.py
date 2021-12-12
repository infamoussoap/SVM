from .DiskKernel import DiskKernel
from .LazyKernel import LazyKernel
from .StockKernel import StockKernel
from .DiskKernelV2 import DiskKernel as DiskKernelV2

kernels = {"stock": StockKernel, "lazy": LazyKernel, "disk": DiskKernel}


def get_kernel(kernel_type):
    val = kernels.get(kernel_type, None)
    if val is None:
        raise ValueError(f"{kernel_type} is not a valid kernel. Only {list(kernels.keys())} are valid kernel types.")
    else:
        return val
