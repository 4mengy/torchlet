from torch import Tensor


class ForwardParams:
    def __init__(self, req_indptr_cpu: Tensor):
        self.req_indptr_cpu = req_indptr_cpu
