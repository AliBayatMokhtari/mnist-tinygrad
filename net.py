from tinygrad.nn import Linear
from tinygrad.tensor import Tensor


class TinyNet:
    def __init__(self) -> None:
        self.l1 = Linear(784, 128, bias=False)
        self.l2 = Linear(128, 10, bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        x = x.leakyrelu()
        x = self.l2(x)
        return x
