import numpy as np
from tinygrad.jit import TinyJit
from tinygrad.tensor import Tensor
from net import TinyNet
from tinygrad.helpers import Timing


@TinyJit
def jit(net: TinyNet, x):
    return net(x).realize()


Tensor.training = False


def test_model(
        X_test: np.ndarray,
        Y_test: np.ndarray,
        net: TinyNet,
):
    with Timing("Time: "):
        avg_acc = 0
        for _ in range(1000):
            samp = np.random.randint(0, X_test.shape[0], size=(64))
            batch = Tensor(X_test[samp], requires_grad=False)
            labels = Y_test[samp]

            out = jit(net, batch)

            pred = out.argmax(axis=-1).numpy()
            avg_acc += (pred == labels).mean()

        print(f"Test Accuracy: {avg_acc / 1000}")
