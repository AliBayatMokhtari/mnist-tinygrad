import numpy as np
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import get_state_dict, safe_save
from tinygrad.tensor import Tensor
from net import TinyNet


Tensor.training = True


def train_model(
        X_train: np.ndarray,
        Y_train: np.ndarray,
        net: TinyNet,
        opt: SGD
):
    for step in range(1000):
        samp = np.random.randint(0, X_train.shape[0], size=(64))
        batch = Tensor(X_train[samp], requires_grad=False)
        labels = Tensor(Y_train[samp])

        out = net(batch)

        loss = Tensor.sparse_categorical_crossentropy(out, labels)

        opt.zero_grad()

        loss.backward()

        opt.step()

        pred = out.argmax(axis=-1)
        acc = (pred == labels).mean()

        if step % 100 == 0:
            print(
                f"Step {step+1} | Loss: {loss.numpy()} | Accuracy: {acc.numpy()}")
