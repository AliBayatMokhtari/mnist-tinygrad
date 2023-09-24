import sys
from tinygrad.nn.optim import SGD
from extra.datasets import fetch_mnist
from train import train_model
from test import test_model
from net import TinyNet
from tinygrad.nn.state import safe_save, load_state_dict, safe_load, get_state_dict

model_file_name = "model.safetensors"
train_command = "--train"
test_command = '--test'

# create model
net = TinyNet()

if __name__ == "__main__":
    args = sys.argv

    if len(args) < 2:
        raise Exception("You have to provide the command...")

    command = args[1]

    # throw error if command not found
    if command not in [train_command, test_command]:
        raise Exception("Invalid command :(")

    # fetch dataset
    X_train, Y_train, X_test, Y_test = fetch_mnist()

    if (command == "--train"):
        # setup optimizer
        opt = SGD([net.l1.weight, net.l2.weight], lr=3e-4)
        # train model with train dataset
        model = train_model(X_train=X_train, Y_train=Y_train, net=net, opt=opt)
        # get state dict
        state_dict = get_state_dict(net)
        # save model in a file
        safe_save(state_dict, model_file_name)
    else:
        # load state dict from file
        state_dict = safe_load(model_file_name)
        # load state dist to model
        load_state_dict(net, state_dict)
        # test model with test dataset
        test_model(X_test=X_test, Y_test=Y_test, net=net)
