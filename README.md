# Installation Steps

- Clone this repository.
- `cd` to `./mnist-tinygrad`.
- Create a virtual environment using `python3 -m venv env`.
- Activate environment using `source ./env/bin/activate` (it's for Linux and Mac, use `.\env\Scripts\activate` for Windows).
- Run `which python` to make sure that your python interpreter comes from your virtual environment. Use `where python` for Windows.
- Install Tinygrad with the official documentation ([link](https://github.com/tinygrad/tinygrad#installation)).

# Usage

Just run one of the following commands to run the model in test or train mode:

- To train the model: `python main.py --train`.
- To test the model: `python main.py --test`.
