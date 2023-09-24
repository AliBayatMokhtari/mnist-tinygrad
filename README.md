# Installation

- Step 1

  - Clone this repository
  - `cd` to `./mnist-tinygrad`.
  - Create a virutuale enviroment using `python3 -m venv env`.
  - Activate enviroment using `source ./env/bin/activate` (it's for linux and mac, for windows use `.\env\Scripts\activate`).
  - Run `which python` to make sure that your python interpreter comes from your virtual enviroment. Use `where python` for Windows.
  - Install `tinygrad` with the offical documentation ([link](https://github.com/tinygrad/tinygrad#installation)).

# Usage

Just run one of the following commands to run the model in test or train mode:

- To train the model: `python main.py --train`.
- To test the model: `python main.py --test`.
