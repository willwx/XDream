from pathlib import Path

# whether gpu is available for NN computation
gpu_available = True
# root directory where network defs and weights are stored; see net_catalogue.py
nets_dir = Path('~/Documents/nets').expanduser()
