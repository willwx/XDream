import os

# whether caffe has gpu available; see net_loader.py
gpu_available = True
# root directory for storing network defs and weights; see net_catalogue.py
nets_dir = os.path.join(os.path.expanduser('~'), 'Documents/nets')
