from __future__ import absolute_import

import os 

__all__ = ['data_dir', 'default_file']

data_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'data')

default_file = os.path.join(data_dir, 'efps_d_le_9.npz')