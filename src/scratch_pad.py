import os as var_os
import pickle as var_pickle
import numpy as var_numpy
from yaml import dump

# Path Stuff
cwd_path = var_os.path.abspath(var_os.path.dirname(__file__))
repo_root_path = var_os.path.abspath(var_os.path.dirname(var_os.path.dirname(__file__)))
print ("Paths: cwd = {}, root = {}".format(cwd_path, repo_root_path))

dump_file_path = repo_root_path+"/.outputs/.models/scratch-pad-pickle-dump.bindata"
print ("Dump File Path = {}".format(dump_file_path))

# Pickle Stuff
dump_obj = var_numpy.array([1,3,5,7,9,2,4,6,8,0])
with open(dump_file_path, 'wb') as dump_file:
    var_pickle.dump(dump_obj, dump_file)
with open(dump_file_path, "rb") as load_file:
    load_obj = var_pickle.load(load_file)
print("Objects: Dump = {}, Load = {}".format(dump_obj, load_obj))
