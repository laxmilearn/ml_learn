import os as pkg_os
import pickle as pkg_pickle
import numpy as pkg_numpy
import pandas as pkg_pandas
from yaml import dump

# Panda Stuff
students_df = pkg_pandas.DataFrame(columns=['name', 'age', 'marks', 'result', 'rank'])
for name in students_df.columns:
    print ("Column Index ({}) = {}".format(name, students_df.columns.get_loc(name)))

# Path Stuff
cwd_path = pkg_os.path.abspath(pkg_os.path.dirname(__file__))
repo_root_path = pkg_os.path.abspath(pkg_os.path.dirname(pkg_os.path.dirname(__file__)))
print ("Paths: cwd = {}, root = {}".format(cwd_path, repo_root_path))

dump_file_path = repo_root_path+"/.outputs/.models/scratch-pad-pickle-dump.bindata"
print ("Dump File Path = {}".format(dump_file_path))

# Pickle Stuff
dump_obj = pkg_numpy.array([1,3,5,7,9,2,4,6,8,0])
with open(dump_file_path, 'wb') as dump_file:
    pkg_pickle.dump(dump_obj, dump_file)
with open(dump_file_path, "rb") as load_file:
    load_obj = pkg_pickle.load(load_file)
print("Objects: Dump = {}, Load = {}".format(dump_obj, load_obj))

