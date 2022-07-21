''' Code to transform helical scan data to normal scan data readable by tomopy-cli
Original version written by Viktor Nikitin at 2-BM.

'''
import numpy as np
import sys
import h5py
import cupy as  cp # subpixel shifts on gpu


def copy_attributes(in_object, out_object):
    '''Copy attributes between 2 HDF5 objects.'''
    for key, value in in_object.attrs.items():
        out_object.attrs[key] = value

def _report(operation, key, obj):
    type_str = type(obj).__name__.split(".")[-1].lower()
    print(f"{operation}  {type_str} : {key}")

def h5py_compatible_attributes(in_object):
    '''Are all attributes of an object readable in h5py?'''
    try:
        # Force obtaining the attributes so that error may appear
        [ 0 for at in in_object.attrs.items() ]
        return True
    except:
        return False

def copy_h5(in_object, out_object, filter_data=[None], log=False):    
    '''Recursively copy&compress the tree.
    
    If attributes cannot be transferred, a copy is created.
    Otherwise, dataset are compressed.
    '''
    for key, in_obj in in_object.items():
        if (key in filter_data):
            continue
        if not isinstance(in_obj, h5py.Datatype) and h5py_compatible_attributes(in_obj):
            if isinstance(in_obj, h5py.Group):
                out_obj = out_object.create_group(key)
                copy_h5(in_obj, out_obj, filter_data, log)
                if log:
                    _report("Copied", key, in_obj)
            elif isinstance(in_obj, h5py.Dataset):
                out_obj = out_object.create_dataset(key, data=in_obj)
                if log:
                    _report("Copied", key, in_obj)
            else:
                raise "Invalid object type %s" % type(in_obj)
            copy_attributes(in_obj, out_obj)
        else:
            # We copy datatypes and objects with non-understandable attributes
            # identically.
            if log:
                _report("Copied", key, in_obj)
            in_object.copy(key, out_object)        
