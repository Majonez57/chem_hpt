# Source - https://stackoverflow.com/a/41586571
# Posted by Martin Thoma, modified by community. See post 'Timeline' for change history
# Retrieved 2026-08-09, License - CC BY-SA 4.0

import h5py
import cv2
import numpy as np
filename = "/home/majonez57/Documents/chem_hpt/chemdata_raw/aug7/opaque/episode_0000.h5"

with h5py.File(filename, "r") as f:
    # Print all root level object names (aka keys) 
    # these can be group or dataset names 
    print("Keys: %s" % f.keys())
    # get first object name/key; may or may NOT be a group
    a_group_key = list(f.keys())[0]

    # get the object type for a_group_key: usually group or dataset
    print(type(f[a_group_key])) 

    # If a_group_key is a group name, 
    # this gets the object names in the group and returns as a list
    data = list(f[a_group_key]['zed__zed_node__rgb__color__rect__image'])
    data = np.asarray(data)
    print(data.shape)
    ZED_CROP_RANGE_W = (280, -93)
    #ZED_CROP_RANGE_H = ()

    data2 = data[:, 0:, ZED_CROP_RANGE_W[0]:ZED_CROP_RANGE_W[1]]
    print(data2.shape)
    for image in data:
        testim = image[:, ZED_CROP_RANGE_W[0]:ZED_CROP_RANGE_W[1]]
        
        cv2.imshow('guh', testim)
        cv2.waitKey(30)
    # If a_group_key is a dataset name, 
    # this gets the dataset values and returns as a list
    data = list(f[a_group_key])
    # preferred methods to get dataset values:
    #ds_obj = f[a_group_key]      # returns as a h5py dataset object
    #ds_arr = f[a_group_key][()]  # returns as a numpy array
