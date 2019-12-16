# DAG for scripts:
## nodes
A) `name=f1_e1 && bfconvert "$name.oir" "$name.ome.btf"`

## Dependencies
btf_2_hdf5_small: A
motion_correction: A
ome2nrrd: A


