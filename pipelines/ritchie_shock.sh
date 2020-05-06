#!/usr/bin/env bash

set -e
cd /data/ritchie
# for name in f090618_3
# do
#     echo processing $name
#     cd $name && nix-shell --pure ~/code/babelfish/shell.nix --run "tiff_Zs_2_tyh5 -o $name.ty.h5 *.tif"
#     nix-shell --pure ~/code/babelfish/shell.nix --run 'caiman_2_tyh5 -s "." .'
#     nix-shell --pure ~/code/babelfish/shell.nix --run "npz_to_hdf5 -g '/behavior' *segmented_movements.npz *.ty.h5 features tail_seg_params shock_et shock_st tailcam_et tailcam_st imcam_et imcam_st"
#     cd .. && nix-shell --pure ~/code/babelfish/shell.nix --run "process_aaron_nidaq $name -m -l $name.mat -b $name.bin"
#     nix-shell --pure ~/code/babelfish/shell.nix --run "hdf5_to_video $name/$name.ty.h5 /imaging/raw --max-z"
# done

for name in f081318_1 f081318_2 f081318_3 f081418_1 f090418_2 f090418_3 f090418_6 f090618_2 f090618_4
do
    echo $name
    cd $name
    nix-shell --pure ~/code/babelfish/shell.nix --run "npz_to_hdf5 -g '/behavior' *segmented_movements.npz *.ty.h5 features tail_seg_params shock_et shock_st tailcam_et tailcam_st imcam_et imcam_st"
    cd ..
done