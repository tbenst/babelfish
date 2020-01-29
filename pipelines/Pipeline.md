# DAG for scripts:
## Example order (dependencies indented):
script_prefix=~/code/babelfish/scripts/to_tyh5/

`name=f1_e1 && bfconvert "raw/$name.oir" "$name.ome.btf"`
  tiffs_2_tyh5 -u 1 -c 1 f1_e1.ome.btf
    hdf5_to_video (optional viz of maxZ)
    (3) ~/code/babelfish/scripts/preprocessing/resize_datase -e f2e1 -d "/imaging/raw" -o "/imaging/small"
hdf5_to_video (tailcam .mat)
  (1) `python -m stytra.offline.track_video`
(2) process_aaron_nidaq
(1 & 2)
  tidy_tail_data
(2 & 3)
  dset_df_f f3e1/ -d "/imaging/small"


## nodes
letter or script name
A) `name=f1_e1 && bfconvert "$name.oir" "$name.ome.btf"`
B) `python -m stytra.offline.track_video`

## Dependencies
motion_correction: A
mc_2_tyh5: motion_correction (not implemented)
tiffs_2_tyh5: A
ome2nrrd: A
hdf5_to_video: tiffs_2_tyh5 | mc_2_tyh5 | None (.mat)
train: tiffs_2_tyh5 | mc_2_tyh5
functional_connectivity_image_map: train
count_vid_frames: hdf5_to_video
B: hdf5_to_video (.mat)
tail_metrics: B & process_aaron_nidaq
tidy_data: process_aaron_nidaq & B


Neural pipeline (ideal...)
motion_correction
ome2nrrd





# bash
```
# convert matlab gROI to .mp4
for f in $(ls *.mat); do ~/code/babelfish/scripts/hdf5_to_video $f "/gROI"; done

# find .mp4 files for offline tail tracking
for f in $(fd ".*FishVR\.mp4"); do if ! [ -f "${f%????}.hdf5" ]; then echo $f; fi; done

# find missing .ome.btf (intermediate format, can delete after making .ty.h5)
for f in $(fd ".*\.oir"); do if ! [ -f "${f%????}.ome.btf" ]; then echo $f; fi; done
for f in $(ls *.oir);  if ! [ -f "${f%????}.ome.btf" ]; then do bfconvert "$f ${f%????}.ome.btf"; done

for f in $(ls *.oir);  if ! [ -f "${f%????}.ome.btf" ]; then do echo bfconvert "$f ${f%????}.ome.btf"; done


# create missing tyh5
for f in $(fd "f\d_*e\d.*\.ome.btf$"); do if ! [ -f "${f%.ome.btf}.ty.h5" ]; then ~/code/babelfish/scripts/to_tyh5/tiffs_2_tyh5 -u 1 $f; fi; done

# find missing stytra hdf5
for f in $(fd ".*FishVR\.mp4"); do if ! [ -f "${f%????}.hdf5" ]; then echo $f; fi; done
```

# Airflow
## setup

```
export AIRFLOW_HOME="~/code/lensman-airflow"
airflow initdb
```

## start
```
airflow webserver -p 8080
airflow scheduler
```

# notes
- 20190429/olympus/f5e1.ty.h5
  - weird shape (11, 999, 225, 512) - should delete
- 20190429/olympus/f5e2.ty.h5
  - weird shape (14, 999, 225, 512) - should delete
need to transpose T & Z...?

working on 20191101, 20191031, 20191017