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