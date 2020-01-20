## Notes
not everything can/should go in hdf5 file. Should do standard directory structure as well. `raw/` is whatever comes from experiment only.


TODO: x_um_per_pixel and co should be on RAW not imaging


  DATE=2020-01-20
  FISH=f1
  EXPERIMENT=e2
  META=6f
  
  $DATE/$FISH_$EXPERIMENT_$META/
    raw/ (READONLY)
      *.oir
      FishVR.mat
      FishVR.bin
    tail/
      FishVR.mp4
      FishVR_trackingparams.json
      FishVR.hdf5
    temp/
      *.ome.btf
      *.nrrd
    ty.h5
    lab_notebook.yml
    maxZ.mp4
    data_frames/
      *.parquet
    plots/
      *.svg

  2020-01-19/f1_e1/


## Spec
Version: 0.1.0

`/` refers to a group or dataset
`-` refers to an attribute or dtype
  same level as text following `-` is a comment / description

root
  - version
    - str
      e.g. "0.1.0"
  /imaging
    functional imaging of neural activity
    TCZHW
    - frame_rate
    - x_um_per_pixel
    - y_um_per_pixel
    - z_um_per_pixel
    - nZ
    - nFrames
    /frame_start:
      - [float]
      time
    /frame_end:
      - [float]
      time
    /small:
      - [unint]
      - channel_{i}
        label for channel
      256 x 256
      TxCxZxHxW
  /tailcam
    - frame_rate
    /frame_start:
      - [float]
    /frame_end:
      - [float]