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