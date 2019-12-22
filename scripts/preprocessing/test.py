import sys
sys.path.append("/nix/store/z83h233zyisd6sc30na5kpgd5svk0qbh-python3-3.7.5-env/lib/python3.7/site-packages")
import tifffile
tiff_fn  = "/data/dlab/zfish_2p/20191101_6f/f2_e1_omr.ome.btf"

mm = tifffile.TiffFile(tiff_fn).asarray(out='memmap', maxworkers=1, suindices=slice(100))
print(type(mm))
