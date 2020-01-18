__import__('tables')

def compression_opts(complevel=7, complib='blosc:lz4', shuffle=True): 
    """Dirty hack for supporting zstd with h5py.
    
    https://github.com/h5py/h5py/issues/611#issuecomment-353694301
    """
    shuffle = 2 if shuffle == 'bit' else 1 if shuffle else 0 
    compressors = ['blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib', 'zstd'] 
    complib = ['blosc:' + c for c in compressors].index(complib) 
    args = { 
        'compression': 32001, 
        'compression_opts': (0, 0, 0, 0, complevel, shuffle, complib) 
    } 
    if shuffle: 
        args['shuffle'] = False 
    return args