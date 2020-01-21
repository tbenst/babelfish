import torch, tables, numpy as np
from tqdm import tqdm


def df_f(t, winsize:int, q:float=20, epsilon:int=10,
         style:str="causal"):
    """Pixel-wise ΔF/F with baseline from sliding-window q percentile filter.
    
    First dim assumed to be time. Causal, except for first winsize entries.
    See Yu Mu, Davis V. Bennet (Cell 2019)"""
    
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = int(round(.01 * float(q) * (winsize - 1)))
    print("t", t.shape, "winsize", winsize, "k", k)
    baseline_temp = t.unfold(0,winsize,1).kthvalue(k).values # winsize smaller than t    
    prev_time_pad = t.shape[0] - baseline_temp.shape[0]
    print("prev_time_pad", prev_time_pad)
    print("baseline_temp", baseline_temp.shape)
    baseline = torch.zeros_like(t)
    if style=="causal":
        baseline[prev_time_pad:] = baseline_temp
        baseline[:prev_time_pad] = baseline_temp[0]
    elif style=="acausal":
        s = int(np.floor(prev_time_pad/2))
        baseline[:s] = baseline_temp[0]
        baseline[s:s+baseline_temp.shape[0]] = baseline_temp
        baseline[s+baseline_temp.shape[0]:] = baseline_temp[-1]
    else:
        raise(NotImplementedError())
    print("calculating df/f...")
    return (t - baseline)/(baseline+epsilon)


def tables_df_f(c:tables.carray.CArray, out:tables.carray.CArray, winsize:int,
                q:float=20, epsilon:int=10, style:str="causal", max_ram:float=24*1e3**3) -> None:
    """Pixel-wise ΔF/F with baseline from sliding-window q percentile filter.
    
    First dim assumed to be time. Causal, except for first winsize entries.
    See Yu Mu, Davis V. Bennet (Cell 2019).
    
    This function chunks array so it fits in GPU RAM, & calls _df_f"""
    
    array_size = np.product(c.shape)*4 # always float32
    overhead = 8 # 4 was not enough, 6 prob ok
    min_chunks = int(np.ceil(array_size/(max_ram)))*overhead
    nrows = c.shape[-2]
    print("array_size", array_size, "min_chunks", min_chunks, nrows)
    nrows_per_chunk = int(np.floor(nrows/min_chunks))
    if nrows_per_chunk==0:
        print("try to fit all into memory")
        # can fit in memory
        nrows_per_chunk = nrows
    chunk_shape = list(c.shape)
    chunk_shape[-2] = nrows_per_chunk
    print("chunk_shape", chunk_shape)
    chunk_size = np.product(chunk_shape)*4
    estimated_gpu_ram = overhead*chunk_size
    print(f"will use {estimated_gpu_ram/1024**3:3f} GB of GPU RAM")
    assert estimated_gpu_ram < max_ram
    
    for s in tqdm(range(0,nrows,nrows_per_chunk)):
        chunk = torch.from_numpy(c[...,s:s+nrows_per_chunk,:].astype(np.float32)).cuda()
        print("chunk", chunk.shape)
        # more than 2x faster to convert to numpy before assignment
        out[...,s:s+nrows_per_chunk,:] = df_f(chunk, winsize, q,
                                            epsilon, style).cpu().numpy()