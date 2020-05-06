# read zstd encoded parquet
# https://stackoverflow.com/questions/58595166/how-to-compress-parquet-file-with-zstandard-using-pandas
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

parquetFilename = "gStartShockInfo.parquet"
df = pq.read_table(parquetFilename)
df = df.to_pandas()