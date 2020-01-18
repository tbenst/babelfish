## shebang

WINNER:
```
#!/usr/bin/env bash
"true" '''\'
exec nix-shell "$(dirname ${BASH_SOURCE[0]})/../shell.nix" --run "$(printf '%q ' python "$0" "$@")"
'''
```

## TODO
- package python-bioformats?
- use tables for zstd
```
    with tables.open_file(output_path, 'w') as tyh5: 
       h5.create_carray(h5.root, "test", tables.Int64Atom(), shape=(10,10), 
           filters=tables.filters.Filters(complib='blosc:zstd'))
```


## temp
for f in 20191017_6f/f1e1_*.oir 20191017_6f/f1e2_*.oir 20191017_6s/f3e1_*.oir 20191031_6f/f2_e1_*.oir 20191031_6f/f3_e1_*.oir 20191101_6f/f2_e1_*.oir 20191101_h2b/f1_e1_*.oir; do echo $(fd "$f"); done

## Documentation
need to `import tables` in order to use zstd with h5py


~/code/babelfish/scripts/to_tyh5/tiffs_2_tyh5 -u 1 -c 3 20191101_h2b/f1_e1_6s_omr.ome.btf
5:12 for first zplane!