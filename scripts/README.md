## common
```
for f in $(ls *.mat); do ~/code/babelfish/scripts/hdf5_to_video $f "/gROI"; done
```


## shebang woes

Curent solution does not allow `()` but does allow `"`
nix-shell Scripts are a PITA. Here's the approaches that I tried.




(runner-up)
```
args="'python \"$0\" "${@}" '"
eval exec "$(which nix-shell)" "$(dirname ${BASH_SOURCE[0]})/../../shell.nix" --run $nx
```

#!/usr/bin/env bash
"true" '''\'
# eval exec "$(which nix-shell)" "$(dirname ${BASH_SOURCE[0]})/../../shell.nix" --run 'python /home/tyler/code/babelfish/scripts/to_tyh5/tiff_channels_2_small "-l" "PV (GCaMP),PyR (RCaMP)" "-u" "1.13671875" "-r" "7.47" "5690_990nm_Ave4_z353_pw550-000_Ch1.tif" "5690_990nm_Ave4_z353_pw550-000_Ch2.tif"'

args="'python \"$0\""
for a in "${@}"; do
    args+=' '
    args+=\"$a\"
done
args+="'"
echo -e $args
# eval "'python' \"$0\" \"${@}\""
# pp="'python' \"$0\" "${@}" "
# eval $pp
nx="'python \"$0\" "${@}" '"
# eval exec "$(which nix-shell)" "$(dirname ${BASH_SOURCE[0]})/../../shell.nix" --run $nx
eval exec "$(which nix-shell)" "$(dirname ${BASH_SOURCE[0]})/../../shell.nix" --run $nx
exit 0

eval $args
eval exec "$(which nix-shell)" --pure "$(dirname ${BASH_SOURCE[0]})/../../shell.nix" --run $args
exit 1

# :)
eval "$(which nix-shell)" --pure "$(dirname ${BASH_SOURCE[0]})/../../shell.nix" --run '"python $0"'
'eval "$(which nix-shell)" --pure "$(dirname ${BASH_SOURCE[0]})/../../shell.nix" --run "python $0"'
exec "$(which nix-shell)" --pure "$(dirname ${BASH_SOURCE[0]})/../../shell.nix" --run 'python /home/tyler/code/babelfish/scripts/to_tyh5/tiff_channels_2_small'
exec "$(which nix-shell)" --pure "$(dirname ${BASH_SOURCE[0]})/../../shell.nix" --run "python $0"

# :(
exec "$(which nix-shell)" --pure "$(dirname ${BASH_SOURCE[0]})/../../shell.nix" --run "python $0 ${[@]}"
eval "$(which nix-shell)" --pure "$(dirname ${BASH_SOURCE[0]})/../../shell.nix" --run python $0 "${@}"
a="python $0" && 'exec "$(which nix-shell)" --pure "$(dirname ${BASH_SOURCE[0]})/../../shell.nix" --run "\'$a\'"
a="python $0" && 'exec "$(which nix-shell)" --pure "$(dirname ${BASH_SOURCE[0]})/../../shell.nix" --run "$a"
exec "$(which nix-shell)" --pure "$(dirname ${BASH_SOURCE[0]})/../../shell.nix" --run $(echo '"python /home/tyler/code/babelfish/scripts/to_tyh5/tiff_channels_2_small"')
exec "$(which nix-shell)" --pure "$(dirname ${BASH_SOURCE[0]})/../../shell.nix" --run ''"python" $0 "${@}"''
exec "$(which nix-shell)" --pure "$(dirname ${BASH_SOURCE[0]})/../../shell.nix" --run "'python $0'"
exec "$(which nix-shell)" --pure "$(dirname ${BASH_SOURCE[0]})/../../shell.nix" --run \'"python $0"\'


# exec "$(which nix-shell)" "$(dirname ${BASH_SOURCE[0]})/../../shell.nix" --run \'"python $0 \"$1\" \"$2\" \"$3\" "\'



# # works
# exec "$(which nix-shell)" "$(dirname ${BASH_SOURCE[0]})/../../shell.nix" --run "$(echo python ~/code/babelfish/scripts/to_tyh5/tiff_channels_2_small -l \"PV (GCaMP),PyR (RCaMP)\" -u 1.13671875 -r 7.47 *.tif)"
exec "$(which nix-shell)" "$(dirname ${BASH_SOURCE[0]})/../../shell.nix" --run 'python ~/code/babelfish/scripts/to_tyh5/tiff_channels_2_small -l "PV (GCaMP),PyR (RCaMP)" -u 1.13671875 -r 7.47 *.tif'
# # works

# ''"python" $0 "${@}"''

'''