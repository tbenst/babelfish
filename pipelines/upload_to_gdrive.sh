set -e
cd /data/ritchie
for name in f080218_1 f081318_1 f081318_2 f081318_3 f081418_1 f090418_2 f090418_3 f090418_6 f090618_2 f090618_4
do
    rclone copy $name/*.ty.h5 sd:shock_data_for_kanaka
    rclone copy $name/$name_raw.mp4 sd:shock_data_for_kanaka
done
cd -