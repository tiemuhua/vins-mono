sudo apt install libopenblas-dev
python3.10 scripts/build_third_libs.py>info.log 2>error.log
python3.10 scripts/apply_patchs.py>info.log 2>error.log

mkdir dataset && cd dataset
curl https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_extract.zip --output 2011_09_26_drive_0001_extract.zip
curl https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip --output 2011_09_26_calib.zip
curl https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_tracklets.zip --output 2011_09_26_drive_0001_tracklets.zip
unzip -d tracklets 2011_09_26_drive_0001_tracklets.zip
unzip -d extract 2011_09_26_drive_0001_extract.zip
unzip -d calib 2011_09_26_calib.zip

curl --retry 5 -C - https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip --output data_odometry_gray.zip