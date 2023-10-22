if["$(uname)"=="Darwin"];then
    brew install openblas
elif["$(expr substr $(uname -s) 1 5)"=="Linux"];then
    sudo apt install libopenblas-dev
elif["$(expr substr $(uname -s) 1 10)"=="MINGW32_NT"];then    
# Windows NT操作系统
fi

mkdir log
git submodule update --init --recursive
python3.10 scripts/apply_patchs.py>log/patch_info.log 2>log/patch_error.log
python3.10 scripts/build_third_libs.py>log/3rd_info.log 2>log/3rd_error.log

# mkdir dataset && cd dataset
# curl https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_extract.zip --output 2011_09_26_drive_0001_extract.zip
# curl https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip --output 2011_09_26_calib.zip
# curl https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_tracklets.zip --output 2011_09_26_drive_0001_tracklets.zip
# unzip -d tracklets 2011_09_26_drive_0001_tracklets.zip
# unzip -d extract 2011_09_26_drive_0001_extract.zip
# unzip -d calib 2011_09_26_calib.zip

# curl --retry 5 -C - https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip --output data_odometry_gray.zip