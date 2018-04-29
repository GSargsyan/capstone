dnf install gcc-c++
dnf install python3-devel
pip3 install -r requirements.txt

mkdir -p var/images
mkdir var/landmarks
mkdir var/data
touch var/data/ratios.json
touch var/data/ratios_avg.json
touch var/data/ratios_stdevs.json
