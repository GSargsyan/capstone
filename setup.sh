dnf install gcc-c++
dnf install python3-devel
pip3 install -r requirements.txt

mkdir -p var/images
mkdir var/landmarks
mkdir var/data
touch data/averages.json

echo "[]" > averages.json
