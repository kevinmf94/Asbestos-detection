apt update

# System utils
apt install unzip -y
apt install vim -y
apt install screen -y
apt install git -y
apt install gcc g++ -y
apt install build-essential autoconf libtool pkg-config python-opengl python-pil python-pyrex python-pyside.qtopengl idle-python2.7 qt4-dev-tools qt4-designer libqtgui4 libqtcore4 libqt4-xml libqt4-test libqt4-script libqt4-network libqt4-dbus python-qt4 python-qt4-gl libgle3 python-dev libssl-dev -y

# Install ffmpeg for OpenCV
apt install ffmpeg libsm6 libxext6 -y

# Install detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install python requirements
pip install -r requirements.txt