## Installation

### MacOS

```
CONDA_SUBDIR=osx-64 conda create -n yolo python=3.8 wheel=0.38.4 ffmpeg=4.2.2 portaudio==19.6.0
conda activate yolo
conda config --env --set subdir osx-64
pip install -r requirements.txt
```

### Windows

```
conda create -n yolo python=3.8 wheel=0.38.4 ffmpeg=4.2.2 
conda activate yolo
pip install -r requirements.txt
```

## Usage

```
python yolo_tracker.py
```

## Compile

```
TODO
```