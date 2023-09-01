# Video Tracking to Audio

This python based software uses ultralytic's YoloV8 (You Only Look Once) state-of-the-art video recognition deep learning model to identify, count and track the number of people on a webcam. It asignes one granular synthetizer to each person and, depending on its X-Y position, the number of people, and the closeness between them, some parameters of the synthetizers - such as filter, pitch shift, panning and reverb - are modified in real-time.

This project was built for the Chilean 2023 exhibition 'Aparatos de Observación Pluvial', founded by the Chilean Ministry of Culture, Art and Patrimony, FONDART 2022. Folio: 670933.

## Installation

### MacOS

```
CONDA_SUBDIR=osx-64 conda create -n yolo python=3.8 wheel=0.38.4 ffmpeg=4.2.2 portaudio=19.6.0 pyinstaller=5.6.2
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

### MacOS

```
pyinstaller yolo_tracker.spec --noconfirm

pyinstaller yolo_tracker.py --onefile --noconfirm --windowed --hidden-import=cv2.cv2 --paths=classes --add-data="audios/bounce.wav:audios" --clean --additional-hooks-dir=~/miniconda3/envs/yolov8/lib/python3.8/site-packages/torch/lib/libtorch_python.dylib --additional-hooks-dir=~/miniconda3/envs/yolov8/lib/python3.8/site-packages/torch/lib/libtorch.dylib --additional-hooks-dir=~/miniconda3/envs/yolov8/lib/python3.8/site-packages/torch/lib/libtorch_global_deps.dylib

5344
```

### Windows

```
TODO
```
