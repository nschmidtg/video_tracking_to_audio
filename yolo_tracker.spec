# -*- mode: python ; coding: utf-8 -*-
from os import path

block_cipher = None


a = Analysis(
    ['yolo_tracker.py'],
    pathex=['classes'],
    binaries=[],
    datas=[('model/yolov8n.pt', 'model'), ('audios/bounce.wav', 'audios'),('/Users/nicolas.schmidt/miniconda3/envs/yolov8/lib/python3.8/site-packages/torch', 'torch')],
    hiddenimports=['cv2.cv2'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='yolo_tracker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
app = BUNDLE(
    exe,
    name='yolo_tracker.app',
    icon=None,
    bundle_identifier=None,
)
