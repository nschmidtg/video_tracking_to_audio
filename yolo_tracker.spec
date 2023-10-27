# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['yolo_tracker.py'],
    pathex=['classes'],
    binaries=[],
    datas=[('ultralytics','.'), ('model/yolov8n.pt','./model'), ("audios/Final/A1.wav","./audios/Final"),("audios/Final/A2.wav","./audios/Final"),("audios/Final/A3.wav","./audios/Final"),("audios/Final/C1.wav","./audios/Final"),("audios/Final/C2.wav","./audios/Final"),("audios/Final/C3.wav","./audios/Final"),("audios/Final/E1.wav","./audios/Final"),("audios/Final/E2.wav","./audios/Final"),("audios/Final/E3.wav","./audios/Final"),("audios/Final/G1.wav","./audios/Final"),("audios/Final/G2.wav","./audios/Final"),("audios/Final/G3.wav","./audios/Final"),("audios/Final/base1.wav","./audios/Final"),("audios/Final/base2.wav","./audios/Final"),("audios/IRs/301-LargeHall.wav","./audios/IRs")],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='yolo_tracker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['favicon-viaje-a-la-lluvia-01.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='yolo_tracker',
)
