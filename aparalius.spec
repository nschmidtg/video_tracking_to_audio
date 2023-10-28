# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['aparalius.py'],
    pathex=[],
    binaries=[],
    datas=[('ultralytics','.')], # ('audios/Final/A1.wav', 'audios/Final'),('audios/Final/A2.wav', 'audios/Final'),('audios/Final/A3.wav', 'audios/Final'),('audios/Final/G1.wav', 'audios/Final'),('audios/Final/G2.wav', 'audios/Final'),('audios/Final/G3.wav', 'audios/Final'),('audios/Final/E1.wav', 'audios/Final'),('audios/Final/E2.wav', 'audios/Final'),('audios/Final/E3.wav', 'audios/Final'),('audios/Final/C1.wav', 'audios/Final'),('audios/Final/C2.wav', 'audios/Final'),('audios/Final/C3.wav', 'audios/Final'), ('audios/Final/base1.wav', 'audios/Final'),  ('audios/Final/base2.wav', 'audios/Final')],
    hiddenimports=['cv2.cv2'],
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
    name='aparalius',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['favicon-viaje-a-la-lluvia-01.ico']
)
