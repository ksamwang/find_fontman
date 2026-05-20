# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs


root = Path.cwd()
datas = [
    (str(root / "python_service"), "python_service"),
]

binaries = []
hiddenimports = []
for package in ("cv2", "torch"):
    binaries += collect_dynamic_libs(package)

hiddenimports += [
    "PIL.Image",
    "PIL.ImageDraw",
    "PIL.ImageFilter",
    "PIL.ImageFont",
    "PIL.ImageOps",
    "cv2",
    "numpy",
    "torch",
    "torch.nn",
    "torch.nn.functional",
]


a = Analysis(
    [str(root / "python_service" / "runtime_main.py")],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "paddle",
        "paddleocr",
        "torch.distributed",
        "torch.testing",
        "torch.utils.tensorboard",
        "torchvision",
        "tensorflow",
        "pandas",
        "matplotlib",
        "scipy",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="fontman-service",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="fontman-runtime",
)
