#!/usr/bin/env python3
"""
Build script for oMLX macOS app.

This script:
1. Builds venvstacks layers (runtime + framework + app)
2. Creates macOS .app bundle
3. Packages into DMG

Usage:
    python build.py              # Build everything
    python build.py --skip-venv  # Skip venvstacks build (use existing)
    python build.py --dmg-only   # Only create DMG from existing build
"""

import argparse
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

import re

SCRIPT_DIR = Path(__file__).parent
BUILD_DIR = SCRIPT_DIR / "_build"
EXPORT_DIR = SCRIPT_DIR / "_export"
DIST_DIR = SCRIPT_DIR / "dist"
WHEELS_DIR = SCRIPT_DIR / "_wheels"
APP_NAME = "oMLX"
APP_BUNDLE = f"{APP_NAME}.app"
VERSION = "0.1.0"


def clean_all():
    """Remove all build artifacts and caches for a clean build."""
    print("\n[Clean] Removing all build artifacts...")

    dirs_to_clean = [
        BUILD_DIR,      # _build/
        EXPORT_DIR,     # _export/
        WHEELS_DIR,     # _wheels/
        DIST_DIR,       # dist/
        SCRIPT_DIR / "requirements",  # venvstacks lock files
    ]

    files_to_clean = [
        SCRIPT_DIR / "_venvstacks_resolved.toml",
    ]

    for d in dirs_to_clean:
        if d.exists():
            shutil.rmtree(d)
            print(f"  Removed {d.relative_to(SCRIPT_DIR)}/")

    for f in files_to_clean:
        if f.exists():
            f.unlink()
            print(f"  Removed {f.relative_to(SCRIPT_DIR)}")

    print("  ✓ Clean complete\n")


def run_cmd(cmd: list, cwd: Path = None, check: bool = True):
    """Run a command and print output."""
    print(f"  → {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    if check and result.returncode != 0:
        print(f"  ✗ Command failed with code {result.returncode}")
        sys.exit(1)
    return result


def _parse_git_requirements(toml_path: Path) -> list[tuple[str, str]]:
    """Extract git-based requirements from venvstacks.toml.

    Returns list of (full_requirement_string, git_url) tuples.
    e.g. ("mlx-lm @ git+https://...@sha", "git+https://...@sha")
    """
    content = toml_path.read_text()
    # Match lines like: "mlx-lm @ git+https://github.com/...@commit"
    pattern = r'"([^"]*\s*@\s*(git\+https://[^""]*))"'
    return re.findall(pattern, content)


def _wheel_version(whl_path: Path) -> str:
    """Extract version from wheel filename (e.g. mlx_lm-0.30.6-py3-none-any.whl -> 0.30.6)."""
    parts = whl_path.stem.split("-")
    if len(parts) >= 2:
        return parts[1]
    return "0.0.0"


def _wheel_pkg_name(whl_path: Path) -> str:
    """Extract normalized package name from wheel filename."""
    return whl_path.stem.split("-")[0].replace("_", "-").lower()


def build_local_wheels():
    """Pre-build wheels for git-pinned packages.

    venvstacks/uv disables source builds, so git-pinned packages must be
    pre-built as wheels. This function:
    1. Parses git URLs from venvstacks.toml
    2. Builds wheels via pip
    3. Returns a mapping of package_name -> version for toml rewriting
    """
    print("\n[0/4] Building wheels for git packages...")

    toml_path = SCRIPT_DIR / "venvstacks.toml"
    git_reqs = _parse_git_requirements(toml_path)

    if not git_reqs:
        print("  No git requirements found, skipping.")
        return {}

    # Clean and recreate wheels dir for fresh builds
    if WHEELS_DIR.exists():
        shutil.rmtree(WHEELS_DIR)
    WHEELS_DIR.mkdir(parents=True)

    for full_req, git_url in git_reqs:
        pkg_name = full_req.split("@")[0].strip()
        print(f"  Building wheel for {pkg_name} ...")
        run_cmd([
            sys.executable, "-m", "pip", "wheel",
            git_url,
            "--no-deps",
            "-w", str(WHEELS_DIR),
        ])

    # Build version mapping from built wheels
    version_map = {}
    for whl in WHEELS_DIR.glob("*.whl"):
        name = _wheel_pkg_name(whl)
        version = _wheel_version(whl)
        version_map[name] = version
        print(f"    {name} == {version}")

    print(f"  ✓ {len(version_map)} wheel(s) built in {WHEELS_DIR}")
    return version_map


def _find_wheel_for_package(pkg_name: str) -> Path | None:
    """Find the built wheel file for a package name."""
    normalized = pkg_name.lower().replace("-", "_")
    for whl in WHEELS_DIR.glob("*.whl"):
        whl_name = whl.stem.split("-")[0].lower()
        if whl_name == normalized:
            return whl
    return None


def _create_resolved_toml(version_map: dict[str, str]) -> Path:
    """Create a temporary venvstacks.toml with git URLs replaced by local file:// paths.

    Git-built wheels have different hashes than PyPI releases of the same version,
    so we must point directly to the local wheel files to avoid hash mismatches.
    """
    toml_path = SCRIPT_DIR / "venvstacks.toml"
    content = toml_path.read_text()

    for full_req, git_url in _parse_git_requirements(toml_path):
        pkg_name = full_req.split("@")[0].strip()
        whl = _find_wheel_for_package(pkg_name)
        if whl:
            whl_uri = whl.resolve().as_uri()
            old_line = f'"{full_req}"'
            new_line = f'"{pkg_name} @ {whl_uri}"'
            content = content.replace(old_line, new_line)
            print(f"    {pkg_name} @ git+... → {whl.name}")

    resolved_path = SCRIPT_DIR / "_venvstacks_resolved.toml"
    resolved_path.write_text(content)
    return resolved_path


def build_venvstacks():
    """Build venvstacks layers."""
    print("\n[1/4] Building venvstacks layers...")

    # Step 1: Build wheels from git-pinned packages
    version_map = build_local_wheels()

    # Step 2: Create resolved toml (git URLs → version pins)
    if version_map:
        print("\n  Resolving git requirements to version pins...")
        resolved_toml = _create_resolved_toml(version_map)
    else:
        resolved_toml = SCRIPT_DIR / "venvstacks.toml"

    # Local wheels args
    local_wheels_args = []
    if WHEELS_DIR.exists() and any(WHEELS_DIR.glob("*.whl")):
        local_wheels_args = ["--local-wheels", str(WHEELS_DIR)]

    # Step 3: Lock environments (always re-lock to match current wheels)
    print("\n  Locking environments...")
    lock_cmd = [
        "pipx", "run", "venvstacks", "lock",
        str(resolved_toml),
    ] + local_wheels_args
    if version_map:
        # Force re-lock when git packages changed (hashes will differ)
        lock_cmd += ["--reset-lock", "*"]
    else:
        lock_cmd += ["--if-needed"]
    run_cmd(lock_cmd)

    # Step 4: Build environments
    print("\n  Building environments (this may take a while)...")
    run_cmd([
        "pipx", "run", "venvstacks", "build",
        str(resolved_toml),
        "--no-lock",
    ] + local_wheels_args)

    # Step 5: Export to local directory for app bundle
    print("\n  Exporting environments...")
    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)

    run_cmd([
        "pipx", "run", "venvstacks", "local-export",
        str(resolved_toml),
        "--output-dir", str(EXPORT_DIR),
    ])

    # Cleanup temporary toml
    if version_map and resolved_toml.exists():
        resolved_toml.unlink()

    return EXPORT_DIR


def create_app_bundle():
    """Create macOS .app bundle."""
    print("\n[2/4] Creating app bundle...")

    app_dir = DIST_DIR / APP_BUNDLE
    contents_dir = app_dir / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_dir = contents_dir / "Resources"
    frameworks_dir = contents_dir / "Frameworks"

    # Clean and create directories
    if app_dir.exists():
        shutil.rmtree(app_dir)

    macos_dir.mkdir(parents=True)
    resources_dir.mkdir(parents=True)
    frameworks_dir.mkdir(parents=True)

    # Copy venvstacks environments to Frameworks
    print("  Copying Python environment...")
    for layer in ["cpython-3.11", "framework-mlx-framework", "app-omlx-app"]:
        src = EXPORT_DIR / layer
        if src.exists():
            dst = frameworks_dir / layer
            shutil.copytree(src, dst, symlinks=True)
            print(f"    Copied {layer}")

    # Copy venvstacks metadata
    venvstacks_meta = EXPORT_DIR / "__venvstacks__"
    if venvstacks_meta.exists():
        shutil.copytree(venvstacks_meta, frameworks_dir / "__venvstacks__", symlinks=True)

    # Copy omlx_app to Resources
    print("  Copying omlx_app...")
    omlx_app_src = SCRIPT_DIR / "omlx_app"
    omlx_app_dst = resources_dir / "omlx_app"
    shutil.copytree(omlx_app_src, omlx_app_dst, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc"
    ))

    # Copy omlx package to Resources
    print("  Copying omlx package...")
    omlx_src = SCRIPT_DIR.parent / "omlx"
    omlx_dst = resources_dir / "omlx"
    if omlx_src.exists():
        shutil.copytree(omlx_src, omlx_dst, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".git", "tests", "examples"
        ))

    # Copy SVG logo files to Resources for menubar icons
    print("  Copying logo SVGs...")
    admin_static = SCRIPT_DIR.parent / "omlx" / "admin" / "static"
    svg_files = [
        "navbar-logo-dark.svg",
        "navbar-logo-light.svg",
        "menubar-outline.svg",
        "menubar-filled.svg",
    ]
    for svg_name in svg_files:
        svg_src = admin_static / svg_name
        if svg_src.exists():
            shutil.copy2(svg_src, resources_dir / svg_name)
            print(f"    Copied {svg_name}")

    # Copy Python binary into MacOS/ so macOS recognizes it as a bundle executable
    print("  Copying Python runtime into MacOS/...")
    src_python = frameworks_dir / "cpython-3.11" / "bin" / "python3"
    dst_python = macos_dir / "python3"
    shutil.copy2(src_python, dst_python)
    dst_python.chmod(0o755)

    # Python binary references @executable_path/../lib/libpython3.11.dylib
    # Create Contents/lib/ with symlink to the actual dylib in Frameworks
    lib_dir = contents_dir / "lib"
    lib_dir.mkdir(exist_ok=True)
    (lib_dir / "libpython3.11.dylib").symlink_to(
        "../Frameworks/cpython-3.11/lib/libpython3.11.dylib"
    )

    # Create launcher script
    print("  Creating launcher...")
    launcher = macos_dir / APP_NAME
    launcher_content = f'''#!/bin/bash
# oMLX Launcher

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTENTS_DIR="$(dirname "$SCRIPT_DIR")"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
FRAMEWORKS_DIR="$CONTENTS_DIR/Frameworks"

# Set PYTHONHOME so Python can find its stdlib
export PYTHONHOME="$FRAMEWORKS_DIR/cpython-3.11"

# Set PYTHONPATH: Resources (omlx_app, omlx) + all venvstacks layer site-packages
export PYTHONPATH="$RESOURCES_DIR:$FRAMEWORKS_DIR/app-omlx-app/lib/python3.11/site-packages:$FRAMEWORKS_DIR/framework-mlx-framework/lib/python3.11/site-packages"

# Prevent .pyc generation at runtime (pre-compiled during build)
export PYTHONDONTWRITEBYTECODE=1

# Run Python from inside Contents/MacOS/ (required for macOS GUI access)
exec "$SCRIPT_DIR/python3" -m omlx_app
'''
    launcher.write_text(launcher_content)
    launcher.chmod(0o755)

    # Create Info.plist
    print("  Creating Info.plist...")
    info_plist = {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": "com.omlx.app",
        "CFBundleVersion": VERSION,
        "CFBundleShortVersionString": VERSION,
        "CFBundleExecutable": APP_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleSignature": "????",
        "CFBundleIconFile": "AppIcon",
        "LSMinimumSystemVersion": "14.0",
        "LSUIElement": True,
        "NSHighResolutionCapable": True,
        "LSArchitecturePriority": ["arm64"],
        "NSHumanReadableCopyright": f"Copyright 2024 oMLX contributors. Version {VERSION}",
    }

    with open(contents_dir / "Info.plist", "wb") as f:
        plistlib.dump(info_plist, f)

    # Create placeholder icon
    create_placeholder_icon(resources_dir)

    print(f"  ✓ Created {app_dir}")
    return app_dir


def _create_composite_svg(dark_svg: Path) -> str:
    """Create a composite SVG: white rounded-rect background + black logo."""
    svg_content = dark_svg.read_text()
    # Extract the <g> element (contains transform + path)
    g_match = re.search(r"<g[^>]*>.*?</g>", svg_content, re.DOTALL)
    g_element = g_match.group(0) if g_match else ""
    # Change fill from white to black for white background
    g_element = g_element.replace('fill="#ffffff"', 'fill="#000000"')

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">
  <rect x="96" y="96" width="832" height="832" rx="186" ry="186" fill="#ffffff"/>
  <svg x="180" y="180" width="664" height="664" viewBox="0 0 497.000000 497.000000">
    {g_element}
  </svg>
</svg>'''


def create_placeholder_icon(resources_dir: Path):
    """Create app icon from SVG logo (dark logo on white background).

    Rendering priority:
    1. Exported venvstacks Python + AppKit (native SVG rendering)
    2. cairosvg (if installed in build env)
    3. Pillow placeholder (last resort)
    """
    icon_path = resources_dir / "AppIcon.icns"
    dark_svg = SCRIPT_DIR.parent / "omlx" / "admin" / "static" / "navbar-logo-dark.svg"

    if not dark_svg.exists():
        print("    Warning: navbar-logo-dark.svg not found, skipping icon")
        return

    # Create composite SVG (white bg + black penguin)
    composite_svg = _create_composite_svg(dark_svg)
    tmp_svg = resources_dir / "_icon_tmp.svg"
    tmp_png = resources_dir / "_icon_tmp.png"
    tmp_svg.write_text(composite_svg)

    try:
        # Method 1: Use exported runtime Python with AppKit (native macOS SVG rendering)
        runtime_python = EXPORT_DIR / "cpython-3.11" / "bin" / "python3"
        if runtime_python.exists() and _render_svg_with_appkit(runtime_python, tmp_svg, tmp_png):
            _png_to_icns(str(tmp_png), icon_path, resources_dir)
            print("    Created app icon from SVG (AppKit)")
        # Method 2: cairosvg
        elif _render_svg_with_cairosvg(composite_svg, tmp_png):
            _png_to_icns(str(tmp_png), icon_path, resources_dir)
            print("    Created app icon from SVG (cairosvg)")
        else:
            print("    Warning: Could not render SVG, no icon created")
    finally:
        tmp_svg.unlink(missing_ok=True)
        tmp_png.unlink(missing_ok=True)


def _render_svg_with_appkit(python_exe: Path, svg_path: Path, png_path: Path) -> bool:
    """Render SVG to PNG using AppKit's native NSImage (via subprocess).

    Uses the venvstacks runtime Python with PYTHONHOME + layer site-packages
    so that PyObjC (AppKit/Foundation) is available.
    """
    script = f'''
import sys
from Foundation import NSData
from AppKit import NSImage, NSBitmapImageRep, NSPNGFileType, NSMakeRect, NSCompositingOperationSourceOver
from AppKit import NSGraphicsContext, NSImageInterpolationHigh

svg_data = NSData.dataWithContentsOfFile_("{svg_path}")
if svg_data is None:
    sys.exit(1)

image = NSImage.alloc().initWithData_(svg_data)
if image is None:
    sys.exit(1)

size = 1024
out_image = NSImage.alloc().initWithSize_((size, size))
out_image.lockFocus()
ctx = NSGraphicsContext.currentContext()
ctx.setImageInterpolation_(NSImageInterpolationHigh)
image.drawInRect_fromRect_operation_fraction_(
    NSMakeRect(0, 0, size, size),
    NSMakeRect(0, 0, image.size().width, image.size().height),
    NSCompositingOperationSourceOver,
    1.0,
)
out_image.unlockFocus()

rep = NSBitmapImageRep.alloc().initWithData_(out_image.TIFFRepresentation())
png_data = rep.representationUsingType_properties_(NSPNGFileType, {{}})
png_data.writeToFile_atomically_("{png_path}", True)
'''
    runtime_dir = python_exe.parent.parent
    app_sp = EXPORT_DIR / "app-omlx-app" / "lib" / "python3.11" / "site-packages"
    fw_sp = EXPORT_DIR / "framework-mlx-framework" / "lib" / "python3.11" / "site-packages"

    env = os.environ.copy()
    env["PYTHONHOME"] = str(runtime_dir)
    env["PYTHONPATH"] = f"{app_sp}:{fw_sp}"

    try:
        result = subprocess.run(
            [str(python_exe), "-c", script],
            capture_output=True, timeout=30, env=env,
        )
        if result.returncode != 0:
            print(f"    AppKit stderr: {result.stderr.decode()[:200]}")
        return result.returncode == 0 and png_path.exists()
    except Exception as e:
        print(f"    AppKit rendering failed: {e}")
        return False


def _render_svg_with_cairosvg(svg_content: str, png_path: Path) -> bool:
    """Render SVG to PNG using cairosvg."""
    try:
        import cairosvg
        cairosvg.svg2png(
            bytestring=svg_content.encode(),
            write_to=str(png_path),
            output_width=1024, output_height=1024,
        )
        return png_path.exists()
    except ImportError:
        return False
    except Exception as e:
        print(f"    cairosvg rendering failed: {e}")
        return False


def _png_to_icns(png_path: str, icon_path: Path, resources_dir: Path):
    """Convert a 1024x1024 PNG to .icns via iconset using sips (macOS built-in)."""
    iconset_dir = resources_dir / "AppIcon.iconset"
    iconset_dir.mkdir(exist_ok=True)

    sizes = [
        (16, "icon_16x16.png"),
        (32, "icon_16x16@2x.png"),
        (32, "icon_32x32.png"),
        (64, "icon_32x32@2x.png"),
        (128, "icon_128x128.png"),
        (256, "icon_128x128@2x.png"),
        (256, "icon_256x256.png"),
        (512, "icon_256x256@2x.png"),
        (512, "icon_512x512.png"),
        (1024, "icon_512x512@2x.png"),
    ]

    for s, name in sizes:
        out = iconset_dir / name
        shutil.copy2(png_path, str(out))
        subprocess.run(
            ["sips", "-z", str(s), str(s), str(out)],
            capture_output=True,
        )

    subprocess.run(
        ["iconutil", "-c", "icns", str(iconset_dir), "-o", str(icon_path)],
        capture_output=True,
    )

    shutil.rmtree(iconset_dir)


def sign_app(app_dir: Path):
    """Ad-hoc sign the app bundle."""
    print("\n[3/4] Signing app bundle...")

    run_cmd([
        "codesign", "--force", "--deep", "--sign", "-",
        str(app_dir)
    ], check=False)

    print(f"  ✓ Signed {app_dir}")


def create_dmg(app_dir: Path):
    """Create DMG installer with Applications symlink for drag-and-drop."""
    print("\n[4/4] Creating DMG...")

    dmg_path = DIST_DIR / f"{APP_NAME}-{VERSION}.dmg"
    dmg_staging = DIST_DIR / "_dmg_staging"

    # Remove existing
    if dmg_path.exists():
        dmg_path.unlink()
    if dmg_staging.exists():
        shutil.rmtree(dmg_staging)

    # Create staging directory
    dmg_staging.mkdir(parents=True)

    # Copy app bundle to staging
    shutil.copytree(app_dir, dmg_staging / APP_BUNDLE, symlinks=True)

    # Create Applications symlink
    applications_link = dmg_staging / "Applications"
    applications_link.symlink_to("/Applications")

    print("  Creating DMG with Applications shortcut...")
    run_cmd([
        "hdiutil", "create",
        "-volname", APP_NAME,
        "-srcfolder", str(dmg_staging),
        "-ov", "-format", "UDZO",
        str(dmg_path)
    ])

    # Cleanup staging
    shutil.rmtree(dmg_staging)

    print(f"  ✓ Created {dmg_path}")
    return dmg_path


def main():
    parser = argparse.ArgumentParser(description="Build oMLX macOS app")
    parser.add_argument("--skip-venv", action="store_true",
                        help="Skip venvstacks build")
    parser.add_argument("--dmg-only", action="store_true",
                        help="Only create DMG from existing build")
    args = parser.parse_args()

    print(f"Building {APP_NAME} v{VERSION}")
    print("=" * 50)

    # Clean all build artifacts before starting (unless dmg-only)
    if not args.dmg_only:
        clean_all()

    DIST_DIR.mkdir(parents=True, exist_ok=True)

    if args.dmg_only:
        app_dir = DIST_DIR / APP_BUNDLE
        if not app_dir.exists():
            print(f"Error: {app_dir} not found. Run full build first.")
            sys.exit(1)
        create_dmg(app_dir)
    else:
        if not args.skip_venv:
            build_venvstacks()
        elif not EXPORT_DIR.exists():
            print("Warning: No existing envs found, building venvstacks...")
            build_venvstacks()

        app_dir = create_app_bundle()
        sign_app(app_dir)
        create_dmg(app_dir)

    print("\n" + "=" * 50)
    print("Build complete!")
    print(f"  App: {DIST_DIR / APP_BUNDLE}")
    print(f"  DMG: {DIST_DIR / f'{APP_NAME}-{VERSION}.dmg'}")


if __name__ == "__main__":
    main()
