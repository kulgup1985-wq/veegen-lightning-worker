"""Minimal GPU lip-sync helpers for the Lightning worker."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

for ffmpeg_bin in glob.glob(os.path.join(BASE_DIR, "ffmpeg", "*", "bin")):
    if os.path.isdir(ffmpeg_bin) and ffmpeg_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")

WAV2LIP_DIR = os.path.join(BASE_DIR, "wav2lip")
CHECKPOINT_PATH = os.path.join(WAV2LIP_DIR, "checkpoints", "wav2lip_gan.pth")
GREEN_RGB = (0, 177, 64)


def run_command(cmd: list[str], label: str = "cmd", cwd: str | None = None) -> subprocess.CompletedProcess:
    timeout = int(os.environ.get("VEEGEN_CMD_TIMEOUT", "900"))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"[LipSync] {label} timed out after {timeout}s:\n{stderr[-3000:]}")
    if result.stdout:
        print(f"[{label}] stdout: {result.stdout[:500]}")
    if result.stderr:
        print(f"[{label}] stderr: {result.stderr[:500]}")
    if result.returncode != 0:
        raise RuntimeError(f"[LipSync] {label} failed:\n{result.stderr[-3000:]}")
    return result


def is_wav2lip_ready() -> bool:
    return (
        os.path.isdir(WAV2LIP_DIR)
        and os.path.isfile(CHECKPOINT_PATH)
        and os.path.isfile(os.path.join(WAV2LIP_DIR, "inference.py"))
    )


def prepare_face_green_screen(
    face_path: str,
    output_path: str,
    canvas_w: int = 512,
    canvas_h: int = 512,
    upper_body_focus: bool = True,
) -> str:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    from rembg import remove

    print("[LipSync] Removing background from face image")
    img = ImageOps.exif_transpose(Image.open(face_path)).convert("RGBA")

    try:
        cutout = remove(
            img,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=8,
            post_process_mask=True,
        )
    except Exception as exc:
        print(f"[LipSync] Alpha matting failed; falling back to standard cutout: {exc}")
        cutout = remove(img, post_process_mask=True)

    cutout = cutout.convert("RGBA")
    alpha = cutout.getchannel("A")
    alpha = alpha.filter(ImageFilter.MedianFilter(size=3))
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=0.45))
    alpha = ImageEnhance.Contrast(alpha).enhance(1.25)
    alpha = alpha.point(lambda p: 0 if p < 6 else (255 if p > 248 else p))
    cutout.putalpha(alpha)

    bbox = alpha.getbbox()
    if bbox:
        pad = 16
        left = max(0, bbox[0] - pad)
        top = max(0, bbox[1] - pad)
        right = min(cutout.width, bbox[2] + pad)
        bottom = min(cutout.height, bbox[3] + pad)
        cutout = cutout.crop((left, top, right, bottom))

    if upper_body_focus and cutout.height > cutout.width * 1.15:
        cutout = cutout.crop((0, 0, cutout.width, max(1, int(cutout.height * 0.68))))

    cutout.thumbnail((canvas_w, canvas_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (*GREEN_RGB, 255))
    x = (canvas_w - cutout.width) // 2
    y = (canvas_h - cutout.height) // 2
    canvas.paste(cutout, (x, y), cutout)

    canvas.convert("RGB").save(output_path)
    return output_path


def run_wav2lip(face_path: str, audio_path: str, output_path: str, resize_factor: int = 2) -> str:
    if not is_wav2lip_ready():
        raise RuntimeError("Wav2Lip is not set up. Run: python setup_worker.py")
    if not os.path.isfile(face_path):
        raise FileNotFoundError(f"Face image not found: {face_path}")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required on PATH.")

    os.makedirs(os.path.join(WAV2LIP_DIR, "temp"), exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join(WAV2LIP_DIR, "inference.py"),
        "--checkpoint_path",
        CHECKPOINT_PATH,
        "--face",
        face_path,
        "--audio",
        audio_path,
        "--outfile",
        output_path,
        "--resize_factor",
        str(resize_factor),
        "--nosmooth",
        "--pads",
        "0",
        "15",
        "0",
        "0",
    ]
    run_command(cmd, "Wav2Lip", cwd=WAV2LIP_DIR)

    if not os.path.isfile(output_path):
        raise RuntimeError("Wav2Lip produced no output file.")
    return output_path

