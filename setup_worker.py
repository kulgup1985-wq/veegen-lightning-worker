"""Install and patch Wav2Lip for the standalone Lightning GPU worker."""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WAV2LIP_DIR = os.path.join(BASE_DIR, "wav2lip")
CHECKPOINTS_DIR = os.path.join(WAV2LIP_DIR, "checkpoints")

WAV2LIP_REPO = "https://github.com/Rudrabha/Wav2Lip.git"
WAV2LIP_GAN_URLS = [
    "https://huggingface.co/spaces/wav2lip/wav2lip/resolve/main/checkpoints/wav2lip_gan.pth",
    "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth",
    "https://huggingface.co/rippertnt/wav2lip/resolve/main/checkpoints/wav2lip_gan.pth",
]
S3FD_URL = "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"


def step(message: str) -> None:
    print("\n" + "-" * 50)
    print(message)
    print("-" * 50)


def clone_wav2lip() -> None:
    step("Cloning Wav2Lip")
    if os.path.isdir(WAV2LIP_DIR):
        print(f"Already exists: {WAV2LIP_DIR}")
        return
    subprocess.check_call(["git", "clone", WAV2LIP_REPO, WAV2LIP_DIR])


def download_checkpoint() -> None:
    step("Downloading Wav2Lip GAN checkpoint")
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    dest = os.path.join(CHECKPOINTS_DIR, "wav2lip_gan.pth")
    if os.path.isfile(dest) and os.path.getsize(dest) > 100_000_000:
        print(f"Already exists: {dest}")
        return

    errors: list[str] = []
    for url in WAV2LIP_GAN_URLS:
        try:
            urllib.request.urlretrieve(url, dest)
            break
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            if os.path.isfile(dest):
                os.remove(dest)
    else:
        raise RuntimeError("Could not download checkpoint:\n" + "\n".join(errors))


def download_face_detection() -> None:
    step("Downloading face detection model")
    det_dir = os.path.join(WAV2LIP_DIR, "face_detection", "detection", "sfd")
    os.makedirs(det_dir, exist_ok=True)
    dest = os.path.join(det_dir, "s3fd.pth")
    if os.path.isfile(dest) and os.path.getsize(dest) > 10_000_000:
        print(f"Already exists: {dest}")
        return
    urllib.request.urlretrieve(S3FD_URL, dest)


def patch_wav2lip_compat() -> None:
    step("Patching Wav2Lip compatibility")
    audio_path = os.path.join(WAV2LIP_DIR, "audio.py")
    if not os.path.isfile(audio_path):
        raise RuntimeError("Wav2Lip audio.py not found")

    with open(audio_path, "r", encoding="utf-8") as f:
        text = f.read()

    old = "librosa.filters.mel(hp.sample_rate, hp.n_fft,"
    new = "librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft,"
    if new in text:
        print("Patch already applied")
        return
    if old not in text:
        raise RuntimeError("Could not find Wav2Lip librosa mel call to patch")

    with open(audio_path, "w", encoding="utf-8") as f:
        f.write(text.replace(old, new))


def verify_setup() -> None:
    step("Verifying setup")
    checks = [
        ("Wav2Lip repo", os.path.isfile(os.path.join(WAV2LIP_DIR, "inference.py"))),
        ("GAN checkpoint", os.path.isfile(os.path.join(CHECKPOINTS_DIR, "wav2lip_gan.pth"))),
        (
            "s3fd model",
            os.path.isfile(
                os.path.join(WAV2LIP_DIR, "face_detection", "detection", "sfd", "s3fd.pth")
            ),
        ),
    ]
    for name, ok in checks:
        print(f"{'OK' if ok else 'MISSING'}: {name}")
        if not ok:
            raise RuntimeError(f"Missing {name}")

    try:
        import torch

        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
        print(f"PyTorch {torch.__version__}: {device}")
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed") from exc


def main() -> None:
    clone_wav2lip()
    download_checkpoint()
    download_face_detection()
    patch_wav2lip_compat()
    verify_setup()
    print("\nWorker setup complete.")


if __name__ == "__main__":
    main()

