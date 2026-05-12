"""Standalone LitServe app for VeeGen GPU lip-sync inference."""

from __future__ import annotations

import base64
import os
import tempfile
import uuid

import litserve as ls

from lipsync_gpu import prepare_face_green_screen, run_wav2lip

os.environ.setdefault("VEEGEN_CMD_TIMEOUT", "540")


class VeeGenLipSyncAPI(ls.LitAPI):
    def setup(self, device):
        self.device = device

    def decode_request(self, request):
        return request

    def predict(self, data):
        mode = (data.get("mode") or "lipsync_only").strip().lower()
        if mode != "lipsync_only":
            raise ValueError("This worker supports only mode='lipsync_only'.")

        face_b64 = data.get("face_b64") or ""
        audio_b64 = data.get("audio_b64") or ""
        face_filename = data.get("face_filename") or f"face_{uuid.uuid4().hex}.jpg"
        audio_filename = data.get("audio_filename") or "voice.wav"
        if not face_b64:
            raise ValueError("face_b64 is required")
        if not audio_b64:
            raise ValueError("audio_b64 is required")

        face_ext = os.path.splitext(face_filename)[1].lower()
        if face_ext not in (".jpg", ".jpeg", ".png", ".webp"):
            face_ext = ".jpg"
        audio_ext = os.path.splitext(audio_filename)[1].lower()
        if audio_ext not in (".wav", ".mp3", ".m4a", ".aac"):
            audio_ext = ".wav"

        with tempfile.TemporaryDirectory() as tmp:
            face_path = os.path.join(tmp, f"face{face_ext}")
            audio_path = os.path.join(tmp, f"audio{audio_ext}")
            green_face = os.path.join(tmp, "face_green.png")
            output_path = os.path.join(tmp, "lipsync_raw.mp4")

            with open(face_path, "wb") as f:
                f.write(base64.b64decode(face_b64))
            with open(audio_path, "wb") as f:
                f.write(base64.b64decode(audio_b64))

            prepare_face_green_screen(face_path, green_face)
            run_wav2lip(green_face, audio_path, output_path, resize_factor=2)

            with open(output_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode("ascii")

        return {
            "filename": "lipsync_raw.mp4",
            "video_b64": video_b64,
        }

    def encode_response(self, output):
        return output


if __name__ == "__main__":
    server = ls.LitServer(VeeGenLipSyncAPI(), accelerator="auto", timeout=60 * 45)
    server.run(port=int(os.environ.get("PORT", "8000")))
