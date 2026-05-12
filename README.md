# VeeGen Lightning GPU Worker

This folder is the minimal Lightning AI worker for VeeGen. It only runs the
GPU-heavy Wav2Lip step.

The main VeeGen app stays on your local machine or VPS. It sends:

- face image as base64
- extracted voice audio as base64

The worker returns:

- raw lip-sync video as base64

## Run In Lightning Studio

Open a GPU Studio, then upload or clone only this `lightning_worker` folder.

```bash
cd lightning_worker
pip install -r requirements.txt
python setup_worker.py
python app.py
```

Expose port `8000` publicly in Lightning. Paste the public URL into VeeGen:

`GPU Backend` -> `Lightning AI LitServe` -> `Backend URL`

