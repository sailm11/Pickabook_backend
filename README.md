✅ InstantID Personalizer – Backend (FastAPI)

Author: SAIl MORYE | Deployment-ready backend documentation

This backend powers the InstantID Personalizer project.
It handles:

Uploading main + optional images

Passing them to the InstantX/InstantID HuggingFace Space

Managing HF authentication

Saving generated images

Returning public URLs to the frontend

Backend is deployed on Render.
Frontend is deployed on Vercel.

📌 1. Features

FastAPI server with /personalize endpoint

Accepts:

image_main (required)

image_optional (optional)

prompt (string)

Calls HuggingFace Space using gradio_client

Saves output inside /generated

Serves images as static files

Handles ZeroGPU quota properly via HF token

Fully deployable on Render

ackend/
│── main.py
│── requirements.txt
│── generated/            # output images (auto-created)
│── README.md             # this file
└── ...

🔧 3. Environment Variables (required for successful inference)

Set these on Render → Environment → Add Variable.

HF_TOKEN (REQUIRED)
Key: HF_TOKEN
Value: hf_xxxxxxxxxxxxxxxxxx


🧪 4. Install & Run (Local Development)
1️⃣ Create virtual environment
cd backend
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

🧾 5. Requirements

Typical dependencies (generated via pip freeze):
fastapi
uvicorn[standard]
python-multipart
pillow
gradio_client
aiofiles

🚀 6. Deploying on Render
pip install -r requirements.txt

Start Command (VERY IMPORTANT)
uvicorn main:app --host 0.0.0.0 --port $PORT

Render injects the port automatically.

Environment Variables

HF_TOKEN = hf_xxxxxxxxxxxxxxxxxxx

