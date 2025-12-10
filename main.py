from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from gradio_client import Client, handle_file
from PIL import Image
import uuid
import os
import shutil

# ----------------------------
# 1. Basic setup
# ----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev; tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
OUTPUT_DIR = os.path.join(BASE_DIR, "generated")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# serve /generated files as static
app.mount("/generated", StaticFiles(directory=OUTPUT_DIR), name="generated")

# InstantID client (Hugging Face space id)
client = Client("InstantX/InstantID")


# ----------------------------
# 2. Health check
# ----------------------------
@app.get("/")
def root():
    return {"message": "InstantID backend running"}


# ----------------------------
# 3. Helper: call InstantID
# ----------------------------
def run_instantid(face_path: str, template_path: str, prompt: str) -> str:
    """
    Call InstantID with local file paths (face + template).
    Returns the path to the generated image file (on local disk).
    """

    result = client.predict(
        face_image_path=handle_file(face_path),
        pose_image_path=handle_file(template_path),
        prompt=prompt,
        negative_prompt=(
            "(lowres, low quality, worst quality:1.2), "
            "(text:1.2), watermark, (frame:1.2), deformed, ugly, "
            "deformed eyes, blur, out of focus, blurry, deformed cat, "
            "deformed, photo, anthropomorphic cat, monochrome, pet collar, "
            "gun, weapon, blue, 3d, drones, drone, buildings in background, green"
        ),
        style_name="Spring Festival",
        num_steps=30,
        identitynet_strength_ratio=0.8,
        adapter_strength_ratio=0.8,
        canny_strength=0.4,
        depth_strength=0.4,
        controlnet_selection=["depth"],
        guidance_scale=5,
        seed=42,
        scheduler="EulerDiscreteScheduler",
        enable_LCM=False,      # note: matches your working example
        enhance_face_region=True,
        api_name="/generate_image",
    )

    generated_image_path = result[0]  # (generated_images, tips)
    return generated_image_path


# ----------------------------
# 4. Main endpoint
# ----------------------------
@app.post("/personalize")
async def personalize(
    image: UploadFile = File(...),
    template_id: str = Form("template_1"),
    prompt: str = Form("make brighter picture"),
):
    """
    1. Save uploaded child photo to disk
    2. Use InstantID with that photo + chosen template
    3. Copy result into ./generated
    4. Return URL to final image
    """

    # 1. Save uploaded image to a local file
    try:
        raw_bytes = await image.read()
        upload_name = f"upload_{uuid.uuid4().hex}.png"
        upload_path = os.path.join(OUTPUT_DIR, upload_name)

        with open(upload_path, "wb") as f:
            f.write(raw_bytes)
    except Exception as e:
        return JSONResponse(status_code=400, content={"detail": f"Failed to save upload: {e}"})

    # 2. Resolve template path
    template_filename = f"{template_id}.png"
    template_path = os.path.join(TEMPLATES_DIR, template_filename)

    if not os.path.exists(template_path):
        return JSONResponse(
            status_code=400,
            content={"detail": f"Template not found: {template_filename}"}
        )

    # 3. Call InstantID
    try:
        instantid_out_path = run_instantid(upload_path, template_path, prompt)
    except Exception as e:
        # still return something instead of crashing
        return JSONResponse(
            status_code=500,
            content={"detail": f"InstantID error: {e}"}
        )

    # 4. Copy InstantID output into our OUTPUT_DIR with a nice name
    final_name = f"result_{uuid.uuid4().hex}.png"
    final_path = os.path.join(OUTPUT_DIR, final_name)

    try:
        shutil.copy(instantid_out_path, final_path)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to copy result image: {e}"}
        )

    # 5. Return URL (this is what frontend will display)
    # If running with uvicorn on localhost:8000 → full URL = http://localhost:8000/generated/<final_name>
    return {"result_url": f"/generated/{final_name}"}
