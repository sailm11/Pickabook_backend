from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from gradio_client import Client, handle_file
from PIL import Image  # still OK to keep, even if not used directly
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
def run_instantid(face_path: str, pose_path: str, prompt: str) -> str:
    """
    Call InstantID with local file paths (face + pose).
    Returns the path to the generated image file (on local disk).
    """

    result = client.predict(
        face_image_path=handle_file(face_path),
        pose_image_path=handle_file(pose_path),
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
        enable_LCM=False,
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
    # main required image (face)
    image_main: UploadFile = File(..., description="Required main image"),
    # optional second image for personalization / pose
    image_optional: UploadFile | None = File(
        None, description="Optional personalization image"
    ),
    prompt: str = Form("make brighter picture"),
):
    """
    1. Save uploaded main photo (required) to disk
    2. If optional second image is provided, save and use as pose image
       Otherwise, reuse main image as pose (so API requirements are satisfied)
    3. Call InstantID
    4. Copy result into ./generated
    5. Return URL to final image
    """

    # 1. Save required main image
    try:
        raw_bytes = await image_main.read()
        main_name = f"main_{uuid.uuid4().hex}.png"
        main_path = os.path.join(OUTPUT_DIR, main_name)

        with open(main_path, "wb") as f:
            f.write(raw_bytes)
    except Exception as e:
        return JSONResponse(
            status_code=400, content={"detail": f"Failed to save main image: {e}"}
        )

    # 2. Save optional personalization image (if provided)
    pose_path = main_path  # default: reuse main image as pose
    if image_optional is not None:
        try:
            opt_bytes = await image_optional.read()
            opt_name = f"opt_{uuid.uuid4().hex}.png"
            opt_path = os.path.join(OUTPUT_DIR, opt_name)

            with open(opt_path, "wb") as f:
                f.write(opt_bytes)

            pose_path = opt_path
        except Exception as e:
            # If optional image fails to save, we just log it and fall back to main image as pose
            # (You could also return 400 if you want stricter behavior)
            print(f"Failed to save optional image, falling back to main: {e}")
            pose_path = main_path

    # 3. Call InstantID
    try:
        instantid_out_path = run_instantid(main_path, pose_path, prompt)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"InstantID error: {e}"},
        )

    # 4. Copy InstantID output into our OUTPUT_DIR with a nice name
    final_name = f"result_{uuid.uuid4().hex}.png"
    final_path = os.path.join(OUTPUT_DIR, final_name)

    try:
        shutil.copy(instantid_out_path, final_path)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to copy result image: {e}"},
        )

    # 5. Return URL (this is what frontend will display)
    return {"result_url": f"/generated/{final_name}"}
