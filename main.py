from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from gradio_client import Client, handle_file
from PIL import Image  # still OK to keep, even if not used directly
import uuid
import os
import shutil


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


app.mount("/generated", StaticFiles(directory=OUTPUT_DIR), name="generated")

HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN");
if HF_TOKEN : 
    client = Client("InstantX/InstantID",token=HF_TOKEN)
else:
    client = Client("InstantX/InstantID")



@app.get("/")
def root():
    return {"message": "InstantID backend running"}



def run_instantid(face_path: str, pose_path: str, prompt: str,template:str| None = None) -> str:
    

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
        style_name=template,
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

    print(result)

    generated_image_path = result[0]  # (generated_images, tips)
    return generated_image_path


@app.post("/personalize")
async def personalize(
   
    image_main: UploadFile = File(..., description="Required main image"),
   
    image_optional: UploadFile | None = File(
        None, description="Optional personalization image"
    ),
    prompt: str = Form("make brighter picture"),

    template: str = Form("Line art")
):
   

   
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

    
    pose_path = main_path  
    if image_optional is not None:
        try:
            opt_bytes = await image_optional.read()
            opt_name = f"opt_{uuid.uuid4().hex}.png"
            opt_path = os.path.join(OUTPUT_DIR, opt_name)

            with open(opt_path, "wb") as f:
                f.write(opt_bytes)

            pose_path = opt_path
        except Exception as e:
          
            print(f"Failed to save optional image, falling back to main: {e}")
            pose_path = main_path

   
    try:
        instantid_out_path = run_instantid(main_path, pose_path, prompt,template=template)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"InstantID error: {e}"},
        )

    
    final_name = f"result_{uuid.uuid4().hex}.png"
    final_path = os.path.join(OUTPUT_DIR, final_name)

    try:
        shutil.copy(instantid_out_path, final_path)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to copy result image: {e}"},
        )

    
    return {"result_url": f"/generated/{final_name}"}
