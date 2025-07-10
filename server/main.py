from io import BytesIO

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import JSONResponse
import base64
from dptUtil import DPTModelRunner
from PIL import Image

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def image_to_base64(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format='png')
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    image = await file.read()

    dptModelRunner = DPTModelRunner()

    prediction = dptModelRunner.predict(image)

    buf_plot, buf_depth = dptModelRunner.createDepthFile(prediction)

    # return StreamingResponse(buf, media_type="image/png")
    return JSONResponse({
        "depth_map": image_to_base64(Image.open(buf_depth)),
        "depth_plot": image_to_base64(Image.open(buf_plot))
    })