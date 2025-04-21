import cv2
import numpy as np
import base64
import json
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Image Processor API")

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the current image
current_image = None
image_path = os.path.join(os.path.dirname(__file__), '_temp_image.npy')

def save_current_image(image):
    """Save the current image to disk"""
    global current_image
    current_image = image
    np.save(image_path, image)

def load_current_image():
    """Load the current image from disk"""
    global current_image
    if current_image is not None:
        return current_image
    if os.path.exists(image_path):
        return np.load(image_path)
    return None

def adjust_brightness_contrast(image, alpha, beta):
    """Adjust the brightness and contrast of an image"""
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def generate_histogram(image):
    """Generate a histogram of the image with improved visualization"""
    fig = Figure(figsize=(4, 3), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    
    colors = ('b', 'g', 'r')
    channel_names = ['Blue', 'Green', 'Red']
    
    # Calculate y-max to set appropriate scale
    y_max = 0
    histograms = []
    
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        histograms.append(hist)
        y_max = max(y_max, np.max(hist))
    
    # Add some headroom to prevent cutting off peaks
    y_max = y_max * 1.1
    
    # Plot with consistent y-axis
    for i, (hist, color, name) in enumerate(zip(histograms, colors, channel_names)):
        ax.plot(hist, color=color, label=name)
    
    ax.set_ylim([0, y_max])
    ax.set_xlim([0, 256])
    ax.set_title("Histogram")
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    ax.legend(loc='upper right')
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure proper layout
    fig.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    return base64.b64encode(buf.getvalue()).decode('utf-8')

class AdjustmentParams(BaseModel):
    alpha: float
    beta: int

# Change path to /api/upload to match React component
@app.post("/api/upload")
async def upload_image(image: UploadFile = File(...)):  # Changed parameter name to 'image'
    """Handle image upload request"""
    try:
        # Read and decode the image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Save the image
        save_current_image(image)
        
        # Generate base64 strings for the image and its histogram
        _, img_encoded = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        hist_base64 = generate_histogram(image)
        
        return JSONResponse(content={
            'original_image': img_base64,
            'original_histogram': hist_base64
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Change path to /api/adjust to match React component
@app.post("/api/adjust")
async def adjust_image(params: AdjustmentParams):
    """Handle image adjustment request"""
    try:
        # Load the current image
        image = load_current_image()
        
        if image is None:
            raise HTTPException(status_code=400, detail="No image available")
        
        # Apply adjustments
        adjusted_image = adjust_brightness_contrast(image, params.alpha, params.beta)
        
        # Generate base64 strings for the adjusted image and its histogram
        _, img_encoded = cv2.imencode('.png', adjusted_image)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        hist_base64 = generate_histogram(adjusted_image)
        
        return JSONResponse(content={
            'adjusted_image': img_base64,
            'adjusted_histogram': hist_base64
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Change path to /api/clear to match React component
@app.post("/api/clear")
async def clear_image():
    """Handle image clear request"""
    global current_image
    current_image = None
    
    if os.path.exists(image_path):
        os.remove(image_path)
    
    return JSONResponse(content={'message': 'Image cleared'})

# For testing in development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)