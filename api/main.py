from fastapi import FastAPI, File, UploadFile, HTTPException
import os
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

# Initialize FastAPI application
app = FastAPI()

# Load the pre-trained model with error handling
try:
    MODEL = tf.keras.models.load_model("my_model.h5")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

# Define class names for prediction output
CLASS_NAMES = ["Augmented Healthy Leaf",
 "Augmented Insect Pest Disease",
 "Augmented Leaf Spot Disease",
 "Augmented Mosaic Virus Disease",
 "Augmented Small Leaf Disease",
 "Augmented White Mold Disease",
 "Augmented Wilt Disease"]

# Health check endpoint
@app.get("/ping")
async def ping():
    """Returns a welcome message to verify API is running"""
    return "Welcome to Eggplant disease prediction API!"

# Function to convert uploaded file to numpy array
def read_file_as_image(data) -> np.ndarray:
    """
    Convert binary image data to numpy array
    Args:
        data: Binary image data
    Returns:
        np.ndarray: Image as numpy array
    Raises:
        HTTPException: If image processing fails
    """
    try:
        image = np.array(Image.open(BytesIO(data)))
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

# Prediction endpoint
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)  # File parameter for image upload
):
    """
    Predict eggplant disease from uploaded image
    Args:
        file: Uploaded image file
    Returns:
        dict: Prediction results with class and confidence
    Raises:
        HTTPException: If prediction fails or input is invalid
    """
    try:
        # Validate file existence
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")
            
        # Validate file type is image
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
            
        # Read and process image
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)  # Add batch dimension
        
        # Make prediction using the loaded model
        predictions = MODEL.predict(img_batch)

        # Get predicted class and confidence
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        
        # Return results as dictionary
        return {
            'class': predicted_class,
            'confidence': float(confidence)
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Main execution block
if __name__ == "__main__":
    """
    Start the FastAPI server
    Uses port from environment variable or defaults to 8060
    """
    try:
        # Get port from environment or use default
        port = int(os.getenv("PORT", 8060))
        # Run the server
        uvicorn.run(app, host="0.0.0.0", port=port)
    except ValueError as e:
        print(f"Invalid port number: {str(e)}")
    except Exception as e:
        print(f"Server failed to start: {str(e)}")
