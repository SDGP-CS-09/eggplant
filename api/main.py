from fastapi import FastAPI, File, UploadFile, HTTPException  # Import FastAPI and necessary modules for file handling and exceptions
import os  # For accessing environment variables
import uvicorn  # For running the FastAPI server
import numpy as np  # For numerical operations and image array handling
from io import BytesIO  # For handling binary data streams
from PIL import Image  # For image processing
import tensorflow as tf  # For loading and using the machine learning model

# Initialize the FastAPI application
app = FastAPI()

# Load the pre-trained TensorFlow model with error handling
try:
    MODEL = tf.keras.models.load_model("my_model.h5")  # Attempt to load the model from file
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")  # Raise an error if model loading fails

# Define the list of possible prediction classes for eggplant diseases
CLASS_NAMES = ["Augmented Healthy Leaf",
 "Augmented Insect Pest Disease",
 "Augmented Leaf Spot Disease",
 "Augmented Mosaic Virus Disease",
 "Augmented Small Leaf Disease",
 "Augmented White Mold Disease",
 "Augmented Wilt Disease"]

# Define a simple health check endpoint
@app.get("/ping")
async def ping():
    """Returns a welcome message to verify API is running"""
    return "Welcome to Eggplant disease prediction API!"  # Return a static welcome message

# Function to convert uploaded image data into a numpy array
def read_file_as_image(data) -> np.ndarray:
    """
    Convert binary image data to numpy array
    Args:
        data: Binary image data from the uploaded file
    Returns:
        np.ndarray: Image converted to a numpy array
    Raises:
        HTTPException: If image processing fails (e.g., corrupt file)
    """
    try:
        image = np.array(Image.open(BytesIO(data)))  # Open image from binary data and convert to numpy array
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")  # Handle image processing errors

# Function to check if the image is likely an eggplant leaf
def is_likely_eggplant_leaf(image: np.ndarray) -> bool:
    """
    Basic heuristic to check if image is likely an eggplant leaf
    Args:
        image: Image as numpy array
    Returns:
        bool: True if likely an eggplant leaf, False otherwise
    """
    try:
        # Calculate the average RGB color across the image
        mean_color = np.mean(image, axis=(0, 1))  # Average over height and width
        r, g, b = mean_color  # Extract red, green, blue values
        
        # Check if green is dominant (typical for leaves) and above a threshold
        if g > r and g > b and g > 50:  # Green should be prominent
            return True
        return False  # Return False if green isn't dominant
    except Exception:
        return False  # If analysis fails, conservatively assume it's not an eggplant leaf

# Define the prediction endpoint for disease classification
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)  # Expect an uploaded file as input
):
    """
    Predict eggplant disease from uploaded image
    Args:
        file: Uploaded image file (multipart/form-data)
    Returns:
        dict: Prediction results with class and confidence, or error message if not an eggplant leaf
    Raises:
        HTTPException: If prediction fails or input is invalid
    """
    try:
        # Validate that a file was uploaded
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")
            
        # Validate that the uploaded file is an image
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
            
        # Convert the uploaded file to a numpy array
        image = read_file_as_image(await file.read())
        
        # Check if the image is likely an eggplant leaf
        if not is_likely_eggplant_leaf(image):
            return {
                "message": "This does not appear to be an eggplant leaf image",  # Inform user it's not a valid input
                "class": None,  # No class predicted
                "confidence": None  # No confidence score
            }
        
        # Prepare image for model prediction by adding a batch dimension
        img_batch = np.expand_dims(image, 0)
        predictions = MODEL.predict(img_batch)  # Run the model to get predictions

        # Extract the predicted class and confidence from the model output
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]  # Get the class with highest probability
        confidence = np.max(predictions[0])  # Get the highest confidence score
        
        # Return the prediction result
        return {
            'class': predicted_class,
            'confidence': float(confidence)  # Convert to float for JSON compatibility
        }
    except HTTPException as e:
        raise e  # Re-raise HTTP exceptions (e.g., 400 errors)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")  # Handle unexpected errors

# Main execution block to start the server
if __name__ == "__main__":
    """
    Start the FastAPI server
    Uses port from environment variable PORT or defaults to 8060
    """
    try:
        # Get the port number from environment variable or use default
        port = int(os.getenv("PORT", 8060))
        # Run the FastAPI application using uvicorn
        uvicorn.run(app, host="0.0.0.0", port=port)
    except ValueError as e:
        print(f"Invalid port number: {str(e)}")  # Handle invalid port number
    except Exception as e:
        print(f"Server failed to start: {str(e)}")  # Handle server startup errors
