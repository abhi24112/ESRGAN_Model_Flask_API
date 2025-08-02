# Environment variables
import os
import time
import gc # garbage collection

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Disable TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# Importing Modules
import io
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename # "My cool movie.mov" ---> 'My_cool_movie.mov'

# Cors for limiting web client request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


# global variable model
model = None

def load_model():
    global model 
    if model is None:
        print("Loading Model...")
        try:
            # Check available devices
            print("Available devices:")
            for device in tf.config.list_physical_devices():
                print(f"  {device}")
                
            model_path = "ESRGAN_Model"
            model = hub.load(model_path)
            print("Model Loaded Successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

app = Flask(__name__)

CORS(app)
# making web client request limited
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["10 per minute"]  # Limit to 10 requests per minute
)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAIN_IMAGE_FOLDER = 'images'
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'

# making directories
os.makedirs(f"{MAIN_IMAGE_FOLDER}/{UPLOAD_FOLDER}", exist_ok=True)
os.makedirs(f"{MAIN_IMAGE_FOLDER}/{OUTPUT_FOLDER}",exist_ok=True)

# Loading the model
print("Initializing server...")
load_model()
print("Server Initialized Successfully...")



# Preprocessing functions
def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Loads image from path and preprocesses to make it model ready
    Args:
        image_path: Path to the image file
    """
    hr_image = tf.image.decode_image(tf.io.read_file(image_path))
    # If PNG, remove the alpha channel. The model only supports
    # images with 3 color channels.
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[..., :-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)

def preprocess_image_from_bytes(image_bytes):
    """Preprocess image from bytes data"""
    hr_image = tf.image.decode_image(image_bytes)
    # If PNG, remove the alpha channel. The model only supports
    # images with 3 color channels.
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[..., :-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)

def save_image_to_bytes(image_tensor):
    """Convert tensor to PIL Image and then to bytes"""
    image = tf.clip_by_value(image_tensor, 0, 255)
    image = tf.cast(image, tf.uint8).numpy()
    pil_image = Image.fromarray(image)
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='JPEG', quality=95)
    img_byte_arr.seek(0)
    return img_byte_arr


@app.route("/", methods=["GET"])
def home():
    return jsonify("home")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy" if model is not None else "Model not Loaded.",
        "model_loaded": model is not None
    }), 200

@app.route("/upscale", methods=["POST"])
def upscale_image():
    if model is None:
        return jsonify({
            "error": "Sorry Model is not Loaded.."
        }), 500
    
    if "file" not in request.files:
        return jsonify({"error": "No file  in the request"}), 400
    
    file = request.files["file"]

    # If user does not select file, browser also submits an empty part without filename
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Read image bytes
            image_bytes = file.read()

            # Add image size validation
            if len(image_bytes) > 16 * 1024 * 1024:  # 16MB
                return jsonify({"error": "File too large. Max size: 16MB"}), 400
            
            # Preprocess image
            hr_image = preprocess_image_from_bytes(image_bytes)
            print("Please wait Image is Generating...")
            # Generate super resolution image
            start_time = time.time()
            fake_image = model(hr_image)
            fake_image = tf.squeeze(fake_image)
            # Convert to bytes
            result_image_bytes = save_image_to_bytes(fake_image)
            processing_time =  time.time() - start_time
            print("Image is Generating in:", processing_time)

            # Memory Cleanup 
            del hr_image, image_bytes, fake_image
            gc.collect()

            # Return the processed image
            return send_file(
                result_image_bytes,
                mimetype='image/jpeg',
                as_attachment=True,
                download_name=f"sr_{secure_filename(file.filename)}"
            )
        except Exception as e:
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500
        
    return jsonify({"error": "Invalid file type"}), 400




if __name__ == "__main__":
    port=int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port = port, debug=False)



