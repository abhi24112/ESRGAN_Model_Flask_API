# SRGAN Flask API

A Flask-based REST API for Super Resolution using SRGAN (Super-Resolution Generative Adversarial Network).

## Features

- **File Upload**: Upload image files directly for super resolution
- **Base64 Support**: Send base64 encoded images via JSON
- **Multiple Formats**: Supports PNG, JPG, JPEG, GIF, BMP, TIFF
- **Health Check**: Monitor API and model status
- **Error Handling**: Comprehensive error handling and validation

## Setup

### 1. Install Dependencies

```cmd
pip install -r requirements.txt
```

### 2. Model Structure

Make sure your model is in the `SRGAN_Model` folder with the following structure:

```
SRGAN_Model/
├── saved_model.pb
└── variables/
    ├── variables.data-00000-of-00001
    └── variables.index
```

### 3. Run the Server

```cmd
python server.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### 1. Health Check

```
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. Home/Info

```
GET /
```

**Response:**

```json
{
  "message": "SRGAN Super Resolution API",
  "version": "1.0",
  "endpoints": {...},
  "model_loaded": true
}
```

### 3. File Upload Super Resolution

```
POST /upscale
Content-Type: multipart/form-data
```

**Parameters:**

- `file`: Image file (PNG, JPG, JPEG, GIF, BMP, TIFF)

**Response:**

- Returns the super resolution image file

**Example using curl:**

```cmd
curl -X POST -F "file=@your_image.jpg" http://localhost:5000/upscale --output sr_result.jpg
```

### 4. Base64 Super Resolution

```
POST /upscale_base64
Content-Type: application/json
```

**Request Body:**

```json
{
  "image": "base64_encoded_image_data"
}
```

**Response:**

```json
{
  "success": true,
  "result_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA...",
  "processing_time": 2.45,
  "original_shape": [256, 256],
  "result_shape": [1024, 1024]
}
```

## Usage Examples

### Python Client Example

```python
import requests
import base64

# File upload
with open('image.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/upscale', files={'file': f})
    with open('sr_result.jpg', 'wb') as out:
        out.write(response.content)

# Base64 upload
with open('image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

response = requests.post('http://localhost:5000/upscale_base64',
                        json={'image': image_data})
result = response.json()
```

### JavaScript/Fetch Example

```javascript
// File upload
const formData = new FormData();
formData.append("file", fileInput.files[0]);

fetch("http://localhost:5000/upscale", {
  method: "POST",
  body: formData,
})
  .then((response) => response.blob())
  .then((blob) => {
    const url = URL.createObjectURL(blob);
    // Use the image URL
  });

// Base64 upload
const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");
// ... draw image to canvas
const base64Data = canvas.toDataURL("image/jpeg").split(",")[1];

fetch("http://localhost:5000/upscale_base64", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    image: base64Data,
  }),
})
  .then((response) => response.json())
  .then((data) => {
    // data.result_image contains the base64 result
  });
```

## Testing

Use the provided test client:

```cmd
python test_client.py
```

Make sure to place a test image in the current directory and update the `test_image` variable in the script.

## Configuration

### Environment Variables

- `FLASK_ENV`: Set to `development` for debug mode
- `TF_CPP_MIN_LOG_LEVEL`: Controls TensorFlow logging (default: 2)

### App Configuration

- Maximum file size: 16MB
- Supported formats: PNG, JPG, JPEG, GIF, BMP, TIFF
- Default host: 0.0.0.0
- Default port: 5000

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid file, missing parameters)
- `413`: File too large (>16MB)
- `500`: Internal server error (model not loaded, processing error)

## Model Requirements

The SRGAN model should be a TensorFlow SavedModel that:

- Accepts input tensors of shape `[batch, height, width, 3]`
- Returns output tensors with super resolution images
- Works with float32 input values

## Troubleshooting

### Model Loading Issues

- Verify the model path in `MODEL_PATH`
- Check that all model files are present
- Ensure TensorFlow version compatibility

### Memory Issues

- Reduce input image size
- Use smaller batch sizes
- Consider model optimization

### Performance

- Use GPU if available
- Optimize TensorFlow for your hardware
- Consider image preprocessing optimizations
