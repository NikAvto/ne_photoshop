from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io, base64
import numpy as np
from scipy.signal import convolve2d

app = Flask(__name__)
CORS(app)

def gaussian_kernel(size, sigma=1):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def averaging_kernel(size):
    return np.ones((size, size)) / (size * size)

def apply_filter(image, kernel):
    output = np.zeros_like(image)
    for c in range(image.shape[2]):
        output[:, :, c] = convolve2d(image[:, :, c], kernel, mode='same', boundary='fill', fillvalue=0)
    return output

def black_white_filter(image):
    h, w, _ = image.shape
    bw_image = np.zeros_like(image)
    for x in range(h):
        for y in range(w):
            r, g, b = image[x, y]
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            bw_image[x, y] = [brightness, brightness, brightness]
    return bw_image

def sobel_filter(image):
    gray = np.mean(image, axis=2)
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)
    Gx = convolve2d(gray, Kx, mode='same', boundary='symm')
    Gy = convolve2d(gray, Ky, mode='same', boundary='symm')
    magnitude = np.sqrt(Gx**2 + Gy**2)
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
    return np.stack([magnitude]*3, axis=2)

def prewitt_filter(image):
    gray = np.mean(image, axis=2)
    Kx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[-1, -1, -1],
                   [ 0,  0,  0],
                   [ 1,  1,  1]], dtype=np.float32)
    Gx = convolve2d(gray, Kx, mode='same', boundary='symm')
    Gy = convolve2d(gray, Ky, mode='same', boundary='symm')
    magnitude = np.sqrt(Gx**2 + Gy**2)
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
    return np.stack([magnitude]*3, axis=2)

def kmeans_filter(image, k=4, max_iter=10):
    h, w, c = image.shape
    data = image.reshape((-1, 3))
    
    np.random.seed(42)
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    
    for _ in range(max_iter):
        distances = np.linalg.norm(data[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([data[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j] for j in range(k)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    clustered = centroids[labels].reshape((h, w, 3))
    return clustered

@app.route('/apply_filter', methods=['POST'])
def apply_filter_route():
    try:
        data = request.get_json()
        img_b64 = data.get('image')
        filter_type = data.get('filter')

        if not img_b64 or not filter_type:
            return jsonify({'error': 'No image or filter type provided'}), 400

        img_bytes = base64.b64decode(img_b64.split(',')[1])
        img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        image = np.asarray(img_pil).astype(np.float32) / 255.0

        if filter_type == 'gaussian':
            kernel = gaussian_kernel(7, sigma=1.5)
            filtered = apply_filter(image, kernel)
        elif filter_type == 'average':
            kernel = averaging_kernel(7)
            filtered = apply_filter(image, kernel)
        elif filter_type == 'black_white':
            filtered = black_white_filter(image)
        elif filter_type == 'sobel':
            filtered = sobel_filter(image)
        elif filter_type == 'prewitt':
            filtered = prewitt_filter(image)
        elif filter_type == 'kmeans':
            filtered = kmeans_filter(image, k=4)
        else:
            return jsonify({'error': 'Unknown filter'}), 400
            
        filtered_img = Image.fromarray(np.clip(filtered * 255, 0, 255).astype(np.uint8))
        buffer = io.BytesIO()
        filtered_img.save(buffer, format='PNG')
        img_b64_out = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jsonify({'image': img_b64_out})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
