from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
from PIL import Image
import cairosvg

# Import DeepSVG related code
import os
os.chdir("..")
from deepsvg.svglib.svg import SVG
from deepsvg import utils
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svglib.utils import to_gif
from deepsvg.svglib.geom import Bbox
from deepsvg.svgtensor_dataset import SVGTensorDataset, load_dataset
from deepsvg.utils.utils import batchify
import torch
from scipy.spatial.distance import cosine

app = Flask(__name__)
CORS(app)  # Allow requests from frontend

# Setup device and model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrained_path = "./pretrained/hierarchical_ordered.pth.tar"
from configs.deepsvg.hierarchical_ordered import Config
cfg = Config()
model = cfg.make_model().to(device)
utils.load_model(pretrained_path, model)
model.eval()
dataset = load_dataset(cfg)

DATASET_DIR = "./dataset/Registered_Dataset_simplified"
PNG_OUTPUT_DIR = "./dataset/rendered_pngs"

# Helper functions
def load_svg(filename):
    svg = SVG.load_svg(filename)
    svg.canonicalize()
    svg.normalize()
    svg.zoom(0.9)
    svg = svg.simplify_heuristic()
    svg = svg.numericalize(256)
    return svg

def encode(data):
    model_args = batchify((data[key] for key in cfg.model_args), device)
    with torch.no_grad():
        z = model(*model_args, encode_mode=True)
        return z

def encode_svg(svg):
    data = dataset.get(svg=svg)
    return encode(data)

def load_and_encode(svg_path):
    try:
        svg = load_svg(svg_path)
        vector = encode_svg(svg)
        return vector
    except Exception as e:
        print(f"Encoding failed for {svg_path}: {e}")
        return None

@app.route('/api/lookup-logo', methods=['POST'])
def lookup_logo():
    if 'logo' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['logo']
    if not file or file.filename == '':
        return jsonify({'error': 'Invalid file'}), 400

    if not file.filename.endswith('.svg'):
        return jsonify({'error': 'Only SVG files are allowed'}), 400

    # Save uploaded SVG temporarily
    temp_id = str(uuid.uuid4())
    temp_path = f"./temp/{temp_id}.svg"
    os.makedirs("./temp", exist_ok=True)
    file.save(temp_path)

    target_vector = load_and_encode(temp_path)
    if target_vector is None:
        return jsonify({'error': 'SVG encoding failed'}), 500

    similarities = []
    for fname in os.listdir(DATASET_DIR):
        if not fname.endswith('.svg'):
            continue
        candidate_path = os.path.join(DATASET_DIR, fname)
        candidate_vector = load_and_encode(candidate_path)
        if candidate_vector is None:
            continue
        similarity = 1 - cosine(target_vector, candidate_vector)
        similarities.append((fname, similarity, candidate_path))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_matches = similarities[:3]

    results = []
    for fname, score, path in top_matches:
        # Convert to PNG
        png_name = fname.replace('.svg', '.png')
        png_path = os.path.join(PNG_OUTPUT_DIR, png_name)
        if not os.path.exists(png_path):
            try:
                cairosvg.svg2png(file_obj=open(path, "rb"), write_to=png_path)
            except:
                continue

        results.append({
            "logoUrl": f"http://localhost:5000/static/{png_name}",
            "companyUrl": f"https://example.com/brand/{fname.replace('.svg','')}"  # Replace with real metadata
        })

    os.remove(temp_path)  # Clean up
    return jsonify({'matches': results})

# Serve static images
@app.route('/static/<filename>')
def serve_png(filename):
    return app.send_from_directory(PNG_OUTPUT_DIR, filename)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
