# =========================
# Standard Library Imports
# =========================
import os
import uuid
import io
import base64

# =========================
# Flask and Web
# =========================
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# =========================
# Image Processing
# =========================
from PIL import Image as ImagePil
import cairosvg
from IPython.display import display

# =========================
# Math, Data, and ML
# =========================
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine, euclidean, directed_hausdorff
from scipy.spatial import procrustes
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix

# =========================
# DeepSVG
# =========================
from deepsvg.svglib.svg import SVG
from deepsvg import utils
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svglib.utils import to_gif
from deepsvg.svglib.geom import Bbox
from deepsvg.svgtensor_dataset import SVGTensorDataset, load_dataset
from deepsvg.utils.utils import batchify, linear
from configs.deepsvg.hierarchical_ordered import Config

# =========================
# Vector DB and DB Clients
# =========================
from pymilvus import connections, FieldSchema, DataType, CollectionSchema, Collection
import pymongo

# =========================
# SVG Utilities
# =========================
from svgpathtools import svg2paths, Path



device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu") 

# Safer path handling to avoid issues with os.chdir
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pretrained_path = os.path.join(BASE_DIR, "pretrained", "hierarchical_ordered.pth.tar")


try:
    state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))  # Use 'cuda' if you have GPU
    print("File loaded successfully.")
    print(state_dict.keys())  # Prints the keys to verify the content
except Exception as e:
    print(f"Error loading the file: {e}")

cfg = Config()
model = cfg.make_model().to(device)
utils.load_model(pretrained_path, model)
model.eval();
dataset = load_dataset(cfg)

def load_svg(filename):
    svg = SVG.load_svg(filename)
    svg.canonicalize()
    svg.normalize()
    svg.zoom(0.9)
    svg = svg.simplify_heuristic()
    svg =svg.numericalize(256)
    return svg

def encode(data):
    model_args = batchify((data[key] for key in cfg.model_args), device)
    with torch.no_grad():
        z = model(*model_args, encode_mode=True)
        return z
    
def encode_svg(svg):
    data = dataset.get(svg=svg)
    return encode(data)


# Connect to Zilliz Cloud
ENDPOINT = "https://in03-754f3454a65e40f.serverless.gcp-us-west1.cloud.zilliz.com"
TOKEN = "2b830a69fb087e580f904877ff816ff1477e67a38c091fc6b8c9c75d3992a458cc2deb681d3ae18dd91900855e1c538013080bf5"
connections.connect(uri=ENDPOINT, token=TOKEN)
# print("Connected to Zilliz Cloud!")

# Load existing collection
collection_name = "fyp_project"
collection = Collection(name=collection_name)


# MongoDB Atlas URI (make sure to keep this secret!)
MONGO_URI = "mongodb+srv://hiru23anjalee:p24gomepFiz7R9HB@cluster0.kt9cubt.mongodb.net/?retryWrites=true&w=majority"

try:
    mongo_client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    mongo_client.server_info()  # Force connection check
    print(  mongo_client.server_info() )
    mongo_db = mongo_client["logoDB"]             # your DB name
    mongo_collection = mongo_db["logos"]          # your collection name
    print("✅ MongoDB Atlas connection established successfully.")
except pymongo.errors.ServerSelectionTimeoutError as err:
    print(f"❌ Failed to connect to MongoDB Atlas: {err}")

app = Flask(__name__)
CORS(app)

DATASET_DIR = "./dataset/Dataset_simplified"
os.makedirs(DATASET_DIR, exist_ok=True)

def load_and_encode(svg_path):
    try:
        svg = load_svg(svg_path)
        vector = encode_svg(svg)
        embedding_array = vector.flatten().numpy()
        if embedding_array.shape[0] != 256:
            raise ValueError(f"Embedding dimension is {embedding_array.shape[0]}, expected 256")
        return embedding_array.tolist(), True
    except Exception as e:
        print(f"DeepSVG encoding failed for {svg_path}: {e}")
        return None, False

# Helper: Join all paths in SVG into one Path object
def join_svg_paths(svg_file):
    paths, _ = svg2paths(svg_file)
    combined_path = Path()
    for path in paths:
        combined_path.extend(path)
    return combined_path


def parse_svg(svg_path, num_samples=250):
    path = join_svg_paths(svg_path)
    total_length = path.length()
    sample_distances = np.linspace(0, total_length, num_samples)
    sampled_points = []
    for distance in sample_distances:
        point = path.point(distance / total_length)
        sampled_points.append((point.real, point.imag))
    return np.array(sampled_points)

def load_and_encode_ab(svg_path):
    try:
        return parse_svg(svg_path)
    except Exception as e:
        print(f"Encoding failed for {svg_path}: {e}")
        return None

@app.route('/api/register-logo', methods=['POST'])
def register_logo():
    if 'logos' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    uploaded_files = request.files.getlist('logos')
    if not uploaded_files:
        return jsonify({'error': 'No files received'}), 400

    results = []

    for file in uploaded_files:
        if not file or file.filename == '' or not file.filename.endswith('.svg'):
            continue  # skip invalid file

        logo_id = str(uuid.uuid4())
        svg_filename = f"{logo_id}.svg"
        svg_path = os.path.join(DATASET_DIR, svg_filename)

        file.save(svg_path)

        embedding, is_deepsvg_successful = load_and_encode(svg_path)
        if not is_deepsvg_successful:
            print(f"DeepSVG failed for {file.filename}. Proceeding with fallback...")

        target_vector = load_and_encode_ab(svg_path)
        if target_vector is None:
            os.remove(svg_path)
            continue

        """begin:Similarity Check"""
        if is_deepsvg_successful:
            try:
                search_results = collection.search(
                    data=[embedding],
                    anns_field="vector",
                    param={"metric_type": "COSINE"},
                    limit=1,
                    output_fields=["milvus_id"]
                )[0]
                for hit in search_results:
                    if hit.distance >= 0.9:  # Cosine similarity threshold
                        os.remove(svg_path)
                        return jsonify({'message': 'Registration failed: Similar logo already exists.'}), 409
            except Exception as e:
                print(f"Milvus similarity check failed: {e}")
        else:


        # 2. Check against MongoDB (Procrustes)
            try:
                for doc in mongo_collection.find({}, {"parsed_coordinates": 1}):
                    if "parsed_coordinates" not in doc:
                        continue
                    score = compute_procrustes_similarity(target_vector, np.array(doc["parsed_coordinates"]))
                    if score >= 0.9:
                        os.remove(svg_path)
                        return jsonify({'message': 'Registration failed: Similar logo already exists (Procrustes similarity ≥ 0.9).'}), 409
            except Exception as e:
                print(f"Procrustes similarity check failed: {e}")


        """end:Similarity Check"""

        if embedding:
            mr = collection.insert([[embedding]])
            milvus_id = mr.primary_keys[0]
        else:
            milvus_id = None

        with open(svg_path, 'r', encoding='utf-8') as svg_file:
            svg_content = svg_file.read()

        mongo_record = {
            "logo_id": logo_id,
            "svg_content": svg_content,
            "milvus_id": milvus_id,
            "file_name": file.filename,
            "parsed_coordinates": target_vector.tolist(),
            "isDeepSVG": is_deepsvg_successful,
            "companyName": request.form.get('companyName'),
            "websiteURL": request.form.get('websiteURL'),
            "metadata": request.form.get('metadata')
        }

        mongo_collection.insert_one(mongo_record)
        os.remove(svg_path)

        results.append({
            "logo_id": logo_id,
            "milvus_id": milvus_id,
            "file_name": file.filename
        })

    if not results:
        return jsonify({'error': 'No valid SVG logos were processed.'}), 400
    return jsonify({
        "message": f"{len(results)} logo(s) registered successfully",
        "results": results
    }), 200




# Constants and directories
TEMP_DIR = "./temp"
os.makedirs(TEMP_DIR, exist_ok=True)



# Helper: Compute Procrustes similarity safely
def compute_procrustes_similarity(shape1, shape2):
    try:
        _, _, disparity = procrustes(shape1, shape2)
        return 1 / (1 + disparity)
    except Exception as e:
        print(f"Procrustes comparison failed: {e}")
        return 0

def z_score_normalize(scores):
    scores = np.array(scores)
    mean = np.mean(scores)
    std = np.std(scores)
    if std == 0:
        return np.zeros_like(scores)
    return (scores - mean) / std

def quality_weighted_fusion(deep_scores, alg_scores):
    deep_norm = z_score_normalize(deep_scores)
    alg_norm = z_score_normalize(alg_scores)
    deep_quality = 1.0 / (np.var(deep_norm) + 1e-6)
    alg_quality = 1.0 / (np.var(alg_norm) + 1e-6)
    total_quality = deep_quality + alg_quality
    deep_weight = deep_quality / total_quality
    alg_weight = alg_quality / total_quality
    fused_scores = deep_weight * deep_norm + alg_weight * alg_norm
    return fused_scores

# Combined lookup endpoint
@app.route('/api/lookup-logo', methods=['POST'])
def combined_lookup():
    print("\n=== New Request ===")
    print("[1/7] Received request for combined lookup")

    if 'logo' not in request.files:
        print("[ERROR] No file part in the request")
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['logo']
    # if file.filename.endswith('.png') or file.filename.endswith('.jpg'):
    #     return png_logo_lookup(file)
    if not file or file.filename == '' or not file.filename.endswith('.svg'):
        print(f"[ERROR] Invalid file: {file.filename if file else 'None'}")
        return jsonify({'error': 'Only SVG files are allowed'}), 400

    temp_id = str(uuid.uuid4())
    temp_path = os.path.join(TEMP_DIR, f"{temp_id}.svg")
    file.save(temp_path)
    print(f"[2/7] Saved temporary SVG: {temp_path}")

    deep_vector = None
    try:
        svg = load_svg(temp_path)
        print("[3/7] SVG loaded for DeepSVG encoding")
        deep_vector = encode_svg(svg).cpu().numpy().flatten()
        print(f"[3/7] DeepSVG vector extracted: {deep_vector.shape}")
    except Exception as e:
        print(f"[WARNING] DeepSVG encoding failed: {e}")

    alg_vector = None
    try:
        alg_vector = parse_svg(temp_path)
        print(f"[3/7] Algorithm vector parsed: {len(alg_vector)} points")
    except Exception as e:
        os.remove(temp_path)
        print(f"[ERROR] Algorithmic parsing failed: {e}")
        return jsonify({'error': 'SVG parsing failed'}), 500



    deep_mongo_docs = []
    if deep_vector is not None:
        print("[4/7] Searching Milvus...")
        try:
            deep_results = collection.search(
                data=[deep_vector],
                anns_field="vector",
                param={"metric_type": "COSINE"},
                limit=10,
                output_fields=["milvus_id"]
            )[0]
            print(f"[4/7] Milvus returned {len(deep_results)} results")
            # Filter DeepSVG results with similarity >= 0.5
            deep_check_results = [hit for hit in deep_results if hit.distance >= 0.5]
            if not deep_check_results:
                print("[INFO] No DeepSVG results with similarity >= 0.5")
            else:
                for hit in deep_results:
                    doc = mongo_collection.find_one({"milvus_id": hit.id})
                    if doc:
                        deep_mongo_docs.append({
                            "_id": str(doc["_id"]),
                            "score": float(hit.distance),
                            "doc": doc
                        })
                        print(f"[4/7] DeepSVG match: ID={doc['_id']}, Score={hit.distance:.4f}")
                    else:
                        print(f"[WARNING] No MongoDB doc for Milvus ID={hit.id}")
        except Exception as e:
            print(f"[ERROR] Milvus search failed: {e}")
    else:
        print("[4/7] Skipping DeepSVG search due to previous failure")


    # Algorithm search
    print("[5/7] Running algorithm-based search...")
    alg_matches = []
    for doc in mongo_collection.find({}, {"parsed_coordinates": 1, "svg_content": 1}):
        if "parsed_coordinates" not in doc or not doc["parsed_coordinates"]:
            print(f"[WARNING] Skipping doc {doc.get('_id')} - missing or empty 'parsed_coordinates'")
            continue
        score = compute_procrustes_similarity(alg_vector, np.array(doc["parsed_coordinates"]))
        alg_matches.append({
            "_id": str(doc["_id"]),
            "score": float(score),
            "doc": doc
        })
    
    alg_top_matches = sorted(alg_matches, key=lambda x: (-x["score"], x["_id"]))[:10]
    print(f"[5/7] Algorithm top scores: {[m['score'] for m in alg_top_matches[:10]]}...")

    # Sort both lists for stable comparison and selection
    """
    deep_mongo_docs_sorted = sorted(deep_mongo_docs, key=lambda x: (-x["score"], x["_id"]))
    alg_top_matches_sorted = sorted(alg_top_matches, key=lambda x: (-x["score"], x["_id"]))

    if not deep_mongo_docs:
        print("[6/7] No DeepSVG results: using top 5 Algorithm matches only")
        selected = alg_top_matches_sorted[:5]
    else:
        deep_ids = [doc["_id"] for doc in deep_mongo_docs_sorted]
        alg_ids = [doc["_id"] for doc in alg_top_matches_sorted]
        print(f"[6/7] ID Comparison - DeepSVG: {deep_ids[:5]}..., Algorithm: {alg_ids[:5]}...")

        if deep_ids[:5] == alg_ids[:5]:
            print("[6/7] Match: Using full DeepSVG results")
            selected = deep_mongo_docs_sorted[:5]
        else:
            print("[6/7] No match: Combining top 3 DeepSVG + 2 unique Algorithm results")
            selected = deep_mongo_docs_sorted[:3]
            deep_top3_ids = set(doc["_id"] for doc in selected)
            unique_alg = [doc for doc in alg_top_matches_sorted if doc["_id"] not in deep_top3_ids]
            selected.extend(unique_alg[:2])
"""

    # Merge all unique IDs from both methods
    all_ids = set([doc["_id"] for doc in deep_mongo_docs] + [doc["_id"] for doc in alg_matches])

    fusion_candidates = []
    for doc_id in all_ids:
        deep_score = next((d["score"] for d in deep_mongo_docs if d["_id"] == doc_id), 0)
        alg_score = next((a["score"] for a in alg_matches if a["_id"] == doc_id), 0)
        doc = next((d["doc"] for d in deep_mongo_docs if d["_id"] == doc_id), None)
        if not doc:
            doc = next((a["doc"] for a in alg_matches if a["_id"] == doc_id), None)
        fusion_candidates.append({
            "_id": doc_id,
            "deep_score": deep_score,
            "alg_score": alg_score,
            "doc": doc
        })

    deep_scores = [c["deep_score"] for c in fusion_candidates]
    alg_scores = [c["alg_score"] for c in fusion_candidates]
    fused_scores = quality_weighted_fusion(deep_scores, alg_scores)

    for i, candidate in enumerate(fusion_candidates):
        candidate["fused_score"] = float(fused_scores[i])

    selected = sorted(fusion_candidates, key=lambda x: -x["fused_score"])[:5]

    print(f"[7/7] Returning {len(selected)} results")

    # PNG rendering
    print("[7/7] Rendering PNGs...")
    results = []
    for item in selected:
        doc = item["doc"]
        mongo_id = str(doc["_id"])

        try:
            png_data = cairosvg.svg2png(bytestring=doc["svg_content"].encode('utf-8'))
            b64_png = base64.b64encode(png_data).decode('utf-8')
            print(f"[7/7] Rendered base64 PNG for {mongo_id}")
        except Exception as e:
            print(f"[ERROR] Failed to render PNG for {mongo_id}: {str(e)}")
            continue

        company_name = doc["companyName"] if "companyName" in doc else "Unknown Company"
        company_url = doc["websiteURL"] if "websiteURL" in doc else f"https://example.com/brand/{mongo_id}"

        results.append({
            "logoUrl": f"data:image/png;base64,{b64_png}",
            # "companyUrl": f"https://example.com/brand/{mongo_id}",
            "companyName": company_name,
            "companyUrl": company_url,
            "score": round(item.get("fused_score", item.get("score", 0)), 4)
        })

    print(f"[7/7] Returning {len(results)} results")

    # CLEANUP: Remove the temp SVG file
    try:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"[CLEANUP] Deleted temporary file: {temp_path}")
    except Exception as e:
        print(f"[CLEANUP ERROR] Failed to delete temp SVG: {e}")

    
    return jsonify({"matches": results})


# #########################


# # DATASET_DIR = "./dataset/Registered_Dataset_simplified"
# # PNG_OUTPUT_DIR = "./dataset/rendered_pngs"

# # os.makedirs(PNG_OUTPUT_DIR, exist_ok=True)

# # ---- Helper Functions ---- #


# # ---- Handles Vectorization ---- #
# import vtracer

# def convert_png_to_svg(input_path, output_path):
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     if os.path.isfile(input_path):
#         print(f"Processing {input_path}...")
#         vtracer.convert_image_to_svg_py(
#             input_path, output_path,
#             colormode='color',
#             hierarchical='stacked',
#             mode='spline',
#             filter_speckle=4,
#             color_precision=6,
#             layer_difference=16,
#             corner_threshold=60,
#             length_threshold=4.0,
#             max_iterations=10,
#             splice_threshold=45,
#             path_precision=3
#         )
#         print(f"Saved {output_path}")

# # ---- Handles Preprocessing of SVGs--- # 
# import re
# from xml.dom import minidom
# from svgpathtools import parse_path

# def apply_translate_to_path(path: Path, dx: float, dy: float) -> Path:
#     translated = Path()
#     for segment in path:
#         seg = segment.translated(complex(dx, dy))
#         translated.append(seg)
#     return translated

# def parse_translate(transform: str):
#     match = re.search(r"translate\(([^,]+),\s*([^)]+)\)", transform)
#     if match:
#         dx = float(match.group(1))
#         dy = float(match.group(2))
#         return dx, dy
#     return 0.0, 0.0

# # Applies only translate transforms for now and writes output to a given file
# def apply_transforms_to_svg_paths(input_svg_path: str, output_svg_path: str):
#     doc = minidom.parse(input_svg_path)
#     path_tags = doc.getElementsByTagName('path')

#     for node in path_tags:
#         d_attr = node.getAttribute('d')
#         transform_attr = node.getAttribute('transform')
#         path = parse_path(d_attr)

#         if transform_attr:
#             dx, dy = parse_translate(transform_attr)
#             path = apply_translate_to_path(path, dx, dy)
#             node.setAttribute('d', path.d())
#             node.removeAttribute('transform')

#     with open(output_svg_path, 'w') as f:
#         doc.writexml(f)
#     doc.unlink()

# # Main processing pipeline that overwrites the given output file
# def process_vectorized_svg(input_path: str, output_path: str):
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     if os.path.isfile(input_path) and input_path.lower().endswith(".svg"):
#         print(f"Processing {input_path}...")

#         apply_transforms_to_svg_paths(input_path, output_path)

#         # Future transformations (placeholders):
#         # remove_overlaps(output_path)
#         # remove_border_path(output_path)
#         # remove_insignificant_paths(output_path)
#         # apply_deepSVG_cleaning(output_path)

#         print(f"Saved processed SVG to {output_path}")

# # ---- Handles SVG to Point Cloud conversion---- #
# def order_points(points):
#     center = np.mean(points, axis=0)
#     angles = np.arctan2(points[:,1] - center[1], points[:,0] - center[0])
#     return points[np.argsort(angles)]

# def join_svg_paths(svg_file):                # --------------SAME AS BEFORE--------------
#     paths, _ = svg2paths(svg_file)
#     combined_path = Path()
#     for path in paths:
#         combined_path.extend(path)
#     return combined_path

# def parse_svg(svg_path, num_samples=250):
#     path = join_svg_paths(svg_path)
#     total_length = path.length()
#     sample_distances = np.linspace(0, total_length, num_samples)
#     sampled_points = []
#     for distance in sample_distances:
#         point = path.point(distance / total_length)
#         sampled_points.append((point.real, point.imag))
#     points = order_points(np.array(sampled_points))     # --------------SMALL CHANGE - ordering--------------               
#     return points

# def load_and_encode(svg_path): # --------------MAYBE SAME AS BEFORE--------------
#     try:
#         return parse_svg(svg_path)
#     except Exception as e:
#         print(f"Encoding failed for {svg_path}: {e}")
#         return None
    
# # ---- Handles Similarity Computations ---- #    

# # Procrustes analysis
# def compute_procrustes_similarity(shape1, shape2):   # --------------SAME AS BEFORE--------------
#     try:
#         _, _, disparity = procrustes(shape1, shape2)
#         return 1 / (1 + disparity)
#     except Exception as e:
#         print(f"Procrustes comparison failed: {e}")
#         return 0
    
# # Hausdorff distance
# def center_shape(points):
#     centroid = np.mean(points, axis=0)
#     return points - centroid

# def scale_to_unit_size(points):
#     min_coords = np.min(points, axis=0)
#     max_coords = np.max(points, axis=0)
#     width, height = max_coords - min_coords
#     scale_factor = 1 / max(width, height)
#     return points * scale_factor

# def align_orientation(points):
#     pca = PCA(n_components=2)
#     pca.fit(points)
#     principal_axis = pca.components_[0]
#     angle = np.arctan2(principal_axis[1], principal_axis[0])
#     rotation_matrix = np.array([
#         [np.cos(-angle), -np.sin(-angle)],
#         [np.sin(-angle), np.cos(-angle)]
#     ])
#     return points @ rotation_matrix.T

# def normalize_shape(points):
#     points = center_shape(points)
#     points = align_orientation(points) 
#     points = scale_to_unit_size(points)
#     return points

# from scipy.spatial.distance import directed_hausdorff
# def compute_hausdorff_similarity(shape1, shape2):
#     shape1 = normalize_shape(shape1)
#     shape2 = normalize_shape(shape2)

#     forward_distance = directed_hausdorff(shape1, shape2)[0]
#     reverse_distance = directed_hausdorff(shape2, shape1)[0]
#     hausdorff_distance = max(forward_distance, reverse_distance)
#     similarity = 1 / (1 + hausdorff_distance)
#     return similarity


# def png_logo_lookup(file):
#     print("\n=== New PNG Logo Lookup Request ===")
#     print(f"[1/6] Received file: {file.filename}")

#     temp_id = str(uuid.uuid4())
#     temp_png_path = f"./temp/{temp_id}.png"
#     temp_svg_raw_path = f"./temp/{temp_id}_raw.svg"
#     temp_svg_path = f"./temp/{temp_id}.svg"
#     os.makedirs("./temp", exist_ok=True)
#     file.save(temp_png_path)
#     print(f"[2/6] Saved temporary PNG: {temp_png_path}")

#     try:
#         # Step 3: PNG to SVG conversion
#         try:
#             print("[3/6] Converting PNG to SVG...")
#             convert_png_to_svg(temp_png_path, temp_svg_raw_path)
#             print(f"[3/6] Raw SVG saved to: {temp_svg_raw_path}")
#         except Exception as e:
#             print(f"[ERROR] PNG to SVG conversion failed: {e}")
#             return jsonify({'error': f'PNG to SVG conversion failed: {str(e)}'}), 500

#         # Step 4: Process vectorized SVG
#         try:
#             print("[4/6] Processing vectorized SVG...")
#             process_vectorized_svg(temp_svg_raw_path, temp_svg_path)
#             print(f"[4/6] Processed SVG saved to: {temp_svg_path}")
#         except Exception as e:
#             print(f"[ERROR] SVG processing failed: {e}")
#             return jsonify({'error': f'SVG processing failed: {str(e)}'}), 500

#         # Step 5: Load and encode
#         print("[5/6] Loading and encoding SVG...")
#         target_vector = load_and_encode(temp_svg_path)
#         if target_vector is None:
#             print("[ERROR] Failed to encode SVG.")
#             return jsonify({'error': 'SVG encoding failed'}), 500
#         print("[5/6] SVG encoding completed.")

#         # Step 6: Similarity check
#         print("[6/6] Computing similarities against dataset...")
#         similarities = []
#         for fname in os.listdir(DATASET_DIR):
#             if not fname.endswith('.svg'):
#                 continue
#             candidate_path = os.path.join(DATASET_DIR, fname)
#             candidate_vector = load_and_encode(candidate_path)
#             if candidate_vector is None:
#                 print(f"[WARNING] Skipping {fname} - failed to encode.")
#                 continue

#             score = compute_hausdorff_similarity(target_vector, candidate_vector)
#             similarities.append((fname, score, candidate_path))

#         similarities.sort(key=lambda x: x[1], reverse=True)
#         top_matches = similarities[:3]
#         print(f"[6/6] Top scores: {[round(s[1], 4) for s in top_matches]}")

#         results = []
#         for fname, score, path in top_matches:
#             png_name = fname.replace('.svg', '.png')
#             png_path = os.path.join(PNG_OUTPUT_DIR, png_name)
#             if not os.path.exists(png_path):
#                 try:
#                     cairosvg.svg2png(file_obj=open(path, "rb"), write_to=png_path)
#                     print(f"[INFO] Rendered PNG: {png_name}")
#                 except Exception as e:
#                     print(f"[ERROR] Failed to render PNG for {fname}: {e}")
#                     continue

#             results.append({
#                 "logoUrl": f"http://localhost:5000/static/{png_name}",
#                 "companyUrl": f"https://example.com/brand/{fname.replace('.svg','')}",
#                 "score": round(score, 4)
#             })

#         print(f"[COMPLETE] Returning {len(results)} match(es) for {file.filename}")
#         return jsonify({'matches': results})

#     finally:
#         # Cleanup temp files
#         print("[CLEANUP] Deleting temporary files...")
#         for path in [temp_png_path, temp_svg_raw_path, temp_svg_path]:
#             if os.path.exists(path):
#                 try:
#                     os.remove(path)
#                     print(f"[CLEANUP] Deleted: {path}")
#                 except Exception as e:
#                     print(f"[CLEANUP ERROR] Failed to delete {path}: {e}")
#         print("[CLEANUP] Temporary files cleaned up.")

# ##########################


# Run only once
if __name__ == '__main__':
    app.run(port=5000, debug=True, use_reloader=False)