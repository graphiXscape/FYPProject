import os
import numpy as np
import re
import tempfile
from xml.etree import ElementTree as ET
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from PIL import Image as ImagePil
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
from svgpathtools import svg2paths, Path, parse_path
from xml.dom import minidom
import vtracer
import torch
import cairosvg

# DeepSVG imports (assume available in sys.path)
from deepsvg.svglib.svg import SVG
from deepsvg import utils as deepsvg_utils
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svgtensor_dataset import SVGTensorDataset, load_dataset
from deepsvg.utils.utils import batchify
from configs.deepsvg.hierarchical_ordered import Config

# Device and model setup (singleton pattern)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cfg = Config()
model = cfg.make_model().to(device)
pretrained_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../pretrained/hierarchical_ordered.pth.tar")
deep_dataset = load_dataset(cfg)
deep_model_loaded = False
try:
    deepsvg_utils.load_model(pretrained_path, model)
    model.eval()
    deep_model_loaded = True
except Exception as e:
    print(f"[DeepSVG] Model load failed: {e}")

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
    data = deep_dataset.get(svg=svg)
    return encode(data)

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

def join_svg_paths(svg_file):
    paths, _ = svg2paths(svg_file)
    combined_path = Path()
    for path in paths:
        combined_path.extend(path)
    return combined_path

def order_points(points):
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:,1] - center[1], points[:,0] - center[0])
    return points[np.argsort(angles)]

def parse_svg(svg_path, num_samples=250):
    path = join_svg_paths(svg_path)
    total_length = path.length()
    sample_distances = np.linspace(0, total_length, num_samples)
    sampled_points = []
    for distance in sample_distances:
        point = path.point(distance / total_length)
        sampled_points.append((point.real, point.imag))
    points = order_points(np.array(sampled_points))
    return points

def load_and_encode_ab(svg_path):
    try:
        return parse_svg(svg_path)
    except Exception as e:
        print(f"Encoding failed for {svg_path}: {e}")
        return None

def compute_procrustes_similarity(shape1, shape2):
    try:
        _, _, disparity = procrustes(np.array(shape1), np.array(shape2))
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

def center_shape(points):
    centroid = np.mean(points, axis=0)
    return points - centroid

def scale_to_unit_size(points):
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    width, height = max_coords - min_coords
    scale_factor = 1 / (max(width, height) if max(width, height) != 0 else 1)
    return points * scale_factor

def align_orientation(points):
    pca = PCA(n_components=2)
    pca.fit(points)
    principal_axis = pca.components_[0]
    angle = np.arctan2(principal_axis[1], principal_axis[0])
    rotation_matrix = np.array([
        [np.cos(-angle), -np.sin(-angle)],
        [np.sin(-angle), np.cos(-angle)]
    ])
    return points @ rotation_matrix.T

def normalize_shape(points):
    points = center_shape(points)
    points = align_orientation(points)
    points = scale_to_unit_size(points)
    return points

def compute_hausdorff_similarity(shape1, shape2):
    from scipy.spatial.distance import directed_hausdorff
    shape1 = normalize_shape(shape1)
    shape2 = normalize_shape(shape2)
    forward_distance = directed_hausdorff(shape1, shape2)[0]
    reverse_distance = directed_hausdorff(shape2, shape1)[0]
    hausdorff_distance = max(forward_distance, reverse_distance)
    similarity = 1 / (1 + hausdorff_distance)
    return similarity

def hex_to_rgb(hex_color):
    hex_color = hex_color.strip().lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def extract_colors(svg_path):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    fills = set()
    for elem in root.iter():
        for attr in ['fill', 'stroke']:
            color = elem.attrib.get(attr)
            if color and re.match(r'^#?[0-9a-fA-F]{3,6}$', color):
                fills.add(color.lower())
    return fills

def color_difference(hex1, hex2):
    rgb1 = sRGBColor(*hex_to_rgb(hex1), is_upscaled=True)
    rgb2 = sRGBColor(*hex_to_rgb(hex2), is_upscaled=True)
    lab1 = convert_color(rgb1, LabColor)
    lab2 = convert_color(rgb2, LabColor)
    return delta_e_cie2000(lab1, lab2)

def compare_svg_colors(svg1, svg2, threshold=2.3):
    colors1 = extract_colors(svg1)
    colors2 = extract_colors(svg2)
    total = len(colors1)
    matched = 0
    for color1 in colors1:
        for color2 in colors2:
            diff = color_difference(color1, color2)
            if diff <= threshold:
                matched += 1
                break
    if total == 0:
        return 0.0
    similarity_score = matched / total
    return similarity_score

def convert_png_to_svg(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.isfile(input_path):
        vtracer.convert_image_to_svg_py(
            input_path, output_path,
            colormode='color',
            hierarchical='stacked',
            mode='spline',
            filter_speckle=4,
            color_precision=6,
            layer_difference=16,
            corner_threshold=60,
            length_threshold=4.0,
            max_iterations=10,
            splice_threshold=45,
            path_precision=3
        )

def apply_translate_to_path(path: Path, dx: float, dy: float) -> Path:
    translated = Path()
    for segment in path:
        seg = segment.translated(complex(dx, dy))
        translated.append(seg)
    return translated

def parse_translate(transform: str):
    match = re.search(r"translate\\(([^,]+),\\s*([^)]+)\\)", transform)
    if match:
        dx = float(match.group(1))
        dy = float(match.group(2))
        return dx, dy
    return 0.0, 0.0

def apply_transforms_to_svg_paths(input_svg_path: str, output_svg_path: str):
    doc = minidom.parse(input_svg_path)
    path_tags = doc.getElementsByTagName('path')
    for node in path_tags:
        d_attr = node.getAttribute('d')
        transform_attr = node.getAttribute('transform')
        path = parse_path(d_attr)
        if transform_attr:
            dx, dy = parse_translate(transform_attr)
            path = apply_translate_to_path(path, dx, dy)
            node.setAttribute('d', path.d())
            node.removeAttribute('transform')
    with open(output_svg_path, 'w') as f:
        doc.writexml(f)
    doc.unlink()

def process_vectorized_svg(input_path: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.isfile(input_path) and input_path.lower().endswith(".svg"):
        apply_transforms_to_svg_paths(input_path, output_path)

# PNG logo lookup logic

def png_logo_lookup(file, mongo_collection):
    import uuid
    temp_id = str(uuid.uuid4())
    temp_png_path = f"./temp/{temp_id}.png"
    temp_svg_raw_path = f"./temp/{temp_id}_raw.svg"
    temp_svg_path = f"./temp/{temp_id}.svg"
    os.makedirs("./temp", exist_ok=True)
    file.save(temp_png_path)
    try:
        convert_png_to_svg(temp_png_path, temp_svg_raw_path)
        process_vectorized_svg(temp_svg_raw_path, temp_svg_path)
        alg_vector = parse_svg(temp_svg_path)
        alg_matches = []
        for doc in mongo_collection.find({}, {"parsed_coordinates": 1, "svg_content": 1, "companyName": 1, "websiteURL": 1}):
            if "parsed_coordinates" not in doc or not doc["parsed_coordinates"]:
                continue
            score = compute_hausdorff_similarity(alg_vector, np.array(doc["parsed_coordinates"]))
            alg_matches.append({
                "_id": str(doc["_id"]),
                "score": float(score),
                "doc": doc
            })
        alg_top_matches = sorted(alg_matches, key=lambda x: (-x["score"], x["_id"]))[:5]
        results = []
        for item in alg_top_matches:
            doc = item["doc"]
            mongo_id = str(doc["_id"])
            try:
                png_data = cairosvg.svg2png(bytestring=doc["svg_content"].encode('utf-8'))
                b64_png = base64.b64encode(png_data).decode('utf-8')
            except Exception as e:
                continue
            company_name = doc.get("companyName", "Unknown Company")
            company_url = doc.get("websiteURL", f"https://example.com/brand/{mongo_id}")
            results.append({
                "logoUrl": f"data:image/png;base64,{b64_png}",
                "companyName": company_name,
                "companyUrl": company_url,
                "score": round(item["score"], 4)
            })
        return {"matches": results}
    finally:
        for path in [temp_png_path, temp_svg_raw_path, temp_svg_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass 