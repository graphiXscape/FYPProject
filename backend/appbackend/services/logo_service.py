import os
import uuid
import base64
import tempfile
from flask import jsonify
from appbackend.database.mongo import get_mongo_collection
from appbackend.database.milvus import get_milvus_collection
from appbackend.config import Config
from appbackend.utils.svg_utils import (
    load_svg, encode_svg, parse_svg, load_and_encode, load_and_encode_ab,
    compute_procrustes_similarity, quality_weighted_fusion, compare_svg_colors,
    png_logo_lookup
)

mongo_collection = get_mongo_collection()
collection = get_milvus_collection()

def register_logo_service(request):
    if 'logos' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    uploaded_files = request.files.getlist('logos')
    if not uploaded_files:
        return jsonify({'error': 'No files received'}), 400

    results = []

    for file in uploaded_files:
        if not file or file.filename == '' or not file.filename.endswith('.svg'):
            continue

        logo_id = str(uuid.uuid4())
        svg_filename = f"{logo_id}.svg"
        svg_path = os.path.join(Config.DATASET_DIR, svg_filename)
        file.save(svg_path)

        embedding, is_deepsvg_successful = load_and_encode(svg_path)
        if not is_deepsvg_successful:
            print(f"DeepSVG failed for {file.filename}. Proceeding with fallback...")

        target_vector = load_and_encode_ab(svg_path)
        if target_vector is None:
            os.remove(svg_path)
            continue

        # Similarity Check
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
                    if hit.distance >= 0.9:
                        os.remove(svg_path)
                        return jsonify({'message': 'Registration failed: Similar logo already exists.'}), 409
            except Exception as e:
                print(f"Milvus similarity check failed: {e}")
        else:
            try:
                for doc in mongo_collection.find({}, {"parsed_coordinates": 1}):
                    if "parsed_coordinates" not in doc:
                        continue
                    score = compute_procrustes_similarity(target_vector, doc["parsed_coordinates"])
                    if score >= 0.9:
                        os.remove(svg_path)
                        return jsonify({'message': 'Registration failed: Similar logo already exists (Procrustes similarity ≥ 0.9).'}), 409
            except Exception as e:
                print(f"Procrustes similarity check failed: {e}")

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

def lookup_logo_service(request):
    if 'logo' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['logo']
    if file.filename.endswith('.png') or file.filename.endswith('.jpg'):
        return png_logo_lookup(file, mongo_collection)
    if not file or file.filename == '' or not file.filename.endswith('.svg'):
        return jsonify({'error': 'Only SVG files are allowed'}), 400

    temp_id = str(uuid.uuid4())
    temp_path = os.path.join(Config.TEMP_DIR, f"{temp_id}.svg")
    file.save(temp_path)

    deep_vector = None
    try:
        svg = load_svg(temp_path)
        deep_vector = encode_svg(svg).cpu().numpy().flatten()
    except Exception as e:
        print(f"[WARNING] DeepSVG encoding failed: {e}")

    alg_vector = None
    try:
        alg_vector = parse_svg(temp_path)
    except Exception as e:
        os.remove(temp_path)
        print(f"[ERROR] Algorithmic parsing failed: {e}")
        return jsonify({'error': 'SVG parsing failed'}), 500

    deep_mongo_docs = []
    if deep_vector is not None:
        try:
            deep_results = collection.search(
                data=[deep_vector],
                anns_field="vector",
                param={"metric_type": "COSINE"},
                limit=10,
                output_fields=["milvus_id"]
            )[0]
            deep_check_results = [hit for hit in deep_results if hit.distance >= 0.5]
            if deep_check_results:
                for hit in deep_results:
                    doc = mongo_collection.find_one({"milvus_id": hit.id})
                    if doc:
                        deep_mongo_docs.append({
                            "_id": str(doc["_id"]),
                            "score": float(hit.distance),
                            "doc": doc
                        })
        except Exception as e:
            print(f"[ERROR] Milvus search failed: {e}")

    alg_matches = []
    for doc in mongo_collection.find({}, {"parsed_coordinates": 1, "svg_content": 1}):
        if "parsed_coordinates" not in doc or not doc["parsed_coordinates"]:
            continue
        score = compute_procrustes_similarity(alg_vector, doc["parsed_coordinates"])
        alg_matches.append({
            "_id": str(doc["_id"]),
            "score": float(score),
            "doc": doc
        })
    alg_top_matches = sorted(alg_matches, key=lambda x: (-x["score"], x["_id"]))[:10]

    all_ids = set([doc["_id"] for doc in deep_mongo_docs] + [doc["_id"] for doc in alg_top_matches])
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
    fused_2d_scores = quality_weighted_fusion(deep_scores, alg_scores)
    COLOR_WEIGHT = 0.2
    COLOR_THRESHOLD = 0.7

    for i, candidate in enumerate(fusion_candidates):
        doc_svg_content = candidate["doc"]["svg_content"]
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as temp_svg_file:
            temp_svg_file.write(doc_svg_content.encode('utf-8'))
            temp_svg_file_path = temp_svg_file.name
        try:
            color_score = compare_svg_colors(temp_path, temp_svg_file_path)
        except Exception as e:
            print(f"[COLOR COMPARE ERROR] Failed for {candidate['_id']}: {e}")
            color_score = 0
        finally:
            if os.path.exists(temp_svg_file_path):
                os.remove(temp_svg_file_path)
        candidate["color_score"] = color_score
        if color_score >= COLOR_THRESHOLD:
            final_score = fused_2d_scores[i] + COLOR_WEIGHT * (color_score - COLOR_THRESHOLD)
        else:
            final_score = fused_2d_scores[i]
        candidate["fused_score"] = float(final_score)

    selected = sorted(fusion_candidates, key=lambda x: -x["fused_score"])[:5]
    results = []
    for item in selected:
        doc = item["doc"]
        mongo_id = str(doc["_id"])
        try:
            import cairosvg
            png_data = cairosvg.svg2png(bytestring=doc["svg_content"].encode('utf-8'))
            b64_png = base64.b64encode(png_data).decode('utf-8')
        except Exception as e:
            print(f"[ERROR] Failed to render PNG for {mongo_id}: {str(e)}")
            continue
        company_name = doc.get("companyName", "Unknown Company")
        company_url = doc.get("websiteURL", f"https://example.com/brand/{mongo_id}")
        results.append({
            "logoUrl": f"data:image/png;base64,{b64_png}",
            "companyName": company_name,
            "companyUrl": company_url,
            "score": round(item.get("fused_score", item.get("score", 0)), 4)
        })
    try:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        print(f"[CLEANUP ERROR] Failed to delete temp SVG: {e}")
    return jsonify({"matches": results})
