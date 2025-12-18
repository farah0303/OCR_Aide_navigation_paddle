"""Lightweight HTTP API for OCR extraction.

Exposes:
- GET /api/ocr/extract   → health check
- POST /api/ocr/extract  → accept a single file upload and run OCR
"""

import os
import tempfile
import time
import datetime
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

# Reuse the core OCR processing utilities
from main import AI_CORRECTION_AVAILABLE, process_file

try:
    from advanced_correction.advanced_correction import corriger_texte_ai
except Exception:
    corriger_texte_ai = None

try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None

app = Flask(__name__)

_mongo_client: Optional[MongoClient] = None
_mongo_collection = None

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.environ.get("MONGO_DB", "ocr_database")
MONGO_COLLECTION = os.environ.get("MONGO_COLLECTION", "fichiers_drive")


@app.after_request
def add_cors_headers(response):
    """Simple CORS headers for dev usage."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def get_mongo_collection():
    """Lazily create Mongo collection handle if configuration is present."""
    global _mongo_client, _mongo_collection
    if _mongo_collection:
        return _mongo_collection
    if not MONGO_URI or not MongoClient:
        return None
    _mongo_client = MongoClient(MONGO_URI)
    _mongo_collection = _mongo_client[MONGO_DB][MONGO_COLLECTION]
    return _mongo_collection


def parse_bool(value: Any, default: bool) -> bool:
    """Parse flexible boolean inputs from strings or native bools."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def parse_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


@app.route("/api/ocr/extract", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok"}), 200


@app.route("/api/ocr/extract", methods=["POST"])
def extract_ocr():
    files = request.files.getlist("file")
    if not files:
        return jsonify({"error": "file is required (multipart/form-data with 'file' field)"}), 400
    if len(files) != 1:
        return jsonify({"error": "only one file allowed"}), 400

    upload = files[0]
    if upload.filename is None or upload.filename.strip() == "":
        return jsonify({"error": "file name is empty"}), 400

    filename = secure_filename(upload.filename)
    temp_dir = tempfile.mkdtemp(prefix="ocr_api_")
    temp_path = os.path.join(temp_dir, filename)

    # Extract options from form fields (works with multipart/form-data)
    opts: Dict[str, Any] = request.form.to_dict(flat=True)
    zoom = parse_float(opts.get("zoom", 2.0), 2.0)
    use_angle = parse_bool(opts.get("use_angle", True), True)
    use_gpu = parse_bool(opts.get("use_gpu", False), False)
    apply_ai = parse_bool(opts.get("apply_ai_correction", False), False)

    try:
        upload.save(temp_path)
        t0 = time.perf_counter()
        text, details = process_file(temp_path, use_angle=use_angle, zoom=zoom, use_gpu=use_gpu)

        if apply_ai and AI_CORRECTION_AVAILABLE and corriger_texte_ai:
            try:
                text = corriger_texte_ai(text)
            except Exception as e:
                return jsonify({"error": f"AI correction failed: {e}"}), 500
        elif apply_ai and not AI_CORRECTION_AVAILABLE:
            return jsonify({"error": "AI correction not available on this server"}), 400

        elapsed = time.perf_counter() - t0

        # Attempt to log processing in MongoDB if configured
        mongo_status = {"status": "skipped"}
        collection = get_mongo_collection()
        if collection is not None:
            try:
                file_size = os.path.getsize(temp_path)
            except Exception:
                file_size = None

            doc = {
                "nom_fichier": upload.filename,
                "chemin_local": temp_path,
                "mime_type": upload.mimetype or "application/octet-stream",
                "taille": file_size,
                "cree_le": datetime.datetime.utcnow(),
                "modifie_le": datetime.datetime.utcnow(),
                "texte_ocr": text,
            }
            try:
                result = collection.insert_one(doc)
                mongo_status = {"status": "ok", "inserted_id": str(result.inserted_id)}
            except Exception as e:
                mongo_status = {"status": "error", "error": str(e)}
        elif MONGO_URI and not MongoClient:
            mongo_status = {"status": "error", "error": "pymongo not installed"}

        return jsonify({
            "status": "ok",
            "filename": upload.filename,
            "seconds": round(elapsed, 3),
            "params": {
                "zoom": zoom,
                "use_angle": use_angle,
                "use_gpu": use_gpu,
                "apply_ai_correction": apply_ai,
            },
            "details": details,
            "text": text,
            "mongo": mongo_status,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.isdir(temp_dir):
                os.rmdir(temp_dir)
        except Exception:
            pass


if __name__ == "__main__":
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "5000"))
    app.run(host=host, port=port)
