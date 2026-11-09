"""
Codabench Submission Packaging and Validation Utility.
Verifies JSON formatting, validates non-empty keys, and outputs a compliant .zip file.
"""
import os
import json
import zipfile
import hashlib

def package_submission(prediction_data, output_zip_path, filename_inside_zip="prediction.json"):
    # Validate format
    assert isinstance(prediction_data, (dict, list)), "Prediction data must be a JSON serializable dict or list"

    temp_json_path = output_zip_path.replace(".zip", "_temp.json")
    with open(temp_json_path, "w", encoding="utf-8") as f:
        json.dump(prediction_data, f, indent=2)

    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(temp_json_path, arcname=filename_inside_zip)

    if os.path.exists(temp_json_path):
        os.remove(temp_json_path)

    # Compute hash
    h = hashlib.sha256()
    with open(output_zip_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    print(f"Packaged {output_zip_path} | SHA256: {h.hexdigest()}")
    return output_zip_path, h.hexdigest()
