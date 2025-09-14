def normalize_doc(doc: dict) -> dict:
    doc["_id"] = str(doc["_id"])  # Chuyển ObjectId thành chuỗi
    return doc
