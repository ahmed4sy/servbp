from flask import Flask, request, jsonify
import cv2
import numpy as np
import easyocr
import os
import uuid

app = Flask(__name__)
VALID_TOKEN = "H2j3Lk9QpZ"


@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided", "status": 400})
    token = request.headers.get("Authorization")
    if not token or token != VALID_TOKEN:
        return jsonify({"error": "Invalid or missing token", "status": 401})

    image = request.files["image"]
    unique_filename = f"{uuid.uuid4()}.jpg"
    image.save(unique_filename)
    if (int)(get_file_size(unique_filename) / 1024) > 5000:
        os.remove(unique_filename)
        return jsonify({"message": "file is too large", "status": 400})
    img = cv2.imread(unique_filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = powerfulAnalysisOCR(img)
    res = convert_to_native_types(res)
    os.remove(unique_filename)
    return jsonify({"message": "successfully processed", "status": 200, "data": res})


def get_file_size(file_path):
    file_size = os.path.getsize(file_path)
    return file_size


def convert_to_native_types(data):
    if isinstance(data, dict):
        return {k: convert_to_native_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_native_types(i) for i in data]
    elif isinstance(data, np.int64):
        return int(data)
    elif isinstance(data, np.float64):
        return float(data)
    else:
        return data


def powerfulAnalysisOCR(gray):
    rdr = easyocr.Reader(["en"])
    imgslist = []
    res = rdr.readtext(gray)
    print(len(res))
    for item in res:
        x = item[0][0][0]
        y = item[0][0][1]
        w = item[0][2][0] - item[0][0][0]
        h = item[0][2][1] - item[0][0][1]
        imgslist.append([x - 5, y - 5, w + 5, h + 5])
    marged = merge_bubbles(imgslist)
    resulttemp = [[box[0], box[1], box[2], box[3]] for box in marged]
    return resulttemp


def merge_bubbles(rectangles):
    merged = []
    while rectangles:
        current = rectangles.pop(0)
        x, y, w, h = current
        merged_rect = current
        i = 0
        while i < len(rectangles):
            x2, y2, w2, h2 = rectangles[i]

            # حساب المسافة بين مراكز المستطيلات
            center1 = (x + w / 2, y + h / 2)
            center2 = (x2 + w2 / 2, y2 + h2 / 2)
            distance = (
                (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
            ) ** 0.5

            # حساب متوسط أحجام المستطيلات
            avg_size = (max(w, h) + max(w2, h2)) / 2

            # تحديد العتبة بناءً على متوسط الأحجام
            threshold = avg_size  # يمكن ضبط هذه النسبة حسب الحاجة

            # إذا كانت المسافة أقل من العتبة، قم بدمج المستطيلات
            if distance <= threshold:
                x_new = min(x, x2)
                y_new = min(y, y2)
                w_new = max(x + w, x2 + w2) - x_new
                h_new = max(y + h, y2 + h2) - y_new
                merged_rect = [x_new, y_new, w_new, h_new]
                x, y, w, h = merged_rect
                rectangles.pop(i)
            else:
                i += 1
        merged.append(merged_rect)
    return merged


if __name__ == "__main__":
    app.run(debug=True)
