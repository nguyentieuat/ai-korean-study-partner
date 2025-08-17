from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel
from topik1.generate_question import generate_topik_question

app = Flask(__name__, static_url_path='/static')
CORS(app)  # Cho phép mọi origin truy cập

@app.post("/api/generate_question")
def generate_question():
    try:
        # Lấy JSON từ request
        data = request.get_json()
        level = data.get("level", "").lower()
        category = data.get("category")
        cau = data.get("cau")
        if level == "topik1":
            return generate_topik_question(danh_muc=category, cau=str(cau))
    except Exception as e:
        return jsonify({"error": f"Invalid request: {str(e)}"}), 400
   
# Health check endpoint cho ELB
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    # Chạy tqg service ở cổng 5003
    app.run(host='0.0.0.0', port=5003, debug=True)
