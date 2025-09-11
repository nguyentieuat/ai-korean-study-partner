from flask import Flask, request, jsonify
from flask_cors import CORS
# from api_evaluateMFAW2V import evaluateMFAW2V_bp
from api_vitsTTS import vitsTTS_bp

app = Flask(__name__, static_url_path='/static')
CORS(app)  # Cho phép mọi origin truy cập


# app.register_blueprint(evaluateMFAW2V_bp)
app.register_blueprint(vitsTTS_bp)

# Health check endpoint cho ELB
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    # Chạy ASR service ở cổng 5004
    app.run(host='0.0.0.0', port=5004, debug=True)
