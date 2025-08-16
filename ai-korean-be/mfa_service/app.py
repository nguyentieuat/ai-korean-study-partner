from flask import Flask, request, jsonify
from flask_cors import CORS
# from api_evaluateMFAW2V import evaluateMFAW2V_bp
from api_evaluateASRneedleman import evaluateASRneedleman_bp

app = Flask(__name__, static_url_path='/static')
CORS(app)  # Cho phép mọi origin truy cập


# app.register_blueprint(evaluateMFAW2V_bp)
app.register_blueprint(evaluateASRneedleman_bp)

if __name__ == '__main__':
    # Chạy MFA service ở cổng 5002
    app.run(host='0.0.0.0', port=5002, debug=True)
