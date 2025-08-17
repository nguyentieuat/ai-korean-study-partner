from flask import Flask, request, jsonify
from flask_cors import CORS
from api_conversation import conversation_bp

app = Flask(__name__, static_url_path='/static')
CORS(app)  # Cho phép mọi origin truy cập


app.register_blueprint(conversation_bp)


# Health check endpoint cho ELB
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    # Chạy chat service ở cổng 5001
    app.run(host='0.0.0.0', port=5001, debug=True)

