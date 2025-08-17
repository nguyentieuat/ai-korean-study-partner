from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from api_conversation import conversation_bp
from api_pronunciation import pronunciation_bp
from api_topikgenerate import topikgenerate_bp
from api_cooperate import cooperate_bp
import os

app = Flask(__name__, static_url_path='/static')
CORS(app)  # Cho phép mọi origin truy cập


app.register_blueprint(conversation_bp)
app.register_blueprint(pronunciation_bp)
app.register_blueprint(topikgenerate_bp)
app.register_blueprint(cooperate_bp)

# Health check endpoint cho ELB
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route("/static/<path:filename>")
def serve_pronun_file(filename):
    file_path = os.path.join("static/", filename)
    if not os.path.isfile(file_path):
        return abort(404)

    # Trả file về client
    return send_from_directory("static/", filename)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
