from flask import Flask, request, jsonify
from flask_cors import CORS
from api_conversation import conversation_bp
from api_pronunciation import pronunciation_bp

app = Flask(__name__, static_url_path='/static')
CORS(app)  # Cho phép mọi origin truy cập


app.register_blueprint(conversation_bp)
app.register_blueprint(pronunciation_bp)

if __name__ == '__main__':
    app.run(debug=True)
