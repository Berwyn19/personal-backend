from flask import Flask, request, jsonify
from flask_cors import CORS
from search import load_vectordb, search, clean_text, complete_chat

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins="*")

# Load FAISS vector DB once at startup
vectordb = load_vectordb()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("question", "")

    if not query:
        return jsonify({"error": "No question provided"}), 400

    try:
        context = clean_text(search(query, vectordb))
        response = complete_chat(query, context)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8000)
