import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from search import load_vectordb, search, clean_text, complete_chat

app = Flask(__name__)
CORS(app, origins="*")  # later replace "*" with ["https://your-frontend.vercel.app"]

vectordb = load_vectordb()

@app.post("/chat")
def chat():
    data = request.get_json() or {}
    query = data.get("question", "").strip()
    if not query:
        return jsonify({"error": "No question provided"}), 400

    try:
        context = clean_text(search(query, vectordb))
        response = complete_chat(query, context)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))        # Render supplies $PORT
    app.run(host="0.0.0.0", port=port, debug=False)
