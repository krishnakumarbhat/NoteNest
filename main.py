import os
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

from flask import Flask, send_file, request, redirect, url_for

app = Flask(__name__)

# Milvus connection parameters
MILVUS_HOST = os.environ.get("MILVUS_HOST", "localhost")
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
COLLECTION_NAME = "notes"
DIMENSION = 10  # Example dimension, adjust based on your embedding model

# Connect to Milvus
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

# Define Milvus collection schema (basic example)
fields = [
    FieldSchema(name="note", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=500),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
]
schema = CollectionSchema(fields, "Notes collection")

# Create Milvus collection if it doesn't exist
if COLLECTION_NAME not in connections.get_connection("default").list_collections():
    collection = Collection(COLLECTION_NAME, schema)
else:
    collection = Collection(COLLECTION_NAME)

@app.route("/")
def index():
    return send_file('templates/index.html')

def main():
    app.run(port=int(os.environ.get('PORT', 80)))

@app.route("/add_note", methods=["POST"])
def add_note():
    note = request.form.get("note")
    if note:
        # In a real application, you would generate an embedding for the note here.
        # For this basic example, we'll use a placeholder vector.
        embedding = [0.0] * DIMENSION

        data = [[note], [embedding]]
        collection.insert(data)
        collection.flush()
    return redirect(url_for("index"))

if __name__ == "__main__":
    main()
