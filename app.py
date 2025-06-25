from flask import Flask, render_template, request, redirect, url_for
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility
import uuid

app = Flask(__name__)

# Milvus Configuration
COLLECTION_NAME = "notes_collection"
DIMENSION = 768  # Example dimension, adjust based on your embedding model
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

# Connect to Milvus
try:
    utility.connect(host=MILVUS_HOST, port=MILVUS_PORT)
except Exception as e:
    print(f"Could not connect to Milvus: {e}")
    exit()

# Define Milvus Schema and Collection
fields = [
    FieldSchema(name="note_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
    FieldSchema(name="note_text", dtype=DataType.VARCHAR, max_length=500),  # Store the original note text
    FieldSchema(name="note_embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
]
schema = CollectionSchema(fields, "A collection for storing notes and their embeddings.")

if not utility.has_collection(COLLECTION_NAME):
    collection = Collection(COLLECTION_NAME, schema)
    # Create index (example - adjust based on your needs)
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 1024},
    }
    collection.create_index(field_name="note_embedding", index_params=index_params)
    collection.load()
else:
    collection = Collection(COLLECTION_NAME)
    collection.load()

# Placeholder for embedding generation (you'll need a real embedding model)
def generate_embedding(text):
    # In a real application, you would use a library like SentenceTransformers
    # or call an external API to get the vector embedding of the text.
    # This is a placeholder returning a dummy vector.
    print(f"Generating dummy embedding for: {text}")
    return [0.0] * DIMENSION

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_note', methods=['POST'])
def add_note():
    note_text = request.form.get('note_text')
    if not note_text:
        return "Note text is required!", 400

    note_id = str(uuid.uuid4())
    note_embedding = generate_embedding(note_text)

    # Insert into Milvus
    try:
        data = [[note_id], [note_text], [note_embedding]]
        collection.insert(data)
        collection.flush()
        print(f"Note added with ID: {note_id}")
    except Exception as e:
        print(f"Error inserting note into Milvus: {e}")
        return "Error adding note to database", 500

    return redirect(url_for('index'))

@app.route('/search_notes', methods=['GET'])
def search_notes():
    query_text = request.args.get('query')
    if not query_text:
        return render_template('search_results.html', notes=[], query="")

    query_embedding = generate_embedding(query_text)

    # Search Milvus
    try:
        search_params = {
            "data": [query_embedding],
            "anns_field": "note_embedding",
            "param": {"metric_type": "L2", "params": {"nprobe": 10}},
            "limit": 10,  # Number of results to return
            "output_fields": ["note_text"]
        }
        results = collection.search(**search_params)

        found_notes = []
        for hit in results[0]:  # Assuming single query
            found_notes.append({
                "note_text": hit.entity.get("note_text"),
                "distance": hit.distance
            })

    except Exception as e:
        print(f"Error searching Milvus: {e}")
        return "Error searching notes", 500

    return render_template('search_results.html', notes=found_notes, query=query_text)

if __name__ == '__main__':
    # Create dummy templates for basic functionality
    # index.html
    with open('templates/index.html', 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Note Taking App</title>
        </head>
        <body>
            <h1>Add a New Note</h1>
            <form action="{{ url_for('add_note') }}" method="post">
                <textarea name="note_text" rows="4" cols="50" placeholder="Enter your note here"></textarea><br>
                <button type="submit">Add Note</button>
            </form>

            <hr>

            <h1>Search Notes</h1>
            <form action="{{ url_for('search_notes') }}" method="get">
                <input type="text" name="query" placeholder="Search notes">
                <button type="submit">Search</button>
            </form>
        </body>
        </html>
        """)

    # search_results.html
    with open('templates/search_results.html', 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Search Results</title>
        </head>
        <body>
            <h1>Search Results for "{{ query }}"</h1>
            {% if notes %}
                <ul>
                    {% for note in notes %}
                        <li>{{ note.note_text }} (Distance: {{ note.distance|round(2) }})</li>
                    {% endfor %}
                </ul>
            {% else %}
                <p>No notes found.</p>
            {% endif %}
            <p><a href="{{ url_for('index') }}">Back to Add Note</a></p>
        </body>
        </html>
        """)


    app.run(debug=True)