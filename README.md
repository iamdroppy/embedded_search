# Embedded Search using Nomic

Text-to-image search using open-source embeddings and a local vector database (Qdrant).  
This repo lets you:
- Embed a folder of images with a vision model
- Store normalized embeddings in Qdrant
- Launch a Gradio UI to search those images using natural language

It uses:
- Vision model: [nomic-ai/nomic-embed-vision-v1.5](https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5)
- Text model: [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- Vector DB: [Qdrant](https://qdrant.tech)
- UI: [Gradio](https://gradio.app)

Both image and text embeddings are L2-normalized, and Qdrant uses cosine distance for similarity.

---

## Features

- Simple, self-contained script (no framework required)
- Embed images from a folder (png, jpg, jpeg, webp)
- Store vectors in Qdrant with filename payloads
- Text query → text embedding → vector similarity search
- Interactive Gradio gallery returning top-K matches
- Clear logging via Rich
- Safe-guards for empty vectors and non-image files

---

## How it works

1. Image embedding
   - Images are processed with `AutoImageProcessor` and the vision model.
   - We take the first token from `last_hidden_state` (CLS-equivalent) and L2-normalize it.
   - Resulting vector size is 768.

2. Text embedding
   - Text is tokenized with `AutoTokenizer` and passed to the text model.
   - We take the first token from `last_hidden_state` and L2-normalize it.
   - Resulting vector size is 768.

3. Vector storage and search
   - A Qdrant collection is created (if missing) with size 768 and cosine distance.
   - Each image embedding is stored with payload: `{ "filename": "<absolute path>" }`. **TODO**: Change to file name only for easier gallery or other ends.
   - On search, the text query embedding is computed and used to query Qdrant.
   - Top-K results are displayed as images in a Gradio gallery, along with console logs of IDs, scores, and payloads.

---

## Requirements

- Python 3.9+
- A running Qdrant instance (default localhost:6333)

Python packages:
- torch
- transformers
- pillow
- qdrant-client
- rich
- gradio

Install:
```bash
pip install --upgrade pip
pip install "transformers>=4.40" "pillow>=10" "qdrant-client>=1.9" "rich>=13" "gradio>=4"
# Install torch appropriate to your system (CPU/CUDA):
# See https://pytorch.org/get-started/locally/ for the correct command, e.g.:
# CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Start Qdrant

Quickstart with Docker:
```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

- Default script expects Qdrant at host `localhost` and port `6333`.
- If you change host/port, pass them via CLI flags.

---

## Usage

The script has two modes: save (embed and store) and search (launch UI).

CLI help:
```bash
python main.py --help
```

### 1) Save embeddings to Qdrant

- Place your images in a folder (default: ./images).
- Supported extensions: .png, .jpg, .jpeg, .webp.

Example:
```bash
python main.py --save \
  --image-dir ./images \
  --collection nomic-test \
  --vector-host localhost \
  --vector-port 6333
```

What happens:
- Loads vision model nomic-embed-vision-v1.5
- Creates collection (if missing) with size 768 and cosine distance
- Iterates allowed files, embeds, normalizes, and upserts with a random point ID
- Stores absolute path in payload under "filename"

Notes:
- If an image cannot be opened or embedded, it’s skipped and logged.
- Collection name default for CLI is `nomic-test`.

### 2) Search via Gradio UI

Example:
```bash
python main.py --search \
  --collection nomic-test \
  --vector-host localhost \
  --vector-port 6333
```

What happens:
- Loads text model nomic-embed-text-v1.5
- Embeds your query text
- Queries Qdrant and shows the top results in a Gradio gallery
- Default gallery is 5 columns x 2 rows, slider controls Top-K (1..50, default 10)
- The app typically launches at http://127.0.0.1:7860

Tip: You can enter natural language queries describing the content of the images you embedded.

---

## Examples

Embed a sample folder of images:
```bash
python main.py --save --image-dir ./sample_images --collection nomic-test
```

Launch the search UI:
```bash
python main.py --search --collection nomic-test
```

Change Qdrant endpoint:
```bash
python main.py --save --vector-host 192.168.1.10 --vector-port 6333
python main.py --search --vector-host 192.168.1.10 --vector-port 6333
```

Use a different collection:
```bash
python main.py --save --collection my-images
python main.py --search --collection my-images
```

---

## Programmatic use

You can also use the classes directly.

Embed and store a single image:
```python
from main import EmbedVision, VectorDB, embed_image

vision = EmbedVision()
db = VectorDB(host="localhost", port=6333)

# Ensure collection exists (768 for the Nomic models used here)
db.create_collection(name="nomic-test", vector_size=768)

vec = embed_image("./images/cat.jpg", vision)
db.insert_vector("nomic-test", vec, payload={"filename": "/abs/path/to/images/cat.jpg"})
```

Text search without UI:
```python
from main import EmbedText, VectorDB

db = VectorDB(host="localhost", port=6333)
txt = EmbedText()

qvec = txt.embed("a small kitten on a couch")
results = db.search_vector("nomic-test", qvec, top_k=5)
for r in results:
    print(r.id, r.score, r.payload.get("filename"))
```

---

## Configuration details

- Model names
  - Vision: nomic-ai/nomic-embed-vision-v1.5
  - Text: nomic-ai/nomic-embed-text-v1.5
  - You can change these in `EmbedVision` and `EmbedText` initializers.

- Vector size
  - Hardcoded as 768 when creating collections (nomic embed vector size).
  - If you switch models, ensure the collection vector size matches the embedding dimension.

- Distance
  - Qdrant uses COSINE distance.

- Payload
  - Only `filename` is stored (absolute path).
  - If you move files after embedding, the search UI may fail to open them.

- De-duplication
  - None by default; IDs are random. Re-embedding the same file will create multiple points.

- Allowed image extensions
  - .png, .jpg, .jpeg, .webp

---

## Troubleshooting

- Connection refused to Qdrant
  - Ensure Qdrant is running and reachable at the host/port you passed.
  - Try: `curl http://localhost:6333/readyz` → should return OK.

- Model download is slow or fails
  - Hugging Face may throttle or require authentication in some contexts.
  - Try setting `HF_HOME` or pre-downloading models.

- CUDA/CPU issues
  - Install PyTorch matching your CUDA runtime, or use CPU wheels.
  - Force CPU by setting `CUDA_VISIBLE_DEVICES=""` before running.

- Empty results
  - Verify you ran `--save` first on the same collection name.
  - Check that the images were recognized (file extensions).
  - Confirm the Top-K slider is > 0.

- Images fail to load in UI
  - Files are referenced by absolute paths saved at embed time.
  - If files were moved or deleted, re-embed or update payloads.

---

## Roadmap ideas

- [ ] Add metadata in the gallery
- [ ] Batch upserts and progress bars
- [ ] Optional deduplication by filename hash
- [ ] Configurable models and vector size via CLI flags
- [ ] Dockerfile and/or Compose for end-to-end spin-up
- [ ] Multi-modal search via `--search` UI (image → image) and filtering

---

## Acknowledgements

- [Nomic AI](https://nomic.ai) for open-source embedding models
- [Qdrant](https://qdrant.tech) for the vector database
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Gradio](https://gradio.app) for the UI
- [Rich](https://github.com/Textualize/rich) for logging

---

## License

View the `LICENSE` file for more information.
