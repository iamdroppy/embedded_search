import os
import random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from PIL import Image
from rich import print as echo
from rich.markup import escape
from qdrant_client import QdrantClient
from qdrant_client.http import models

class VectorDB:
    """
    Wrapper class for interacting with a Qdrant vector database.
    Provides methods for collection management, vector insertion, and search.
    """
    def __init__(self, host: str, port: int):
        """
        Initialize the VectorDB client.

        Args:
            host (str): Host address of the Qdrant server.
            port (int): Port number of the Qdrant server.
        """
        self.client = QdrantClient(host=host, port=port)

    def has_collection(self, name: str) -> bool:
        """
        Check if a collection exists in the database.

        Args:
            name (str): Name of the collection.

        Returns:
            bool: True if collection exists, False otherwise.
        """
        try:
            collection = self.client.get_collection(name)
            return collection is not None
        except Exception as e:
            echo(f"[red]Error checking collection {escape(name)}: {e}[/red]")
            return False

    def create_collection(self, name: str, vector_size: int):
        """
        Create a new collection with the specified vector size.

        Args:
            name (str): Name of the collection.
            vector_size (int): Size of the vectors to be stored.

        Returns:
            Collection object.
        """
        if self.has_collection(name):
            echo(f"[yellow]Collection already exists:[/yellow] {escape(name)}")
            return self.client.get_collection(name)
        collection = self.client.recreate_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
        )
        echo(f"[green]Created collection:[/green] {escape(name)}")
        return collection

    def insert_vector(self, collection_name: str, vector: torch.Tensor, payload: dict = None):
        """
        Insert a vector and its payload into a collection.

        Args:
            collection_name (str): Name of the collection.
            vector (torch.Tensor): Vector to insert.
            payload (dict, optional): Additional metadata to store.
        """
        if vector.numel() == 0:
            echo(f"[red]Empty vector, skipping insertion into collection {escape(collection_name)}[/red]")
            return
        vector_list = vector.squeeze(0).tolist()
        point_id = random.randint(1, 1_000_000_000)
        self.client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(id=point_id, vector=vector_list, payload=payload or {})
            ]
        )
        echo(f"[green]Inserted vector into collection:[/green] {escape(collection_name)} with point ID {point_id}")

    def search_vector(self, collection_name: str, vector: torch.Tensor, top_k: int = 5):
        """
        Search for similar vectors in a collection.

        Args:
            collection_name (str): Name of the collection.
            vector (torch.Tensor): Query vector.
            top_k (int, optional): Number of top results to return.

        Returns:
            list: Search results.
        """
        if vector.numel() == 0:
            echo(f"[red]Empty vector, cannot perform search in collection {escape(collection_name)}[/red]")
            return []
        vector_list = vector.squeeze(0).tolist()
        results = self.client.search(
            collection_name=collection_name,
            query_vector=vector_list,
            limit=top_k
        )
        echo(f"[blue]Search results in collection:[/blue] {escape(collection_name)}")
        return results

class EmbedVision:
    """
    Image embedding class using a vision transformer model.
    """
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-vision-v1.5"):
        """
        Initialize the vision embedder.

        Args:
            model_name (str, optional): HuggingFace model name.
        """
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    def embed(self, image: Image.Image) -> torch.Tensor:
        """
        Generate an embedding for a given image.

        Args:
            image (PIL.Image.Image): Input image.

        Returns:
            torch.Tensor: Normalized image embedding.
        """
        inputs = self.processor(image, return_tensors="pt")
        with torch.no_grad():
            img_emb = self.model(**inputs).last_hidden_state
        img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)
        return img_embeddings

class EmbedText:
    """
    Text embedding class using a transformer model.
    """
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        """
        Initialize the text embedder.

        Args:
            model_name (str, optional): HuggingFace model name.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    def embed(self, text: str) -> torch.Tensor:
        """
        Generate an embedding for a given text string.

        Args:
            text (str): Input text.

        Returns:
            torch.Tensor: Normalized text embedding.
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            text_emb = self.model(**inputs).last_hidden_state
        text_embeddings = F.normalize(text_emb[:, 0], p=2, dim=1)
        return text_embeddings

def embed_image(file_path: str, vision_embedder: EmbedVision) -> torch.Tensor:
    """
    Embed an image file using the provided vision embedder.

    Args:
        file_path (str): Path to the image file.
        vision_embedder (EmbedVision): Vision embedding model.

    Returns:
        torch.Tensor: Image embedding tensor.
    """
    try:
        echo(f"[blue]Embedding file:[/blue] {escape(file_path)}")
        image = Image.open(file_path)
        image_embedding = vision_embedder.embed(image)
        echo(f"Embedded [red]image[/red] to vector of size {image_embedding.size()}.")
        return image_embedding
    except Exception as e:
        echo(f"[red]Error embedding file {file_path}: {e}[/red]")
        return torch.empty(0)

def save_embeddings(
    collection_name: str = "image_text_embeddings",
    vector_host: str = "localhost",
    vector_port: int = 6333,
    image_dir: str = "./images"
):
    """
    Embed all images in a directory and save their embeddings to the vector database.

    Args:
        collection_name (str, optional): Name of the vector collection.
        vector_host (str, optional): Host of the vector database.
        vector_port (int, optional): Port of the vector database.
        image_dir (str, optional): Directory containing images to embed.
    """
    vision_embedder = EmbedVision()
    vector_db = VectorDB(host=vector_host, port=vector_port)
    vector_db.create_collection(name=collection_name, vector_size=768)

    allowed_extensions = ('.png', '.jpg', '.jpeg', '.webp')

    for filename in os.listdir(image_dir):
        if not filename.lower().endswith(allowed_extensions):
            echo(f"[yellow]Skipping non-image file:[/yellow] [yellow bold]{escape(filename)}[/yellow bold]")
            continue
        file_path = os.path.join(image_dir, filename)
        image_embedding = embed_image(file_path, vision_embedder)
        full_file_path = os.path.abspath(file_path)
        echo(f"[blue]Storing embedding for file:[/blue] [blue bold]{escape(full_file_path)}[/blue bold]")
        vector_db.insert_vector(
            collection_name=collection_name,
            vector=image_embedding,
            payload={"filename": full_file_path}
        )

def search_embedding(
    collection_name: str = "image_text_embeddings",
    vector_host: str = "localhost",
    vector_port: int = 6333
):
    """
    Launch a Gradio UI to search for images by text query.

    Args:
        collection_name (str, optional): Name of the vector collection.
        vector_host (str, optional): Host of the vector database.
        vector_port (int, optional): Port of the vector database.
    """
    import gradio as gr

    def perform_search(query: str, top_k: int):
        """
        Perform a search in the vector database using a text query.

        Args:
            query (str): Text query.
            top_k (int): Number of top results to return.

        Returns:
            list: List of images corresponding to the top results.
        """
        vector_db = VectorDB(host=vector_host, port=vector_port)
        text_embedder = EmbedText()
        search_vector = text_embedder.embed(query)
        results = vector_db.search_vector(collection_name=collection_name, vector=search_vector, top_k=top_k)
        if not results:
            echo(f"[red]No results found for query:[/red] {escape(query)}")
            return [None] * top_k
        echo(f"[green]Top {top_k} results for query:[/green] {escape(query)}")
        images = []
        for result in results:
            echo(f"[pink]Point ID:[/pink] {result.id}, [pink]Score:[/pink] {result.score}, [pink]Payload:[/pink] {result.payload}")
            path = result.payload.get("filename")
            try:
                img = Image.open(path)
                images.append(img)
            except Exception as e:
                echo(f"[red]Error loading image {path}: {e}[/red]")
                images.append(None)
        while len(images) < top_k:
            images.append(None)
        return images

    with gr.Blocks() as search_ui:
        with gr.Row():
            with gr.Column():
                text_input = gr.TextArea(label="Input Text")
            with gr.Column():
                top_k_input = gr.Slider(label="Top K", minimum=1, maximum=50, step=1, value=10)
                search_button = gr.Button("Search")
        with gr.Row():
            image_gallery = gr.Gallery(label="Results", columns=5, rows=2, object_fit="contain", height="auto")
        search_button.click(
            fn=perform_search,
            inputs=[text_input, top_k_input],
            outputs=image_gallery
        )
    search_ui.launch()

def main():
    """
    Main entry point for the script.
    Parses command-line arguments and runs the selected mode.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Embed images and store in Qdrant vector database.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--save", action="store_true", help="Embed images and save to vector database.")
    group.add_argument("--search", action="store_true", help="Search vector database with a text query.")
    parser.add_argument("--collection", '-c', type=str, default="nomic-test", help="Name of the vector collection.")
    parser.add_argument("--vector-host", '-e', type=str, default="localhost", help="Host of the vector database.")
    parser.add_argument("--vector-port", '-p', type=int, default=6333, help="Port of the vector database.")
    parser.add_argument("--image-dir", type=str, default="./images", help="Directory containing images to embed.")
    args = parser.parse_args()

    if args.save:
        save_embeddings(args.collection, args.vector_host, args.vector_port, args.image_dir)
    elif args.search:
        search_embedding(args.collection, args.vector_host, args.vector_port)

if __name__ == "__main__":
    main()