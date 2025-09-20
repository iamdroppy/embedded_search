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
from embeds import EmbedVision, EmbedText
from vector_db import VectorDB
from loguru import logger

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
        logger.debug(f"Starting: Embedding file: {escape(file_path)}")
        image = Image.open(file_path)
        image_embedding = vision_embedder.embed(image)
        logger.debug(f"Image embedded to a size of {image_embedding.size()}.")
        return image_embedding
    except Exception as e:
        logger.error(f"Error embedding file {file_path}: {e}")
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
            logger.warning(f"Skipping non-image file: {escape(filename)}")
            continue
        file_path = os.path.join(image_dir, filename)
        image_embedding = embed_image(file_path, vision_embedder)
        full_file_path = os.path.abspath(file_path)
        logger.info(f"Storing embedding for file: {escape(full_file_path)}...")
        vector_db.insert_vector(
            collection_name=collection_name,
            vector=image_embedding,
            payload={"filename": full_file_path}
        )

def search_embedding_ui(
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
            logger.warning(f"No results found for query: {escape(query)}")
            return [None] * top_k
        logger.info(f"Top {top_k} results for query: {escape(query)}")
        images = []
        for result in results:
            logger.info(f"Point ID: {result.id}, Score: {result.score}, Payload: {result.payload}")
            path = result.payload.get("filename")
            try:
                img = Image.open(path)
                images.append(img)
            except Exception as e:
                logger.error(f"Error loading image {path}: {e}")
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
        search_embedding_ui(args.collection, args.vector_host, args.vector_port)

if __name__ == "__main__":
    main()