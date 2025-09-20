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
