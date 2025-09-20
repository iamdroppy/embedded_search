import os
import random
import torch
from PIL import Image
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
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
