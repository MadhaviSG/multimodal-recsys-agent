"""
VLM: PaliGemma Visual Preference Extraction
=============================================
Image input → user preference signal → retrieval-ready query string.

ML System Design decisions documented inline.

Design decision: VLM → text description → hybrid retrieval (Option 1).
Alternative: VLM → embedding → ANN search (Option 2, production upgrade).

Option 1 chosen because:
- Plugs into existing hybrid retrieval pipeline with zero new infrastructure
- Text description is auditable — can log and debug failures
- No projection layer training required

Option 2 (at scale): train a projection layer mapping VLM image embeddings
into item tower space. Enables direct ANN retrieval from poster images.
Requires poster-to-interaction training signal. Noted as future extension.

Usage:
    processor = VLMProcessor()
    query = processor.image_to_query("poster.jpg")
    # query → drop into agent pipeline as text query
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image


@dataclass
class ImagePreferences:
    description: str           # full visual description
    genres: list[str]          # inferred genres
    mood: str                  # dark/light/tense/funny/etc.
    themes: list[str]          # inferred themes
    era: str                   # modern/retro/80s/etc.
    query: str                 # retrieval-ready combined query string


# Structured preference extraction prompt
PREFERENCE_PROMPT = """Look at this movie poster and extract viewing preferences.

Respond ONLY with valid JSON:
{
    "description": "<2-3 sentence description of visual style and mood>",
    "genres": ["<genre1>", "<genre2>"],
    "mood": "<one word: dark/light/tense/uplifting/funny/scary/romantic/thrilling>",
    "themes": ["<theme1>", "<theme2>", "<theme3>"],
    "era": "<decade or 'modern': 1980s/1990s/2000s/modern>",
    "target_audience": "<general/adult/family/teen>"
}

Be specific. Base everything on visual cues: color palette, typography,
imagery, composition. Do not guess the movie title.
"""


class VLMProcessor:
    """
    PaliGemma-based visual preference extraction.

    Design decision: PaliGemma over LLaVA or GPT-4V.
    - PaliGemma: open source, runs locally, smaller (3B params)
    - LLaVA: open source, larger (7-13B), higher quality
    - GPT-4V: best quality, API cost per call, no local control

    For our use case (poster preference extraction, not detailed VQA),
    PaliGemma 3B is sufficient and runs on a single GPU.

    At scale: GPT-4V or Claude 3 for higher accuracy + multilingual support.
    """

    def __init__(self, model_id: str = "google/paligemma-3b-pt-224", device: str = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy load — only when first image is processed."""
        if self._model is None:
            from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
            print(f"Loading PaliGemma on {self.device}...")
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = PaliGemmaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self._model.eval()

    def load_image(self, source: str) -> Image.Image:
        """
        Load image from file path or URL.
        Converts to RGB (handles RGBA posters gracefully).
        """
        if source.startswith("http://") or source.startswith("https://"):
            import requests
            from io import BytesIO
            response = requests.get(source, timeout=10)
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(source)

        return img.convert("RGB")

    def describe_poster(self, image: Image.Image) -> str:
        """
        Generate a free-form description of the movie poster.
        Used as fallback when structured extraction fails.
        """
        self._load_model()

        prompt = "Describe this movie poster in detail. What genre, mood, and themes does it convey?"
        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
            )

        description = self._processor.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return description.strip()

    def extract_preferences(self, image: Image.Image) -> ImagePreferences:
        """
        Extract structured viewing preferences from poster.

        Design decision: structured JSON output over free-form description.
        JSON fields (genres, mood, themes) map directly to Qdrant payload
        fields — enables metadata-filtered retrieval from visual input.
        Free-form description is kept as fallback for retrieval query.
        """
        import json
        self._load_model()

        inputs = self._processor(
            text=PREFERENCE_PROMPT,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
            )

        raw = self._processor.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Parse JSON with fallback
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: use raw output as description
            return ImagePreferences(
                description=raw,
                genres=[],
                mood="unknown",
                themes=[],
                era="modern",
                query=raw[:200],
            )

        # Build retrieval query from structured preferences
        query_parts = []
        if data.get("genres"):
            query_parts.append(" ".join(data["genres"]))
        if data.get("mood"):
            query_parts.append(data["mood"])
        if data.get("themes"):
            query_parts.extend(data["themes"][:2])
        if data.get("era") and data["era"] != "modern":
            query_parts.append(data["era"])

        query = " ".join(query_parts) + " movies"

        return ImagePreferences(
            description=data.get("description", raw),
            genres=data.get("genres", []),
            mood=data.get("mood", ""),
            themes=data.get("themes", []),
            era=data.get("era", "modern"),
            query=query,
        )

    def image_to_query(self, image_source: str) -> dict:
        """
        Full pipeline: image path/URL → agent-ready dict.

        Returns dict with:
            query: str           — retrieval-ready text query
            genre_filter: list   — for Qdrant metadata filter
            preferences: dict    — full structured preferences for logging

        Design decision: return genre_filter separately from query.
        This enables the candidate generator to apply Qdrant metadata
        filtering directly from visual input — not just text search.
        Visual → filtered ANN retrieval is more precise than text alone.
        """
        image = self.load_image(image_source)
        prefs = self.extract_preferences(image)

        return {
            "query": prefs.query,
            "genre_filter": prefs.genres if prefs.genres else None,
            "preferences": {
                "description": prefs.description,
                "mood": prefs.mood,
                "themes": prefs.themes,
                "era": prefs.era,
            },
            "is_visual": True,
        }

    def get_visual_embedding(self, image: Image.Image) -> torch.Tensor:
        """
        Extract raw visual embedding from PaliGemma vision encoder.

        This is the Option 2 foundation — not used in current pipeline.
        At scale: train a projection layer mapping this embedding into
        item tower space for direct ANN retrieval from poster images.

        Returns: (1, hidden_dim) float32 tensor
        """
        self._load_model()

        inputs = self._processor(
            text="",
            images=image,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            # Extract vision encoder output (before language model)
            vision_outputs = self._model.vision_tower(
                inputs["pixel_values"]
            )
            # Mean pool over patch tokens
            embedding = vision_outputs.last_hidden_state.mean(dim=1)

        return embedding.cpu().float()