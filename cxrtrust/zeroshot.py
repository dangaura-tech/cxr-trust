"""Zero-shot multi-label classification with a vision-language model.

Each finding gets a set of *positive* and *negative* prompts. Their text embeddings are averaged
(prompt ensembling, Radford et al. 2021) and the image is scored by the contrast between the two
prototypes,

    logit_k(x) = s * (cos(f(x), t_k^+) - cos(f(x), t_k^-)),

where ``s`` is the model's logit scale. This turns CLIP's softmax over classes into independent
binary decisions, as multi-label radiology requires (in the spirit of CheXzero, Tiu et al.
2022). The logits are ordinary logits, so the rest of the toolkit applies unchanged:
temperature scaling, conformal prediction and selective prediction.

The default backend is BiomedCLIP (Zhang et al., 2023) through ``open_clip`` (``pip install
cxrtrust[vlm]``); the weights are downloaded from the Hugging Face hub on first use. Any object
with ``encode_images(list of PIL) -> (N, D)``, ``encode_texts(list of str) -> (T, D)`` and a
``logit_scale`` attribute can be used instead.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

POSITIVE_TEMPLATES = (
    "a chest x-ray showing {}.",
    "chest radiograph with {}.",
    "{} is present on this chest x-ray.",
    "findings consistent with {}.",
)
NEGATIVE_TEMPLATES = (
    "a chest x-ray with no {}.",
    "chest radiograph without {}.",
    "no evidence of {}.",
    "{} is absent on this chest x-ray.",
)

# label -> phrase used inside the templates (lower case, radiology wording)
PHRASES = {
    "Enlarged Cardiomediastinum": "an enlarged cardiomediastinum",
    "Cardiomegaly": "cardiomegaly",
    "Lung Opacity": "a lung opacity",
    "Lung Lesion": "a lung lesion",
    "Pleural Effusion": "a pleural effusion",
    "Pleural Other": "a pleural abnormality",
    "Support Devices": "support devices such as tubes or lines",
    "Pleural_Thickening": "pleural thickening",
    "Nodule": "a lung nodule",
    "Mass": "a lung mass",
}

# findings whose natural prompts are not "<finding> present / absent"
SPECIAL_PROMPTS = {
    "No Finding": (
        ["a normal chest x-ray.", "no acute cardiopulmonary abnormality.", "a chest radiograph with no findings."],
        ["an abnormal chest x-ray.", "a chest radiograph with acute findings.", "a chest x-ray showing disease."],
    ),
}


def build_prompts(labels: Sequence[str]) -> Dict[str, Tuple[List[str], List[str]]]:
    prompts = {}
    for lab in labels:
        if lab in SPECIAL_PROMPTS:
            prompts[lab] = SPECIAL_PROMPTS[lab]
            continue
        phrase = PHRASES.get(lab, lab.replace("_", " ").lower())
        prompts[lab] = ([t.format(phrase) for t in POSITIVE_TEMPLATES], [t.format(phrase) for t in NEGATIVE_TEMPLATES])
    return prompts


class VisionLanguageBackend(Protocol):
    logit_scale: float

    def encode_images(self, images) -> np.ndarray: ...

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


class OpenCLIPBackend:
    """open_clip model loaded lazily (default: BiomedCLIP ViT-B/16 + PubMedBERT)."""

    def __init__(
        self,
        model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name, self.batch_size = model_name, batch_size
        self._device = device
        self._model = self._preprocess = self._tokenizer = None

    def _load(self):
        import open_clip
        import torch

        if self._model is None:
            self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model, self._preprocess = open_clip.create_model_from_pretrained(self.model_name)
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            self._model = self._model.to(self._device).eval()
        return torch

    @property
    def logit_scale(self) -> float:
        self._load()
        return float(self._model.logit_scale.exp().item())

    def encode_images(self, images) -> np.ndarray:
        torch = self._load()
        out = []
        with torch.no_grad():
            for s in range(0, len(images), self.batch_size):
                x = torch.stack([self._preprocess(im.convert("RGB")) for im in images[s : s + self.batch_size]])
                out.append(self._model.encode_image(x.to(self._device)).float().cpu().numpy())
        return _l2(np.concatenate(out))

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._load()
        with torch.no_grad():
            tok = self._tokenizer(list(texts)).to(self._device)
            return _l2(self._model.encode_text(tok).float().cpu().numpy())


class ZeroShotClassifier:
    """Prompt-ensembled positive-vs-negative zero-shot classifier over a list of findings."""

    def __init__(
        self,
        backend: VisionLanguageBackend,
        labels: Sequence[str],
        prompts: Optional[Dict[str, Tuple[Sequence[str], Sequence[str]]]] = None,
    ) -> None:
        self.backend, self.labels = backend, list(labels)
        self.prompts = prompts or build_prompts(self.labels)
        self._text: Optional[np.ndarray] = None

    def text_prototypes(self) -> np.ndarray:
        """(K, 2, D): averaged, re-normalised positive and negative prompt embeddings per finding."""
        if self._text is None:
            protos = []
            for lab in self.labels:
                pos, neg = self.prompts[lab]
                protos.append([_l2(self.backend.encode_texts(list(p)).mean(0)) for p in (pos, neg)])
            self._text = np.asarray(protos)
        return self._text

    def logits(self, images) -> np.ndarray:
        img = self.backend.encode_images(images)  # (N, D)
        t = self.text_prototypes()
        sim = np.einsum("nd,kcd->nkc", img, t)
        return self.backend.logit_scale * (sim[..., 0] - sim[..., 1])

    def predict_proba(self, images) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logits(images)))
