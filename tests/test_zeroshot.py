import numpy as np
import pytest

from cxrtrust.data import CHEXPERT_LABELS, NIH_LABELS
from cxrtrust.zeroshot import PHRASES, ZeroShotClassifier, build_prompts


class FakeBackend:
    """Deterministic 'vision-language model': images and texts live in a shared toy space.

    Dimension k encodes "finding k is present"; negative prompts point the other way. An image
    is a dict of present findings, so the zero-shot scores are known in closed form.
    """

    logit_scale = 50.0

    def __init__(self, labels):
        self.labels = labels
        self.calls = 0

    def encode_texts(self, texts):
        self.calls += 1
        out = np.zeros((len(texts), len(self.labels) + 1))
        for i, t in enumerate(texts):
            k = next(k for k, lab in enumerate(self.labels) if PHRASES.get(lab, lab.lower()) in t)
            neg = any(w in t for w in ("no ", "without", "absent"))
            out[i, k] = -1.0 if neg else 1.0
            out[i, -1] = 1.0
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def encode_images(self, images):
        out = np.zeros((len(images), len(self.labels) + 1))
        for i, present in enumerate(images):
            out[i, :-1] = [0.5 if lab in present else -0.5 for lab in self.labels]
            out[i, -1] = 1.0
        return out / np.linalg.norm(out, axis=1, keepdims=True)


@pytest.mark.parametrize("labels", [CHEXPERT_LABELS, NIH_LABELS])
def test_prompts_cover_every_label(labels):
    prompts = build_prompts(labels)
    assert set(prompts) == set(labels)
    for pos, neg in prompts.values():
        assert len(pos) >= 3 and len(neg) >= 3
        assert not set(pos) & set(neg)


def test_zero_shot_scores_follow_the_image_content():
    labels = ["Cardiomegaly", "Edema", "Pneumothorax"]
    backend = FakeBackend(labels)
    clf = ZeroShotClassifier(backend, labels)
    probs = clf.predict_proba([{"Edema"}, {"Cardiomegaly", "Pneumothorax"}, set()])
    assert probs.shape == (3, 3)
    assert probs[0, 1] > 0.9 and probs[0, 0] < 0.1
    assert probs[1, 0] > 0.9 and probs[1, 2] > 0.9 and probs[1, 1] < 0.1
    assert (probs[2] < 0.1).all()
    calls = backend.calls
    clf.logits([{"Edema"}])
    assert backend.calls == calls  # text prototypes are cached


def test_prototypes_are_unit_norm():
    labels = ["Edema", "Nodule"]
    t = ZeroShotClassifier(FakeBackend(labels), labels).text_prototypes()
    assert t.shape[:2] == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(t, axis=-1), 1.0, atol=1e-9)
