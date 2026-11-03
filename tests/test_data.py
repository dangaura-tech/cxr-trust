import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cxrtrust.data import ChestXrayDataset, build_transforms, patient_split
from cxrtrust.data.datasets import apply_uncertainty_policy, load_chexpert_csv, load_nih_csv


def test_uncertainty_policies():
    raw = np.array([[1.0, 0.0, -1.0, np.nan]])
    t, m = apply_uncertainty_policy(raw.copy(), "ones")
    np.testing.assert_array_equal(t, [[1, 0, 1, 0]])
    np.testing.assert_array_equal(m, [[1, 1, 1, 1]])
    t, m = apply_uncertainty_policy(raw.copy(), "ignore")
    np.testing.assert_array_equal(t, [[1, 0, 0, 0]])
    np.testing.assert_array_equal(m, [[1, 1, 0, 1]])
    with pytest.raises(ValueError):
        apply_uncertainty_policy(raw.copy(), "random")


def test_patient_split_has_no_leakage():
    df = pd.DataFrame({"Path": [f"img{i}.png" for i in range(100)], "patient_id": [f"p{i // 4}" for i in range(100)]})
    s = patient_split(df, val_frac=0.2, test_frac=0.2, seed=1)
    sets = [set(s[k]["patient_id"]) for k in ("train", "val", "test")]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert sum(len(v) for v in s.values()) == 100


def test_csv_loaders(tmp_path):
    chex = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "Path": [
                "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg",
                "CheXpert-v1.0-small/train/patient00001/study1/view2_lateral.jpg",
            ],
            "Frontal/Lateral": ["Frontal", "Lateral"],
            "Edema": [1.0, -1.0],
        }
    ).to_csv(chex, index=False)
    df = load_chexpert_csv(str(chex), ["Edema"])
    assert len(df) == 1 and df.loc[0, "patient_id"] == "patient00001"

    nih = tmp_path / "nih.csv"
    pd.DataFrame(
        {"Image Index": ["a.png", "b.png"], "Finding Labels": ["Mass|Edema", "No Finding"], "Patient ID": [1, 2]}
    ).to_csv(nih, index=False)
    df = load_nih_csv(str(nih), ["Mass", "Edema", "Hernia"])
    assert df[["Mass", "Edema", "Hernia"]].to_numpy().tolist() == [[1, 1, 0], [0, 0, 0]]


def test_dataset_item_and_pos_weight(tmp_path):
    for i in range(4):
        Image.fromarray(np.random.default_rng(i).integers(0, 255, (80, 80), dtype=np.uint8)).save(tmp_path / f"{i}.png")
    df = pd.DataFrame({"Path": [f"{i}.png" for i in range(4)], "A": [1, 0, 0, 0], "B": [1, 1, -1, 0]})
    ds = ChestXrayDataset(df, str(tmp_path), ["A", "B"], "ignore", build_transforms({"image_size": 64}, train=True))
    item = ds[2]
    assert item["image"].shape == (3, 64, 64)
    assert item["mask"].tolist() == [1.0, 0.0]
    np.testing.assert_allclose(ds.pos_weight().numpy(), [3.0, 0.5])
