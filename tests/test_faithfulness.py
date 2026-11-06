import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from cxrtrust.faithfulness import deletion_insertion, energy_in_boxes, load_nih_boxes, pointing_game


class RegionModel(nn.Module):
    """Logit of class 0 = mean intensity inside a fixed square (rows/cols 8..15)."""

    def forward(self, x):
        region = x[:, :, 8:16, 8:16].mean(dim=(1, 2, 3))
        return torch.stack([12 * (region - 0.5), torch.zeros_like(region)], 1)


def _image():
    x = torch.full((1, 1, 32, 32), 0.2)
    x[..., 8:16, 8:16] = 1.0
    return x


@pytest.mark.parametrize("baseline", ["blur", "mean", "zero"])
def test_faithful_map_beats_random_map(baseline):
    model, x = RegionModel(), _image()
    good = torch.zeros(32, 32)
    good[8:16, 8:16] = 1.0
    good += 0.01 * torch.rand(32, 32)
    rand = torch.rand(32, 32, generator=torch.Generator().manual_seed(0))
    g = deletion_insertion(model, x, good, 0, steps=16, baseline=baseline)
    r = deletion_insertion(model, x, rand, 0, steps=16, baseline=baseline)
    assert g["deletion_auc"] < r["deletion_auc"]
    assert g["insertion_auc"] > r["insertion_auc"]
    assert len(g["deletion"]) == 17
    assert g["deletion"][0] == pytest.approx(g["insertion"][-1], abs=1e-6)  # both ends are the original image


def test_low_resolution_saliency_is_upsampled():
    out = deletion_insertion(RegionModel(), _image(), torch.rand(8, 8), 0, steps=4)
    assert np.isfinite(out["deletion_auc"])


def test_pointing_game_and_energy():
    sal = np.zeros((100, 100))
    sal[50, 60] = 1.0
    sal[10, 10] = 0.5
    assert pointing_game(sal, [(55, 45, 10, 10)], tolerance=0)
    assert not pointing_game(sal, [(0, 0, 20, 20)], tolerance=0)
    assert pointing_game(sal, [(70, 50, 5, 5)], tolerance=15)
    assert energy_in_boxes(sal, [(55, 45, 10, 10)]) == pytest.approx(1 / 1.5)
    assert energy_in_boxes(np.zeros((4, 4)), [(0, 0, 2, 2)]) == 0.0


def test_load_nih_boxes(tmp_path):
    csv = tmp_path / "BBox_List_2017.csv"
    csv.write_text(
        "Image Index,Finding Label,Bbox [x,y,w,h],,,\n"
        "00013118_008.png,Atelectasis,225.08,547.02,86.78,79.19\n"
        "00013118_008.png,Nodule,100,100,10,20\n"
        "00014716_007.png,Effusion,686.1,272.0,110.0,322.0\n"
    )
    assert len(pd.read_csv(csv)) == 3
    boxes = load_nih_boxes(str(csv), image_size=512)
    assert len(boxes["00013118_008.png"]) == 2
    label, (x, y, w, h) = boxes["00013118_008.png"][1]
    assert label == "Nodule" and (x, y, w, h) == pytest.approx((50, 50, 5, 10))
