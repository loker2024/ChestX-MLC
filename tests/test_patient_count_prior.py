"""抓住零强度改变端点、符号颠倒、重复计数和格式损坏。"""
import json

import numpy as np
import pandas as pd
import pytest

from data.patient_counts import build_patient_counts
from infer.patient_count_prior import (
    PatientCountPrior, adjust_probabilities, apply_prior_frame, load_prior,
    recipe_fingerprint, save_prior, require_research_split,
)


def test_zero_strength_and_single_study_preserve_endpoints():
    probabilities = np.array([0., 1., .2, .3, .4, .5, .6, .7, .8, .9])
    np.testing.assert_array_equal(adjust_probabilities(probabilities, 20, PatientCountPrior()), probabilities)
    np.testing.assert_array_equal(adjust_probabilities(probabilities, 1, PatientCountPrior(beta=.2)), probabilities)


def test_signed_adjustment_caps_and_does_not_mutate():
    probabilities = np.full((3, 10), .5)
    result = adjust_probabilities(probabilities, np.array([1, 10, 100]), PatientCountPrior(beta=.2))
    np.testing.assert_array_equal(result[0], probabilities[0])
    np.testing.assert_allclose(result[1], result[2])
    assert (result[1, :9] > .5).all()
    assert result[1, 9] < .5
    np.testing.assert_array_equal(probabilities, np.full((3, 10), .5))


@pytest.mark.parametrize("counts", [0, -1, 1.5, float("nan"), True])
def test_invalid_counts_fail_even_when_disabled(counts):
    with pytest.raises(ValueError):
        adjust_probabilities(np.full(10, .5), counts, PatientCountPrior())


def test_frame_aligns_ids_and_retains_labels():
    frame = pd.DataFrame({"Study_id": ["03", "01", "02"], "label_1": [0, 1, 0],
                          **{f"p{c}": [.5] * 3 for c in range(1, 11)}})
    counts = build_patient_counts(pd.DataFrame({"Subject_id": ["a", "a", "b"],
                                               "Study_id": ["01", "02", "03"]}))
    result = apply_prior_frame(frame, counts, PatientCountPrior(beta=.1))
    assert result["Study_id"].tolist() == ["03", "01", "02"]
    assert result["label_1"].tolist() == [0, 1, 0]
    assert result.loc[0, "p1"] == .5
    assert result.loc[1, "p1"] > .5
    assert frame.loc[1, "p1"] == .5


def test_prior_roundtrip_and_reject_unknown_fields(tmp_path):
    path = tmp_path / "prior.json"
    save_prior(PatientCountPrior(beta=.1), path)
    assert load_prior(path).beta == .1
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["cap"] = 9
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_prior(path)


def test_test_split_is_not_enabled_by_editing_status():
    with pytest.raises(ValueError, match="test"):
        require_research_split("test")


def test_recipe_fingerprint_tracks_model_and_fusion_but_not_data_paths():
    class FakeConfig:
        def __init__(self, root):
            self.root = root

        def to_dict(self):
            return {"model": {"name": "tiny"}, "image": {"size": 320},
                    "train": {"epochs": 5}, "seed": 42, "data": {"train": self.root}}

    a, b = FakeConfig("first"), FakeConfig("second")
    assert recipe_fingerprint([a]) == recipe_fingerprint([b])
    assert recipe_fingerprint([a], fusion="logit") != recipe_fingerprint([a])


def test_submission_entry_rejects_prior_before_opening_data():
    from run_infer import main
    with pytest.raises(ValueError, match='test'):
        main(['--checkpoint', '/nonexistent.pt', '--patient-count-prior', '/nonexistent.json'])


def test_ensemble_test_rejects_prior_before_opening_checkpoints():
    from run_ensemble import main
    with pytest.raises(ValueError, match='test'):
        main(['--checkpoint', '/nonexistent.pt', '--split', 'test',
              '--patient-count-prior', '/nonexistent.json'])
