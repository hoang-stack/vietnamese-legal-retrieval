import json

from vilegal.data import Dataset
from vilegal.experiments import Recorder, neural_stage


def test_neural_stage_scores_kaggle_runs(tmp_path):
    ds = Dataset(
        doc_ids=["d1", "d2", "d3"],
        doc_texts=["a", "b", "c"],
        queries={"q1": "x", "q2": "y"},
        qrels={"train": {}, "dev": {"q1": {"d2"}}, "test": {"q2": {"d3"}}},
    )
    runs = {
        "Model": {"dev": {"q1": ["d2", "d1"]}, "test": {"q2": ["d1", "d3"]}},
        "Model, epoch 1": {"dev": {"q1": ["d1", "d2"]}, "test": {"q2": ["d3"]}},
    }
    (tmp_path / "runs.json").write_text(json.dumps(runs))
    rec = Recorder(tmp_path / "results")
    neural_stage(ds, rec, tmp_path)
    by = {(r["system"], r["split"]): r for r in rec.rows}
    assert by[("Model", "dev")]["mrr@10"] == 1.0
    assert by[("Model", "test")]["mrr@10"] == 0.5
    assert ("Model, epoch 1", "test") not in by  # per-epoch runs are only scored on dev
    assert by[("Model, epoch 1", "dev")]["stage"] == "finetune_epochs"
