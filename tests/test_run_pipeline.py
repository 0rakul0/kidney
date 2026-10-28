from pathlib import Path

import run_pipeline


def test_train_script_is_resolved_at_repository_root():
    assert run_pipeline.PIPELINE_SCRIPTS["train"] == run_pipeline.PROJECT_ROOT / "train_renal_classifier.py"
    assert run_pipeline.PIPELINE_SCRIPTS["train"].exists()


def test_engineering_scripts_remain_in_engineering_directory():
    for command in ("init-labels", "extract"):
        assert run_pipeline.PIPELINE_SCRIPTS[command].parent == Path(run_pipeline.PROJECT_ROOT) / "engenharia_dataset"
