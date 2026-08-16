from __future__ import annotations

import os
from pathlib import Path
import subprocess


def test_train_launcher_anchors_relative_resume_to_callers_directory(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    launcher = project_root / "scripts/train.sh"
    fake_python = tmp_path / "fake-python"
    capture = tmp_path / "argv.txt"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$CAPTURE\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    caller = tmp_path / "caller with spaces"
    caller.mkdir()

    env = dict(os.environ, PYTHON=str(fake_python), CAPTURE=str(capture))
    subprocess.run(
        [
            "bash",
            str(launcher),
            "--gpus",
            "0",
            "--resume",
            "checkpoints/latest.pt",
            "--resume-mode",
            "weights-only",
        ],
        cwd=caller,
        env=env,
        check=True,
    )

    argv = capture.read_text(encoding="utf-8").splitlines()
    resume_index = argv.index("--resume")
    assert argv[resume_index + 1] == str(caller / "checkpoints/latest.pt")
    mode_index = argv.index("--resume-mode")
    assert argv[mode_index + 1] == "weights-only"
