"""Regenerate the public training notebook. / 重新生成公开训练 Notebook。"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    """Report the canonical notebook location. / 输出规范 Notebook 的位置。"""
    target = Path(__file__).resolve().parents[1] / "06_h3_next_hour_onnx_training_and_go_validation.ipynb"
    notebook = json.loads(target.read_text(encoding="utf-8"))
    target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
