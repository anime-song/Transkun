from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import soundfile as sf


def audio_duration_sec(wav_path: Path) -> float:
    """WAV 長 [s] をメタデータのみで取得（高速）"""
    with sf.SoundFile(str(wav_path)) as f:
        return f.frames / f.samplerate


def decide_split(stems: List[str]) -> Dict[str, str]:
    """stem → split(train/test/validation) 連番で決定"""
    stems_sorted = sorted(stems)  # 001, 002, 003, ...
    n_total = len(stems_sorted)
    n_train = int(n_total * 0.95)
    n_test = int(n_total * 0.00)

    mapping: Dict[str, str] = {}
    for idx, stem in enumerate(stems_sorted):
        if idx < n_train:
            mapping[stem] = "train"
        elif idx < n_train + n_test:
            mapping[stem] = "test"
        else:
            mapping[stem] = "validation"
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "original_other / transcribe / piano_* を組み合わせ、"
            "バリアントごとに重複行を持つ 1 枚の CSV を作成します。"
        )
    )
    parser.add_argument("base_dir", type=Path, help="データセットのルートディレクトリ")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("dataset.csv"),
        help="出力 CSV パス (default: ./dataset.csv)",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    other_dir = base_dir / "original_other"
    midi_dir = base_dir / "transcribe"
    piano_dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("piano_")]
    )

    if not piano_dirs:
        raise SystemExit("piano_* フォルダが見つかりません。")

    # stem 一覧取得 → split 決定
    stems = ["_".join(p.stem.split("_")[:-1]) for p in other_dir.glob("*_other.wav")]
    print(stems)
    if not stems:
        raise SystemExit("original_other に *_other.wav が見つかりません。")
    stem2split = decide_split(stems)

    # レコード生成
    records: list[dict] = []
    for stem in stems:
        other_path = other_dir / f"{stem}_other.wav"
        midi_path = midi_dir / f"{stem}_piano.mid"

        # other / midi が存在しない場合はスキップ
        if not (other_path.exists() and midi_path.exists()):
            print(f"[WARN] 欠損によりスキップ: {stem}")
            continue

        # duration は other の長さ -1 秒（下限 0）
        duration = max(audio_duration_sec(other_path) - 1.0, 0.0)

        # piano_* の数だけ行を追加
        for piano_dir in piano_dirs:
            piano_path = piano_dir / f"{stem}_piano.wav"
            if not piano_path.exists():
                print(f"[WARN] {piano_path} が無いためスキップ")
                continue

            records.append(
                dict(
                    split=stem2split[stem],
                    midi_filename=midi_path.relative_to(base_dir).as_posix(),
                    audio_filename=piano_path.relative_to(base_dir).as_posix(),
                    other_filename=other_path.relative_to(base_dir).as_posix(),
                    duration=duration,
                )
            )

    if not records:
        raise SystemExit("有効なペアが見つかりませんでした。")

    # CSV 出力
    df = pd.DataFrame(
        records,
        columns=[
            "split",
            "midi_filename",
            "audio_filename",
            "other_filename",
            "duration",
        ],
    )
    df.to_csv(args.output, index=False)
    print(f"CSV を保存しました → {args.output}")


if __name__ == "__main__":
    main()
