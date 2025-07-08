from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
import pretty_midi
import soundfile as sf
from tqdm import tqdm


def transpose_midi(
    input_midi: Path, output_midi: Path, semitone_shift: int, overwrite: bool
) -> None:
    """MIDI を半音シフトして保存する。"""
    if output_midi.exists() and not overwrite:
        return

    midi_data = pretty_midi.PrettyMIDI(str(input_midi))
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            note.pitch += semitone_shift
    output_midi.parent.mkdir(parents=True, exist_ok=True)
    midi_data.write(str(output_midi))


def pitch_shift_audio(
    input_audio: Path, output_audio: Path, semitone_shift: int, overwrite: bool
) -> None:
    """WAV を半音シフトして保存する（チャンネル数自動判定）。"""
    if output_audio.exists() and not overwrite:
        return

    y, sr = sf.read(str(input_audio))
    if y.ndim == 1:
        shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=semitone_shift)
    else:
        shifted = np.stack(
            [
                librosa.effects.pitch_shift(y=y[:, ch], sr=sr, n_steps=semitone_shift)
                for ch in range(y.shape[1])
            ],
            axis=1,
        )

    output_audio.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_audio), shifted, sr)


def _augment_one(
    task: Tuple[Dict, str, int, bool],
) -> Dict:  # → 戻り値は “新しいメタデータ行” を表す dict
    """
    個別タスク：
    (CSV の 1 行, dataset_dir, 半音シフト, overwrite) を受け取り
    新しい MIDI/WAV とメタデータ行を作成して dict を返す。
    """
    row_dict, dataset_dir_str, semitone_shift, overwrite = task
    dataset_dir = Path(dataset_dir_str)

    original_midi_rel = Path(row_dict["midi_filename"])
    original_wav_rel = Path(row_dict["audio_filename"])
    original_midi_path = dataset_dir / original_midi_rel
    original_wav_path = dataset_dir / original_wav_rel

    suffix = f"_pitch{semitone_shift}"
    new_midi_rel = original_midi_rel.with_name(
        original_midi_rel.stem + suffix + original_midi_rel.suffix
    )
    new_wav_rel = original_wav_rel.with_name(
        original_wav_rel.stem + suffix + original_wav_rel.suffix
    )
    new_midi_path = dataset_dir / new_midi_rel
    new_wav_path = dataset_dir / new_wav_rel

    # 実ファイル生成
    transpose_midi(original_midi_path, new_midi_path, semitone_shift, overwrite)
    pitch_shift_audio(original_wav_path, new_wav_path, semitone_shift, overwrite)

    # メタデータ行を作成
    new_row = dict(row_dict)  # shallow copy
    new_row["midi_filename"] = str(new_midi_rel)
    new_row["audio_filename"] = str(new_wav_rel)
    new_row["duration"] = librosa.get_duration(path=str(new_wav_path))
    return new_row


def build_task_list(
    metadata_df: pd.DataFrame,
    dataset_dir: Path,
    overwrite: bool,
    semitone_range: range,
) -> List[Tuple[Dict, str, int, bool]]:
    """train split のみ抽出し、全タスクのリストを作成する。"""
    tasks: List[Tuple[Dict, str, int, bool]] = []
    for _, meta_row in metadata_df.iterrows():
        if meta_row["split"] != "train":
            continue
        for semitone in semitone_range:
            tasks.append((meta_row.to_dict(), str(dataset_dir), semitone, overwrite))

    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Maestro train split に対して MIDI/WAV の半音シフト拡張を並列実行し、"
            "新しいメタデータ CSV を生成します。"
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Maestro データセットのルートディレクトリ",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        required=True,
        help="元メタデータ CSV パス",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="出力メタデータ CSV パス",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count(),
        help="並列プロセス数 (デフォルト: 論理 CPU 数)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存の移調済みファイルを再生成する",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")

    metadata_df = pd.read_csv(args.metadata_csv)

    task_list = build_task_list(
        metadata_df,
        dataset_dir=args.dataset_dir,
        overwrite=args.overwrite,
        semitone_range=range(-5, 7),  # -5 〜 +6 半音
    )

    augmented_rows: List[Dict] = []

    # プロセスプール実行
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(_augment_one, task) for task in task_list]

        # 進捗バー付きで回収
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Augmenting",
            unit="file",
        ):
            augmented_rows.append(future.result())

    # 結合して保存
    combined_df = pd.concat(
        [metadata_df, pd.DataFrame(augmented_rows)], ignore_index=True
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(args.output_csv, index=False)
    print(
        f"{len(augmented_rows)} ファイルを拡張し、CSV を保存しました → {args.output_csv}"
    )


if __name__ == "__main__":
    main()
