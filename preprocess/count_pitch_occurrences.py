import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pretty_midi as pm


def collect_midi_paths(root: Path) -> list[Path]:
    """再帰的に .mid / .midi を検索して一覧を返す"""
    return sorted(root.rglob("*.mid*"))  # *.mid と *.midi の両方をカバー


def count_pitches(
    midi_path: Path, ignore_drums: bool = True
) -> tuple[Counter, Counter]:
    """
    1 つの MIDI について、ピッチごとの:
        - 発音回数
        - 発音継続時間（秒）
    をカウントして返す
    """
    midi = pm.PrettyMIDI(str(midi_path))
    count_by_pitch: Counter[int] = Counter()
    duration_by_pitch: Counter[int] = Counter()

    for inst in midi.instruments:
        if ignore_drums and inst.is_drum:
            continue
        for note in inst.notes:
            pitch = note.pitch  # 0–127
            count_by_pitch[pitch] += 1
            duration_by_pitch[pitch] += note.end - note.start

    return count_by_pitch, duration_by_pitch


PIANO_MIN, PIANO_MAX = 21, 108  # 追加: 88鍵の範囲


def aggregate(
    directory: Path, ignore_drums: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """ディレクトリ配下のすべての MIDI を集計"""
    paths = collect_midi_paths(directory)
    if not paths:
        raise RuntimeError("MIDI ファイルが見つかりませんでした。")

    total_counts = Counter()
    total_durations = Counter()

    for path in paths:
        counts, durations = count_pitches(path, ignore_drums)
        total_counts.update(counts)
        for pitch, dur in durations.items():
            total_durations[pitch] += dur

    counts = np.array(
        [total_counts[p] for p in range(PIANO_MIN, PIANO_MAX + 1)], dtype=int
    )
    durations = np.array(
        [total_durations[p] for p in range(PIANO_MIN, PIANO_MAX + 1)], dtype=float
    )

    return counts, durations


def plot_histogram(values: np.ndarray, title: str, ylabel: str):
    num_keys = PIANO_MAX - PIANO_MIN + 1  # => 88
    x_labels = np.arange(1, num_keys + 1)  # 鍵盤番号 1–88
    plt.figure(figsize=(14, 4))
    plt.bar(x_labels, values, width=0.8)
    plt.title(title)
    plt.xlabel("Piano Key Number (1=A0 … 88=C8)")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="MIDI pitch distribution analyser")
    parser.add_argument(
        "directory", type=str, help="Path to the directory containing MIDI files"
    )
    parser.add_argument(
        "--keep-drums", action="store_true", help="Include drum tracks (channel 10)"
    )
    args = parser.parse_args()

    directory = Path(args.directory).expanduser().resolve()
    counts, durations = aggregate(directory, ignore_drums=not args.keep_drums)

    plot_histogram(counts, "Pitch Occurrence Count", "Note Count")
    plot_histogram(durations, "Pitch Total Duration", "Duration (seconds)")


if __name__ == "__main__":
    main()
