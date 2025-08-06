from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple

import pretty_midi
from tqdm import tqdm


def collect_midi_files(root: Path) -> List[Path]:
    """再帰的に .mid / .midi を集める"""
    return sorted(path for path in root.rglob("*")
                  if path.suffix.lower() in {".mid", ".midi"})


def longest_silence_seconds(midi: pretty_midi.PrettyMIDI) -> float:
    """MIDI 全体で最長の沈黙秒数を計算"""
    # (開始・終了) イベントをすべて集める
    events: List[Tuple[float, int]] = []  # (time, +1/-1) +1: note_on, -1: note_off
    for instrument in midi.instruments:
        for note in instrument.notes:
            events.append((note.start, +1))
            events.append((note.end, -1))

    if not events:
        # ノートが 1 つも無い場合は全長が沈黙
        return midi.get_end_time()

    # タイムライン先頭/末尾も沈黙判定に含める
    events.append((0.0, 0))                         # 曲頭
    events.append((midi.get_end_time(), 0))         # 曲末
    events.sort(key=lambda x: x[0])                 # 時刻で昇順ソート

    active_notes = 0
    last_time = events[0][0]
    max_silence = 0.0

    for time, delta in events:
        # active_notes == 0 で経過した秒数 = 沈黙
        if active_notes == 0:
            silence = time - last_time
            max_silence = max(max_silence, silence)
        active_notes += delta
        last_time = time

    return max_silence


def main(folder: Path, threshold_sec: float) -> None:
    midi_paths = collect_midi_files(folder)
    if not midi_paths:
        print("指定フォルダに MIDI ファイルが見つかりませんでした。")
        return

    print(f"[INFO] {len(midi_paths)} 個の MIDI を解析中...")
    hits: List[Tuple[Path, float]] = []

    for path in tqdm(midi_paths):
        try:
            midi = pretty_midi.PrettyMIDI(str(path))
            silence = longest_silence_seconds(midi)
            if silence >= threshold_sec:
                hits.append((path, silence))
        except Exception as e:
            print(f"[WARN] 解析失敗: {path} ({e})")

    if hits:
        print("\n=== 8 秒以上沈黙があるファイル ===")
        for path, silence in hits:
            print(f"{path}  :  最長沈黙 {silence:.2f} s")
    else:
        print("\n条件に合致するファイルはありませんでした。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIDI の長い沈黙検出")
    parser.add_argument("folder", type=str, help="検索対象フォルダ (再帰的に走査)")
    parser.add_argument("--threshold", type=float, default=8.0,
                        help="沈黙判定の閾値 [秒] (デフォルト: 8)")
    args = parser.parse_args()

    main(Path(args.folder).expanduser(), args.threshold)