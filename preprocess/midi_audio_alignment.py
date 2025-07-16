from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import pretty_midi

try:
    import librosa
except ImportError:
    librosa = None


def power_to_db(power: np.ndarray, amin: float = 1e-20) -> np.ndarray:
    power = np.maximum(power, amin)
    return 10.0 * np.log10(power)


def rms_to_dbfs(rms: np.ndarray, peak: float) -> np.ndarray:
    """RMS をピーク基準の dBFS に変換 (+0 dBFS = peak)。

    一般的な dBFS はフルスケール (1.0) 基準だが、ここでは波形ピークを 0 dBFS とみなす簡易版。
    """
    peak = max(peak, 1e-12)
    # RMS -> power 比
    rel_power = (rms / peak) ** 2
    return power_to_db(rel_power)


def detect_audio_onset_rms(
    wav_path: Path,
    search_seconds: float = 10.0,
    frame_size: int = 1024,
    hop_size: int = 512,
    relative_db_drop: float = 40.0,
    absolute_dbfs: float = -60.0,
) -> float:
    """冒頭無音をスキップし音の立ち上がり秒を RMS で推定 (シンプル)。

    手順:
        1) 先頭 search_seconds 分だけ読み込み (短縮で高速化)。
        2) ステレオ等は平均でモノ化。
        3) フレーム RMS 計算。
        4) グローバルピーク RMS を基準に `peak_db - relative_db_drop` を閾値候補に。
        5) ただし最低ラインとして `absolute_dbfs` を上書き (max を取る)。
        6) 初めて RMS_dB が閾値を超えるフレームを onset とする。

    戻り値: onset_sec。該当なしなら 0.0 を返す。
    """
    if not wav_path.exists():
        raise FileNotFoundError(wav_path)

    # 音声読み込み: search_seconds だけ (sf.read はフレーム指定可能)
    info = sf.info(str(wav_path))
    sr = info.samplerate
    max_frames = int(search_seconds * sr)

    # 読み込むフレーム数 (ファイル長未満)
    frames_to_read = min(max_frames, info.frames)
    audio, _ = sf.read(
        str(wav_path), start=0, stop=frames_to_read, dtype="float32", always_2d=True
    )

    # モノ化 (平均)
    audio_mono = audio.mean(axis=1)

    # フレーム分割
    if len(audio_mono) < frame_size:
        # 短すぎる場合は全体 RMS を返す (0秒扱い)
        return 0.0

    # オーバーラップ RMS
    n_frames = 1 + (len(audio_mono) - frame_size) // hop_size
    rms_values = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_size
        end = start + frame_size
        segment = audio_mono[start:end]
        rms_values[i] = np.sqrt(np.mean(segment**2, dtype=np.float64))

    peak_rms = float(rms_values.max(initial=1e-12))
    rms_db = rms_to_dbfs(rms_values, peak=peak_rms)  # peak を 0 dBFS とする相対 dB

    threshold_db = max(
        -relative_db_drop, absolute_dbfs
    )  # 例: max(-40, -60) -> -40 dBFS

    # 初閾値超過フレーム
    above = np.where(rms_db >= threshold_db)[0]
    if above.size == 0:
        return 0.0

    first_frame_index = int(above[0])
    onset_sample = first_frame_index * hop_size
    onset_sec = onset_sample / sr
    return onset_sec


def detect_audio_onset_librosa(
    wav_path: Path,
    search_seconds: float = 10.0,
) -> Optional[float]:
    """librosa の onset_strength / onset_detect を用いた補助推定。

    --librosa-onset 指定時に呼び出す。librosa 未インストールなら None。
    """
    if librosa is None:
        return None
    if not wav_path.exists():
        raise FileNotFoundError(wav_path)

    # librosa.load で search_seconds 制限
    y, sr = librosa.load(str(wav_path), mono=True, duration=search_seconds)
    if y.size == 0:
        return 0.0

    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, units="frames", backtrack=True
    )
    if onset_frames.size == 0:
        return 0.0

    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return float(onset_times[0])


def get_midi_first_note_sec(midi_path: Path) -> float:
    """pretty_midi で最初のノート開始秒を取得。ノートがない場合 0.0。"""
    if not midi_path.exists():
        raise FileNotFoundError(midi_path)
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    first = min((n.start for inst in pm.instruments for n in inst.notes), default=0.0)
    return float(first)


def list_files_with_ext(root: Path, exts: List[str], recursive: bool) -> List[Path]:
    """拡張子リストでファイル列挙 (小文字比較)。"""
    exts = [e.lower().lstrip(".") for e in exts]
    pattern = "**/*" if recursive else "*"
    files = []
    for p in root.glob(pattern):
        if p.is_file() and p.suffix.lower().lstrip(".") in exts:
            files.append(p)
    return files


def build_stem_map(paths: List[Path]) -> Dict[str, Path]:
    """stem -> path の辞書。重複 stem がある場合は最後に見つかったもので上書き (警告は別途)。"""
    m: Dict[str, Path] = {}
    for p in paths:
        stem = p.stem  # 拡張子除くファイル名
        if stem in m:
            print(
                f"[WARN] Duplicate stem '{stem}' detected. Overwriting previous: {m[stem]} -> {p}"
            )
        m[stem] = p
    return m


def pair_midi_wav(
    midi_dir: Path,
    wav_dir: Path,
    midi_exts: List[str],
    audio_exts: List[str],
    recursive: bool = False,
) -> List[Tuple[Path, Path]]:
    """stem 一致で MIDI / WAV をペアリング。"""
    midi_files = list_files_with_ext(midi_dir, midi_exts, recursive)
    audio_files = list_files_with_ext(wav_dir, audio_exts, recursive)

    midi_map = build_stem_map(midi_files)
    audio_map = build_stem_map(audio_files)

    common_stems = sorted(set(midi_map.keys()) & set(audio_map.keys()))
    if not common_stems:
        print("[WARN] No matching stems between MIDI and audio directories.")
        return []

    pairs: List[Tuple[Path, Path]] = []
    for stem in common_stems:
        pairs.append((midi_map[stem], audio_map[stem]))
    return pairs


def process_pairs(
    pairs: List[Tuple[Path, Path]],
    tolerance_ms: float,
    relative_db_drop: float,
    absolute_dbfs: float,
    search_seconds: float,
    frame_size: int,
    hop_size: int,
    use_librosa_onset: bool = False,
) -> pd.DataFrame:
    records = []

    for midi_path, wav_path in pairs:
        try:
            midi_first = get_midi_first_note_sec(midi_path)
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] MIDI read failed: {midi_path}: {e}")
            midi_first = 0.0

        # RMS ベース
        try:
            onset_rms = detect_audio_onset_rms(
                wav_path,
                search_seconds=search_seconds,
                frame_size=frame_size,
                hop_size=hop_size,
                relative_db_drop=relative_db_drop,
                absolute_dbfs=absolute_dbfs,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] WAV read/detect failed: {wav_path}: {e}")
            onset_rms = 0.0

        onset_final = onset_rms

        # librosa 補助
        if use_librosa_onset:
            onset_lb = detect_audio_onset_librosa(
                wav_path, search_seconds=search_seconds
            )
            if onset_lb is not None:
                # より早い方を採用 (早期オンセット優先)
                onset_final = (
                    min(onset_final, onset_lb) if onset_final > 0 else onset_lb
                )

        offset_sec = onset_final - midi_first
        offset_ms = offset_sec * 1000.0
        within = abs(offset_ms) <= tolerance_ms

        print(
            f"{wav_path.name:30s}  MIDI:{midi_first:8.3f}s  WAV:{onset_final:8.3f}s  "
            f"Δ:{offset_sec:8.3f}s ({offset_ms:8.1f} ms)  within={within}"
        )

        records.append(
            {
                "midi_path": str(midi_path),
                "wav_path": str(wav_path),
                "midi_first_note_sec": midi_first,
                "audio_onset_sec": onset_final,
                "onset_offset_sec": offset_sec,
                "onset_offset_ms": offset_ms,
                "within_tolerance": within,
            }
        )

    return pd.DataFrame.from_records(records)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MIDI ⇔ WAV 先頭ズレ検出 (stem マッチ)")
    p.add_argument(
        "--midi-dir", type=Path, required=True, help="MIDI ファイルを含むディレクトリ"
    )
    p.add_argument(
        "--wav-dir",
        type=Path,
        required=True,
        help="WAV (オーディオ) ファイルを含むディレクトリ",
    )
    p.add_argument(
        "--output-csv", type=Path, default=None, help="結果を書き出す CSV パス"
    )

    p.add_argument("--recursive", action="store_true", help="サブフォルダも探索")
    p.add_argument(
        "--audio-ext",
        type=str,
        default="wav",
        help="カンマ区切りオーディオ拡張子 (例: 'wav,flac,aiff')",
    )

    p.add_argument(
        "--tolerance-ms", type=float, default=20.0, help="ズレ許容閾値 (±ms)"
    )
    p.add_argument(
        "--relative-db-drop",
        type=float,
        default=40.0,
        help="ピークから何 dB 下を音開始とみなすか",
    )
    p.add_argument("--absolute-dbfs", type=float, default=-60.0, help="最低ライン dBFS")
    p.add_argument(
        "--search-seconds", type=float, default=30.0, help="先頭から解析する最大秒数"
    )
    p.add_argument("--frame-size", type=int, default=1024, help="RMS フレーム長")
    p.add_argument("--hop-size", type=int, default=512, help="RMS ホップ")
    p.add_argument(
        "--librosa-onset",
        action="store_true",
        help="librosa onset 検出も併用 (早い方を採用)",
    )

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    midi_dir: Path = args.midi_dir
    wav_dir: Path = args.wav_dir
    output_csv: Optional[Path] = args.output_csv

    if not midi_dir.is_dir():
        print(f"[ERROR] MIDI dir not found: {midi_dir}")
        return 1
    if not wav_dir.is_dir():
        print(f"[ERROR] WAV dir not found: {wav_dir}")
        return 1

    audio_exts = [e.strip() for e in args.audio_ext.split(",") if e.strip()]
    if not audio_exts:
        audio_exts = ["wav"]

    pairs = pair_midi_wav(
        midi_dir,
        wav_dir,
        midi_exts=["mid", "midi"],
        audio_exts=audio_exts,
        recursive=args.recursive,
    )

    if not pairs:
        print("[WARN] No pairs to process.")
        return 0

    df = process_pairs(
        pairs,
        tolerance_ms=args.tolerance_ms,
        relative_db_drop=args.relative_db_drop,
        absolute_dbfs=args.absolute_dbfs,
        search_seconds=args.search_seconds,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        use_librosa_onset=args.librosa_onset,
    )

    if output_csv is not None:
        try:
            df.to_csv(output_csv, index=False)
            print(f"[INFO] Saved results -> {output_csv}")
        except Exception as e:
            print(f"[ERROR] Failed to save CSV: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
