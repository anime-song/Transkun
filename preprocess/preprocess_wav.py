from __future__ import annotations

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import soundfile as sf
import numpy as np


def convert_to_wav(src_path: Path, dst_path: Path, target_sr: int) -> str:
    """
    src_path を読み込み、指定 SR で WAV 保存する。
    モノラル入力の場合はチャンネルを複製してステレオ化する。
    戻り値: "48000 → 44100 Hz" などの変換ログ文字列
    """
    import soundfile as sf  # 上部の import に合わせても OK

    # 変換前 SR を取得
    original_sr = sf.info(str(src_path)).samplerate

    # mono=False で多チャンネルをそのまま取得（ここで resample）
    audio, _ = librosa.load(str(src_path), sr=target_sr, mono=False)

    if audio.ndim == 1:  # モノラル → (2, samples) に複製
        audio = np.stack([audio, audio], axis=0)

    # (channels, samples) → (samples, channels)
    audio = audio.T

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst_path), audio, target_sr)

    return f"{original_sr} → {target_sr} Hz"


def process_file(path: Path, target_sr: int, overwrite_wav: bool) -> str:
    ext = path.suffix.lower()
    try:
        if ext == ".wav":
            if overwrite_wav:
                sr_msg = convert_to_wav(path, path, target_sr)
                return f"Updated WAV  {sr_msg}  → {path}"
            else:
                converted = path.with_name(path.stem + "_converted.wav")
                sr_msg = convert_to_wav(path, converted, target_sr)
                return f"Created copy {sr_msg}  → {converted}"

        if ext == ".mp3":
            wav_dst = path.with_suffix(".wav")
            if wav_dst.exists() and not overwrite_wav:
                wav_dst = path.with_name(path.stem + "_from_mp3.wav")
            sr_msg = convert_to_wav(path, wav_dst, target_sr)
            return f"MP3 converted {sr_msg}  → {wav_dst}"

        return f"Skipped (unsupported) → {path}"

    except Exception as exc:
        return f"Failed ({path}): {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively resample WAVs to mono and convert MP3s → WAV. " "Processing is parallelised for speed."
        )
    )
    parser.add_argument("directory", type=Path, help="検索対象のルートディレクトリ")
    parser.add_argument(
        "-r",
        "--rate",
        type=int,
        default=44100,
        help="目標サンプリングレート (default: 44100)",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=mp.cpu_count(),
        help="並列ワーカー数 (default: CPU コア数)",
    )

    args = parser.parse_args()
    root_dir: Path = args.directory

    if not root_dir.is_dir():
        parser.error(f"{root_dir} は有効なディレクトリではありません")

    # 検索対象拡張子
    exts = {".wav", ".mp3"}
    files = [p for p in root_dir.rglob("*") if p.suffix.lower() in exts]

    if not files:
        print("対象ファイルが見つかりませんでした。")
        return

    print(f"Found {len(files)} files. Processing with {args.workers} worker(s)...\n")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_file, p, args.rate, True) for p in files]
        for fut in as_completed(futures):
            print(fut.result())

    print("\nDone.")


if __name__ == "__main__":
    main()
