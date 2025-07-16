import argparse

from .Data import writeMidi
import torch
import moduleconf
import numpy as np
import soundfile as sf
from pathlib import Path
from .utils import computeParamSize
from .model import TransKun


def readAudio(path, normalize=True):
    import pydub

    audio = pydub.AudioSegment.from_mp3(path)
    y = np.array(audio.get_array_of_samples())
    y = y.reshape(-1, audio.channels)
    if normalize:
        y = np.float32(y) / 2**15
    return audio.frame_rate, y


def main():
    import pkg_resources

    defaultWeight = pkg_resources.resource_filename(__name__, "pretrained/2.0.pt")
    defaultConf = pkg_resources.resource_filename(__name__, "pretrained/2.0.conf")

    argumentParser = argparse.ArgumentParser()
    argumentParser.add_argument("audioPath", help="path to the input audio file")
    argumentParser.add_argument("outPath", help="path to the output MIDI file")
    argumentParser.add_argument(
        "--weight", default=defaultWeight, help="path to the pretrained weight"
    )
    argumentParser.add_argument(
        "--conf", default=defaultConf, help="path to the model conf"
    )
    argumentParser.add_argument(
        "--device",
        default="cpu",
        nargs="?",
        help=" The device used to perform the most computations (optional), DEFAULT: cpu",
    )
    argumentParser.add_argument(
        "--segmentHopSize",
        type=float,
        required=False,
        help=" The segment hopsize for processing the entire audio file (s), DEFAULT: the value defined in model conf",
    )
    argumentParser.add_argument(
        "--segmentSize",
        type=float,
        required=False,
        help=" The segment size for processing the entire audio file (s), DEFAULT: the value defined in model conf",
    )
    argumentParser.add_argument(
        "--outWav",
        default=None,
        help=" path to the separated‑piano WAV file (optional). "
        "省略時は <audioPath の親フォルダ>/<stem>_piano.wav",
    )

    args = argumentParser.parse_args()

    path = args.weight
    device = args.device

    # TODO fix the conf
    confPath = args.conf

    confManager = moduleconf.parseFromFile(confPath)
    transkun: TransKun = confManager["Model"].module.TransKun
    conf = confManager["Model"].config

    checkpoint = torch.load(path, map_location=device)

    model: TransKun = transkun(conf=conf).to(device)
    model = torch.compile(model, mode="max-autotune")

    if "best_state_dict" not in checkpoint or checkpoint["best_state_dict"] is None:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint["best_state_dict"], strict=False)

    model.eval()

    audioPath = args.audioPath
    outPath = args.outPath
    if args.outWav is not None:
        write_wav_path = Path(args.outWav)
    else:
        audio_path_obj = Path(audioPath)
        write_wav_path = audio_path_obj.with_name(
            f"{audio_path_obj.stem}_separated.wav"
        )
    torch.set_grad_enabled(False)

    fs, audio = readAudio(audioPath)

    if fs != model.fs:
        import soxr

        audio = soxr.resample(
            audio,  # 1D(mono) or 2D(frames, channels) array input
            fs,  # input samplerate
            model.fs,  # target samplerate
        )

    x = torch.from_numpy(audio).to(device)

    notesEst, recon_audio = model.transcribe(
        x,
        stepInSecond=args.segmentHopSize,
        segmentSizeInSecond=args.segmentSize,
        discardSecondHalf=False,
    )

    outputMidi = writeMidi(notesEst)
    outputMidi.write(outPath)

    recon_np = recon_audio.transpose(0, 1).cpu().numpy()  # (T, C)
    sf.write(write_wav_path, recon_np, conf.fs, subtype="PCM_16")


if __name__ == "__main__":
    main()
