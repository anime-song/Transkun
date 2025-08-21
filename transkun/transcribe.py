import argparse

from .Data import writeMidi
import torch
import moduleconf
import soundfile as sf
from pathlib import Path
from .model import TransKun
import librosa


def read_audio(path: str, sampling_rate: int):
    y, _ = librosa.load(path, sr=sampling_rate, mono=False)

    if y.ndim == 2:
        y = y.T

    return y


def main():
    import pkg_resources

    defaultWeight = pkg_resources.resource_filename(__name__, "pretrained/2.0.pt")
    defaultConf = pkg_resources.resource_filename(__name__, "pretrained/2.0.conf")

    argumentParser = argparse.ArgumentParser()
    argumentParser.add_argument("audioPath", help="path to the input audio file")
    argumentParser.add_argument("outPath", help="path to the output MIDI file")
    argumentParser.add_argument("--weight", default=defaultWeight, help="path to the pretrained weight")
    argumentParser.add_argument("--conf", default=defaultConf, help="path to the model conf")
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
        "--use_state_dict",
        action="store_true",
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
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    if "best_state_dict" not in checkpoint or checkpoint["best_state_dict"] is None or args.use_state_dict:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint["best_state_dict"], strict=False)

    model.eval()
    model = model.to(device, memory_format=torch.channels_last)

    audioPath = args.audioPath
    outPath = args.outPath

    torch.set_grad_enabled(False)

    audio = read_audio(audioPath, sampling_rate=model.fs)

    x = torch.from_numpy(audio).to(device)

    notesEst = model.transcribe(
        x,
        stepInSecond=args.segmentHopSize,
        segmentSizeInSecond=args.segmentSize,
        discardSecondHalf=False,
    )

    outputMidi = writeMidi(notesEst)
    outputMidi.write(outPath)


if __name__ == "__main__":
    main()
