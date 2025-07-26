import math
import copy
import numpy as np
from pathlib import Path
import os
import wave

import pretty_midi
import pickle
import torch
import random
from collections import defaultdict
import csv
from typing import Tuple


# a local definition of the midi note object
class Note:
    def __init__(self, start, end, pitch, velocity, hasOnset=True, hasOffset=True):
        self.start = start
        self.end = end
        self.pitch = pitch
        self.velocity = velocity
        self.hasOnset = hasOnset
        self.hasOffset = hasOffset

    def __repr__(self):
        return str(self.__dict__)


def parseControlChangeSwitch(ccSeq, controlNumber, onThreshold=64, endT=None):
    runningStatus = False

    seqEvent = []

    currentEvent = None
    currentStatus = False

    time = 0

    for c in ccSeq:
        if c.number == controlNumber:
            time = c.time
            if c.value >= onThreshold:
                currentStatus = True
            else:
                currentStatus = False

        if runningStatus != currentStatus:
            if currentStatus == True:
                # use negative number as pitch for the control change event
                # the velocity of a pedal is normalized to 0-1, where values smaller than off is cut off
                currentEvent = Note(time, None, -controlNumber, 127)
            else:
                currentEvent.end = time
                seqEvent.append(currentEvent)

        runningStatus = currentStatus

    if runningStatus and endT is not None:
        # process the case where the state is not closed off at the end
        # print("Warning: running status {} not closed at the end".format(controlNumber));
        currentEvent.end = max(endT, time)
        if currentEvent.end > currentEvent.start:
            seqEvent.append(currentEvent)

    return seqEvent


def parseEventAll(
    notesList,
    ccList,
    supportedCC=[64, 66, 67],
    extendSustainPedal=True,
    pedal_ext_offset=0.0,
):
    # CC 64: sustain
    # CC 66: sostenuto
    # CC 67: una conda
    # normalize all velocity of notes

    notesList = [Note(**n.__dict__) for n in notesList]
    notesList.sort(key=lambda x: (x.start, x.end, x.pitch))

    for n in notesList:
        assert n.start < n.end

    # get the ending time of the last note event for the missing off event at the boundary
    lastT = max([n.end for n in notesList])

    if extendSustainPedal:
        # currently ignore cc 66
        sustainEvents = parseControlChangeSwitch(ccList, controlNumber=64, endT=lastT)
        sustainEvents.sort(key=lambda x: (x.start, x.end, x.pitch))

        if pedal_ext_offset != 0.0:
            for n in sustainEvents:
                n.start += pedal_ext_offset
                n.end += pedal_ext_offset

        notesList = extendPedal(notesList, sustainEvents)

    else:
        # remove overlappings, als remove n.start>=n.end
        notesList = resolveOverlapping(notesList)
    validateNotes(notesList)

    eventSeqs = [notesList]
    # parse CC 64 for

    for ccNum in supportedCC:
        ccSeq = parseControlChangeSwitch(ccList, controlNumber=ccNum, endT=lastT)
        eventSeqs.append(ccSeq)

    events = sum(eventSeqs, [])

    # sort all events by the beginning
    events.sort(key=lambda x: (x.start, x.end, x.pitch))

    return events


def extendPedal(note_events, pedal_events):
    note_events.sort(key=lambda x: (x.start, x.end, x.pitch))
    pedal_events.sort(key=lambda x: (x.start, x.end, x.pitch))
    ex_note_events = []

    idx = 0

    buffer_dict = {}
    nIn = len(note_events)

    for note_event in note_events:
        midi_note = note_event.pitch
        if midi_note in buffer_dict.keys():
            _idx = buffer_dict[midi_note]
            if ex_note_events[_idx].end > note_event.start:
                ex_note_events[_idx].end = note_event.start

        for curPedal in pedal_events:
            if note_event.end < curPedal.end and note_event.end > curPedal.start:
                note_event.end = curPedal.end

        buffer_dict[midi_note] = idx
        idx += 1
        ex_note_events.append(note_event)

    # print("haha")
    ex_note_events.sort(key=lambda x: (x.start, x.end, x.pitch))

    nOut = len(ex_note_events)
    assert nOut == nIn

    ex_note_events = resolveOverlapping(ex_note_events)
    validateNotes(ex_note_events)
    return ex_note_events


def resolveOverlapping(note_events):
    note_events.sort(key=lambda x: (x.start, x.end, x.pitch))

    ex_note_events = []

    idx = 0

    buffer_dict = {}

    for note_event in note_events:
        midi_note = note_event.pitch
        # note_event.end = max(note_event.start+1e-5, note_event.end)
        # note_event.end = max(note_event.start+1e-5, note_event.end)

        if midi_note in buffer_dict.keys():
            _idx = buffer_dict[midi_note]
            if ex_note_events[_idx].end > note_event.start:
                ex_note_events[_idx].end = note_event.start

        buffer_dict[midi_note] = idx
        idx += 1

        ex_note_events.append(note_event)

    ex_note_events.sort(key=lambda x: (x.start, x.end, x.pitch))

    # else:
    # print("overlappingOnsetOffset", note_event)

    # remove all notes that has start == end
    n1 = len(ex_note_events)
    error_notes = [n for n in ex_note_events if not n.start < n.end]
    ex_note_events = [n for n in ex_note_events if n.start < n.end]
    n2 = len(ex_note_events)
    if n1 != n2:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(error_notes)

    validateNotes(ex_note_events)
    return ex_note_events


def validateNotes(notes):
    pitches = defaultdict(list)
    for n in notes:
        if len(pitches[n.pitch]) > 0:
            nPrev = pitches[n.pitch][-1]
            assert n.start >= nPrev.end, str(n) + str(nPrev)

        assert n.start < n.end, n

        pitches[n.pitch].append(n)


def createIndexEvents(eventList):
    from ncls import FNCLS

    # internally uses ncls package
    starts = np.array([_.start for _ in eventList])
    ends = np.array([_.end for _ in eventList])

    index = FNCLS(starts, ends, np.arange(len(eventList)))

    return index


def querySingleInterval(start, end, index):
    starts = np.array([start], dtype=np.double)
    ends = np.array([end], dtype=np.double)
    queryIds = np.array([0])
    r_id, r_loc = index.all_overlaps_both(starts, ends, queryIds)

    return r_loc


def parseMIDIFile(midiPath, extendSustainPedal=False, pedal_ext_offset=0.0):
    # hack for the maps dataset
    pretty_midi.pretty_midi.MAX_TICK = 1e10
    midiFile = pretty_midi.PrettyMIDI(midiPath)
    assert len(midiFile.instruments) == 1

    inst = midiFile.instruments[0]
    events = parseEventAll(
        inst.notes,
        inst.control_changes,
        extendSustainPedal=extendSustainPedal,
        pedal_ext_offset=pedal_ext_offset,
    )
    return events


def createDatasetMaestroCSV(datasetPath, datasetMetaCSVPath, extendSustainPedal=True):
    datasetMetaCSVPath = Path(datasetMetaCSVPath)

    samplesAll = []

    with datasetMetaCSVPath.open(encoding="utf-8") as f:
        # metaInfo = json.load(f)
        metaInfo = csv.DictReader(f)
        for e in metaInfo:
            print(e)
            midiPath = os.path.join(datasetPath, e["midi_filename"])
            audioPath = os.path.join(datasetPath, e["audio_filename"])

            midiFile = pretty_midi.PrettyMIDI(midiPath)
            assert len(midiFile.instruments) == 1

            inst = midiFile.instruments[0]
            if len(midiFile.instruments) > 1:
                raise Exception("contains more than one track")
            events = parseEventAll(
                inst.notes, inst.control_changes, extendSustainPedal=extendSustainPedal
            )

            with wave.open(audioPath) as f:
                fs = f.getframerate()
                nSamples = f.getnframes()
                nChannel = f.getnchannels()

            e["notes"] = events
            e["fs"] = fs
            e["nSamples"] = nSamples
            e["nChannel"] = nChannel
            samplesAll.append(e)

    return samplesAll


def readAudioSlice(audioPath, begin, end, normalize=True):
    from scipy.io import wavfile
    import scipy.io

    normalized = audioPath.replace("\\", os.sep)
    fs, data = wavfile.read(str(Path(normalized).expanduser().resolve()), mmap=True)

    b = math.floor((begin) * fs)
    e = math.floor(end * fs)
    dur = e - b
    # dur = math.ceil((end-begin)*fs)
    e = b + dur
    # print(dur)

    # handle the case where b is negative

    l = data.shape[0]

    if len(data.shape) == 1:
        data = np.stack([data, data], axis=-1)

    result = data[max(b, 0) : min(e, l), :]

    # print("-----------")
    # print(dur, l, b, e)
    # print(result.shape)

    # handle padding
    lPad = max(-b, 0)
    rPad = max(e - l, 0)

    # print(lPad,rPad)

    # print(e-b)
    # print(result.shape)

    # normalize the audio to [-1,1] accoriding to the type
    if normalize:
        tMax = (np.iinfo(result.dtype)).max
        result = np.divide(result, tMax, dtype=np.float32)

    if lPad > 0 or rPad > 0:
        result = np.pad(result, ((lPad, rPad), (0, 0)), "constant")

    return result, fs


# resolution can be as high as 32767, however it may not be supported by certain DAW
def writeMidi(notes, resolution=960):
    validateNotes(notes)
    # outputMidi = pretty_midi.PrettyMIDI(resolution=32767)
    outputMidi = pretty_midi.PrettyMIDI(resolution=resolution)

    piano_program = pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
    piano = pretty_midi.Instrument(program=piano_program)

    for note in notes:
        if note.pitch > 0:
            note = pretty_midi.Note(
                start=note.start, end=note.end, pitch=note.pitch, velocity=note.velocity
            )
            piano.notes.append(note)
        else:
            cc_on = pretty_midi.ControlChange(-note.pitch, note.velocity, note.start)
            cc_off = pretty_midi.ControlChange(-note.pitch, 0, note.end)

            piano.control_changes.append(cc_on)
            piano.control_changes.append(cc_off)

    outputMidi.instruments.append(piano)
    return outputMidi


class DatasetMaestro:
    def __init__(self, datasetPath, datasetAnnotationPicklePath):
        self.datasetPath = datasetPath
        self.datasetAnnotationPicklePath = datasetAnnotationPicklePath

        self.sample_offsets: list[int] = []
        self.durations: list[float] = []

        # 1 周だけ走査して「オフセット＋duration だけ」収集
        with open(self.datasetAnnotationPicklePath, "rb") as fp:
            while True:
                try:
                    offset = fp.tell()
                    sample = pickle.load(fp)  # 1 曲だけロード
                except EOFError:
                    break

                self.sample_offsets.append(offset)
                self.durations.append(float(sample["duration"]))
                # メモリ節約のため sample は即破棄

        print(
            f"Found {len(self.sample_offsets)} pieces in {os.path.basename(self.datasetAnnotationPicklePath)}"
        )
        totalTime = sum(self.durations)
        print("totalDuration: ", totalTime)

    def _load_piece(self, idx: int, create_index_events: bool = True) -> dict:
        """指定 idx の曲だけをロードして返す。"""
        with open(self.datasetAnnotationPicklePath, "rb") as fp:
            fp.seek(self.sample_offsets[idx])
            piece = pickle.load(fp)
        # インデックスがまだ無ければ遅延生成
        if "index" not in piece and create_index_events:
            piece["index"] = createIndexEvents(piece["notes"])
        return piece

    def __getstate__(self):
        return {
            "datasetPath": self.datasetPath,
            "datasetAnnotationPicklePath": self.datasetAnnotationPicklePath,
        }

    def __setstate__(self, d):
        datasetPath = d["datasetPath"]
        datasetAnnotationPicklePath = d["datasetAnnotationPicklePath"]
        self.__init__(datasetPath, datasetAnnotationPicklePath)

    def fetchData(
        self,
        idx,
        begin,
        end,
        audioNormalize,
        notesStrictlyContained,
        other_idx=None,
        other_begin=None,
        other_end=None,
    ):
        piece = self._load_piece(idx)

        # ランダムで他の曲のミックスを合成する
        if other_idx is not None and other_begin is not None and other_end is not None:
            other_piece = self._load_piece(
                other_idx,
                create_index_events=False,
            )
            piece["other_filename"] = other_piece["other_filename"]

        if other_begin is None or other_end is None:
            other_begin = begin
            other_end = end

        # fetch the notes in this interval
        if end < 0 and begin < 0:
            noteIndices = []
        else:
            noteIndices = querySingleInterval(
                max(begin, 0.0), max(end, 0.0), piece["index"]
            )

        notes = [piece["notes"][int(_)] for _ in noteIndices]

        if notesStrictlyContained:
            notes = [
                Note(_.start - begin, _.end - begin, _.pitch, _.velocity)
                for _ in notes
                if _.start >= begin and _.end < end
            ]

        else:
            notes = [
                Note(
                    max(_.start, begin) - begin,
                    min(_.end, end) - begin,
                    _.pitch,
                    _.velocity,
                    _.start >= begin,
                    _.end < end,
                )
                for _ in notes
            ]

        # オーディオ切り出し
        audioPath = os.path.join(self.datasetPath, piece["audio_filename"])
        audioSlice, fs = readAudioSlice(audioPath, begin, end, audioNormalize)

        other_path = os.path.join(self.datasetPath, piece["other_filename"])
        other_slice, fs = readAudioSlice(
            other_path, other_begin, other_end, audioNormalize
        )

        return notes, audioSlice, other_slice, fs


class AugmentatorAudiomentations:
    def __init__(
        self,
        sampleRate=44100,
        pitchShiftRange=(-0.2, 0.2),
        eqDBRange=(-3, 3),
        snrRange=(0, 40),
        convIRFolder=None,
        noiseFolder=None,
    ):
        from audiomentations import (
            AddGaussianSNR,
            Compose,
            PitchShift,
            ApplyImpulseResponse,
            AddBackgroundNoise,
            SevenBandParametricEQ,
            PolarityInversion,
            RoomSimulator,
        )

        transformList = [
            PitchShift(*pitchShiftRange, p=0.5),
            SevenBandParametricEQ(*eqDBRange, p=0.5),
            PolarityInversion(p=0.5),
            RoomSimulator(
                calculation_mode="rt60",
                max_order=3,
                p=0.5,
            ),
        ]

        self.transform = Compose(transformList)

        if convIRFolder is not None:
            irPath = Path(convIRFolder)
            fileList = list(irPath.glob(os.path.join("**", "*.wav")))
            self.reverb = ApplyImpulseResponse(
                fileList, p=0.5, lru_cache_size=2000, leave_length_unchanged=True
            )
            print("aug: convIR enabled")
        else:
            self.reverb = None

        transformNoiseList = []
        if noiseFolder is not None:
            noisePath = Path(noiseFolder)
            fileList = list(noisePath.glob(os.path.join("**", "*.wav")))

            noiseTrans = Compose(
                [
                    PolarityInversion(),
                    PitchShift(),
                    SevenBandParametricEQ(*eqDBRange, p=0.5),
                ]
            )

            transformNoiseList.append(
                AddBackgroundNoise(
                    fileList,
                    *snrRange,
                    p=0.7,
                    lru_cache_size=256,
                    noise_transform=noiseTrans,
                )
            )

            print("aug: noise enabled")

        transformNoiseList.append(
            AddGaussianSNR(min_snr_db=snrRange[0], max_snr_db=snrRange[1], p=0.1)
        )

        self.transformNoise = Compose(transformNoiseList)
        self.sampleRate = sampleRate

    def __call__(self, x):
        x = copy.deepcopy(x)
        x = x.T

        x = self.transform(x, sample_rate=self.sampleRate)

        # apply transform before impulse response
        if self.reverb is not None:
            xReverb = self.reverb(x, sample_rate=self.sampleRate)

            # randomize the wet/dry ratio
            alpha = random.random()

            x = alpha * x + (1 - alpha) * xReverb

        x = self.transformNoise(x, sample_rate=self.sampleRate)
        x = x.T
        return x


def mix_at_snr(
    signal: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    signal に noise を指定 SNR[dB] で加算した mixture を返す。
    戻り値は (mixture, signal_scaled, noise_scaled)。

    前提:
        - signal, noise: shape=(samples,) または (samples, channels)
        - 値域: -1.0〜1.0 の float (推奨: float32)
    動作:
        - 長さを短い方に揃える
        - signal 側はスケールしない（基準）
        - noise 側を snr_db に合うようスケール
        - 合成後 [-1, 1] にクリップ
        - 入力が 1D モノラルなら 1D で返す
    """
    # モノラル判定（両方1Dならモノラル扱い）
    mono_input = (signal.ndim == 1) and (noise.ndim == 1)

    # 2D化 (T, C)
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]
    if noise.ndim == 1:
        noise = noise[:, np.newaxis]

    # 長さ揃え
    length = min(signal.shape[0], noise.shape[0])
    signal = signal[:length]
    noise = noise[:length]

    # パワー計算（float64で安全に）
    signal_power = np.mean(signal.astype(np.float64) ** 2)
    noise_power = np.mean(noise.astype(np.float64) ** 2)

    # スケール
    if signal_power == 0.0 or noise_power == 0.0:
        return (
            signal.astype(np.float32),
            signal.astype(np.float32),
            noise.astype(np.float32),
        )

    scale = np.sqrt(signal_power / noise_power / (10.0 ** (snr_db / 10.0)))
    signal_scaled = signal.astype(np.float32)
    noise_scaled = (noise * scale).astype(np.float32)

    # 合成 & クリップ
    mixture = signal_scaled + noise_scaled
    mixture = np.clip(mixture, -1.0, 1.0).astype(np.float32)

    # モノラルなら1Dに戻す
    if mono_input:
        return mixture[:, 0], signal_scaled[:, 0], noise_scaled[:, 0]
    else:
        return mixture, signal_scaled, noise_scaled


class DatasetMaestroIterator(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: DatasetMaestro,
        hopSizeInSecond,
        chunkSizeInSecond,
        audioNormalize=True,
        notesStrictlyContained=True,
        ditheringFrames=True,
        seed=1234,
        augmentator=None,
    ):
        super().__init__()
        self.dataset = dataset
        self.hopSizeInSecond = hopSizeInSecond
        self.chunkSizeInSecond = chunkSizeInSecond
        self.audioNormalize = audioNormalize
        self.notesStrictlyContained = notesStrictlyContained
        self.ditheringFrames = ditheringFrames
        self.augmentator = augmentator

        randGen = random.Random(seed)

        chunksAll = []
        for idx, duration in enumerate(self.dataset.durations):
            duration = float(duration)
            chunkSizeInSecond = self.chunkSizeInSecond
            hopSizeInSecond = self.hopSizeInSecond

            # split the duration into equal size chunks
            # add 1 more for safe guarding the boundary
            nChunks = math.ceil((duration + chunkSizeInSecond) / self.hopSizeInSecond)

            hopPerChunk = math.ceil(chunkSizeInSecond / self.hopSizeInSecond)

            for j in range(-hopPerChunk, nChunks + hopPerChunk):
                if self.ditheringFrames:
                    shift = randGen.random() - 0.5
                else:
                    shift = 0
                begin = (j + shift) * hopSizeInSecond - chunkSizeInSecond / 2
                # end = (j+ shift)*hopSizeInSecond + chunkSizeInSecond/2
                end = begin + chunkSizeInSecond

                # if duration-begin> hopSizeInSecond:

                # add empty frames
                if begin < duration and end > 0:
                    chunksAll.append((idx, begin, end))

        randGen.shuffle(chunksAll)
        self.chunksAll = chunksAll

    def __len__(self):
        return len(self.chunksAll)

    def __getitem__(self, idx):
        if idx > self.__len__():
            raise IndexError()

        idx, begin, end = self.chunksAll[idx]

        other_idx = None
        other_end = None
        other_begin = None
        if random.random() < 0.5:
            # ランダムで他の曲のミックスを合成する
            other_idx, other_begin, other_end = self.chunksAll[
                random.randint(0, len(self.chunksAll) - 1)
            ]

        notes, target_audio, other_slice, fs = self.dataset.fetchData(
            idx,
            begin,
            end,
            audioNormalize=self.audioNormalize,
            notesStrictlyContained=self.notesStrictlyContained,
            other_idx=other_idx,
            other_begin=other_begin,
            other_end=other_end,
        )

        if self.augmentator is not None:
            target_audio = self.augmentator(target_audio)

        mixture = target_audio
        if other_slice is not None:
            random_snr_db = np.random.uniform(-6.0, 6.0)
            mixture, target_audio, other_audio = mix_at_snr(
                target_audio, other_slice, random_snr_db
            )
            target_audio = np.stack([target_audio, other_audio], axis=-2)  # [T, N, C]

        sample = {
            "notes": notes,
            "audioSlice": mixture,
            "target_audio": target_audio,
            "fs": fs,
            "begin": begin,
        }

        return sample


def collate_fn(batch):
    return batch


def collate_fn_batching(batch):
    notesBatch = [sample["notes"] for sample in batch]
    audioSlices = [torch.from_numpy(sample["audioSlice"]) for sample in batch]
    target_audios = [torch.from_numpy(sample["target_audio"]) for sample in batch]

    nAudioSamplesMin = min([_.shape[0] for _ in audioSlices])
    nAudioSamplesMax = max([_.shape[0] for _ in audioSlices])

    assert nAudioSamplesMax - nAudioSamplesMin < 2

    audioSlices = [_[:nAudioSamplesMin] for _ in audioSlices]
    target_audios = [_[:nAudioSamplesMin] for _ in target_audios]

    audioSlices = torch.stack(audioSlices, dim=0)
    target_audios = torch.stack(target_audios, dim=0)

    return {
        "notes": notesBatch,
        "audioSlices": audioSlices,
        "target_audio": target_audios,
    }


def collate_fn_randmized_len(batch):
    batchNew = []
    r = random.random() * 0.5 + 0.5

    for sample in batch:
        fs = sample["fs"]
        nSample = sample["audioSlice"].shape[0]
        sample["audioSlice"] = sample["audioSlice"][: math.ceil(nSample * r), :]

        T = math.ceil(nSample * r) / fs

        notes = [_ for _ in sample["notes"] if _.end < T]
        sample["notes"] = notes

        batchNew.append(sample)

    return batchNew


def midiToKeyNumber(midiNumber):
    # piano has a midi number range of [21, 108]
    # this function maps the range to [0, 87]
    return midiNumber - 21


def prepareIntervalsNoQuantize(notes, targetPitch):
    validateNotes(notes)

    # tracks of intervals indexed by pitch
    # for pedal event, use a negative number
    tracks = defaultdict(list)

    # split notes into tracks and then snap to the grid
    for n in notes:
        tracks[n.pitch].append(n)

    # process pitch by pitch
    intervals_all = []
    velocity_all = []
    endPointRefine_all = []

    for p in targetPitch:
        intervals = []
        endPointRefine = []
        velocity = []
        # print("pitch:", p)
        for n in tracks[p]:
            # print(n)
            assert n.start >= 0, n.start
            assert n.end >= 0, n.end

            curVelocity = n.velocity

            tmp = (n.start, n.end)

            intervals.append(tmp)
            endPointRefine.append((0, 0))
            velocity.append(curVelocity)

        # print(intervals)
        # print(endPointRefine)
        # print(velocity)

        intervals_all.append(intervals)
        endPointRefine_all.append(endPointRefine)
        velocity_all.append(velocity)

    result = {
        "intervals": intervals_all,
        "endPointRefine": endPointRefine_all,
        "velocity": velocity_all,
    }
    return result


def prepareIntervals(notes, hopSizeInSecond, targetPitch):
    validateNotes(notes)
    # print("hopSizeInSecond:", hopSizeInSecond)

    # tracks of intervals indexed by pitch
    # for pedal event, use a negative number
    tracks = defaultdict(list)

    # split notes into tracks and then snap to the grid
    for n in notes:
        tracks[n.pitch].append(n)

    # process pitch by pitch
    intervals_all = []
    velocity_all = []
    endPointRefine_all = []
    endPointPresence_all = []

    for p in targetPitch:
        intervals = []
        endPointRefine = []
        endPointPresence = []
        velocity = []
        # print("pitch:", p)
        for n in tracks[p]:
            # print(n)
            assert n.start >= 0, n.start
            assert n.end >= 0, n.end

            start_quantized = int(round(n.start / hopSizeInSecond))
            end_quantized = int(round(n.end / hopSizeInSecond))

            start_refine = n.start / hopSizeInSecond - start_quantized
            end_refine = n.end / hopSizeInSecond - end_quantized

            curVelocity = n.velocity

            tmp = (start_quantized, end_quantized)
            tmpPresence = (n.hasOnset, n.hasOffset)
            # print(n)

            # check if two consecutive notes can be seaprated by interval representation
            if len(intervals) > 0 and (
                start_quantized < intervals[-1][1]
                or (
                    end_quantized == intervals[-1][1]
                    and intervals[-1][0] == start_quantized
                )
            ):
                # raise Exception("two notes quantized in the same frame that cannot be separated: {}, {}".format(tmp, intervals[-1]))
                print(
                    "two notes quantized in the same frame that cannot be separated or they are overlapping: {}, {}. These two notes are merged".format(
                        tmp, intervals[-1]
                    )
                )

                # asd
                # print(n)
                # print(intervals[-1])
                # print(start_quantized, end_quantized)

                # two consecutive note on event, treat as the same note, use the same velocity
                intervals[-1] = (intervals[-1][0], end_quantized)
                endPointRefine[-1] = (endPointRefine[-1][0], end_refine)
                endPointPresence[-1] = (endPointPresence[-1][0], n.hasOffset)
            else:
                intervals.append(tmp)
                endPointRefine.append((start_refine, end_refine))
                endPointPresence.append(tmpPresence)
                velocity.append(curVelocity)

        # print(intervals)
        # print(endPointRefine)
        # print(velocity)

        intervals_all.append(intervals)
        endPointRefine_all.append(endPointRefine)
        endPointPresence_all.append(endPointPresence)
        velocity_all.append(velocity)

    result = {
        "intervals": intervals_all,
        "endPointRefine": endPointRefine_all,
        "endPointPresence": endPointPresence_all,
        "velocity": velocity_all,
    }
    return result
