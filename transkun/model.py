import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from .Data import *
from .utils import *
from .Evaluation import *
from .backbone import Backbone, ScaledInnerProductIntervalScorer
from collections import defaultdict
import einops
from torch_log_wmse import LogWMSE
from functools import partial
from . import CRF


class ModelConfig:
    def __init__(self):
        self.segmentHopSizeInSecond = 8
        self.segmentSizeInSecond = 16

        self.hopSize = 1024
        self.windowSize = 2048
        self.fs = 44100
        self.num_bands = 60
        self.num_channels = 2
        self.loss_spec_weight = 1.0

        self.baseSize = 64
        self.nHead = 8
        self.band_split_type = "bs"

        self.nLayers = 6
        self.hiddenFactor = 4

        self.velocityPredictorHiddenSize = 512
        self.refinedOFPredictorHiddenSize = 512

        self.scoringExpansionFactor = 4
        self.useInnerProductScorer = True

        self.scoreDropoutProb = 0.1
        self.contextDropoutProb = 0.1
        self.velocityDropoutProb = 0.1
        self.refinedOFDropoutProb = 0.1

    def __repr__(self):
        return repr(self.__dict__)


Config = ModelConfig


def _stft_l1(
    recon: torch.Tensor,
    target: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
) -> torch.Tensor:
    """
    1 解像度分の L1 損失だけ返す小さな関数。
    checkpoint で呼び出すため、すべての引数を Tensor にする。
    """
    # (B*C, T) -> 複素 STFT
    spec_hat = torch.stft(
        recon, n_fft, hop_length, win_length, window=window, return_complex=True
    )
    with torch.no_grad():  # 勾配不要なので保存しない
        spec_ref = torch.stft(
            target, n_fft, hop_length, win_length, window=window, return_complex=True
        )
    return F.l1_loss(spec_hat, spec_ref)


def multi_resolution_stft_loss(
    recon_audio: torch.Tensor,
    target_audio: torch.Tensor,
    n_fft: int,
    resolutions: list[int] = [4096, 2048, 1024, 512, 256],
    hop_length: int = 147,
    window_fn=torch.hann_window,
    loss_weight: float = 1.0,
) -> torch.Tensor:
    b, c, n, t = recon_audio.shape
    recon = recon_audio.reshape(-1, t)
    target = target_audio.reshape(-1, t)

    total_loss = recon.new_tensor(0.0)
    for win_len in resolutions:
        n_fft_sel = max(n_fft, win_len)
        window = window_fn(win_len, device=recon.device, dtype=recon.dtype)

        loss_i = torch.utils.checkpoint.checkpoint(
            _stft_l1,
            recon,
            target,
            torch.tensor(n_fft_sel),
            torch.tensor(hop_length),
            torch.tensor(win_len),
            window,
            use_reentrant=False,
        )
        total_loss = total_loss + loss_i
    return total_loss * loss_weight


class TransKun(torch.nn.Module):
    Config = ModelConfig

    def __init__(self, conf):
        super().__init__()

        self.conf = conf
        self.hop_size = conf.hopSize

        self.window_size = conf.windowSize
        self.fs = conf.fs
        self.num_channels = conf.num_channels
        self.n_fft = self.window_size
        self.num_stems = 2

        self.segmentSizeInSecond = conf.segmentSizeInSecond
        self.segmentHopSizeInSecond = conf.segmentHopSizeInSecond

        self.target_midi_pitch = [-64, -67] + list(range(21, 108 + 1))

        self.scorer = ScaledInnerProductIntervalScorer(
            conf.baseSize * conf.scoringExpansionFactor,
            1,
            dropoutProb=conf.scoreDropoutProb,
        )

        self.velocityPredictor = nn.Sequential(
            nn.Linear(
                conf.baseSize * 3 * conf.scoringExpansionFactor,
                conf.velocityPredictorHiddenSize,
            ),
            nn.GELU(),
            nn.Dropout(conf.velocityDropoutProb),
            nn.Linear(conf.velocityPredictorHiddenSize, 128),
        )

        # output dequantize offset + presence logits
        self.refinedOFPredictor = nn.Sequential(
            nn.Linear(
                conf.baseSize * 3 * conf.scoringExpansionFactor,
                conf.refinedOFPredictorHiddenSize,
            ),
            nn.GELU(),
            nn.Dropout(conf.refinedOFDropoutProb),
            nn.Linear(conf.refinedOFPredictorHiddenSize, 4),
        )

        self.backbone = Backbone(
            sampling_rate=conf.fs,
            num_channels=conf.num_channels,
            n_fft=self.n_fft,
            hop_size=conf.hopSize,
            n_bands=conf.num_bands,
            hidden_size=conf.baseSize,
            num_heads=conf.nHead,
            ffn_hidden_size_factor=conf.hiddenFactor,
            num_layers=conf.nLayers,
            scoring_expansion_factor=conf.scoringExpansionFactor,
            dropout=conf.contextDropoutProb,
            target_midi_pitches=self.target_midi_pitch,
            band_split_type=conf.band_split_type,
        )

    def getDevice(self):
        return next(self.parameters()).device

    def process_frames_batch(self, inputs):
        # inputs: (B, C, N)
        ctx, recon_audio = self.backbone(inputs)

        # [ nStep, nStep, nBatch, nSym ]
        S_batch, S_skip_batch = self.scorer(ctx)

        # batch the CRF together
        S_batch = S_batch.flatten(-2, -1)
        S_skip_batch = S_skip_batch.flatten(-2, -1)

        # with shape [*, nBatch*nSym]

        crf = CRF.NeuralSemiCRFInterval(S_batch, S_skip_batch)

        return crf, ctx, recon_audio

    def log_prob(self, inputs, notesBatch, target_audio=None):
        B = inputs.shape[0]
        # [B, C, T]
        inputs = inputs.transpose(-1, -2)

        device = inputs.device
        # target_audio: [B, T, N, C]
        if target_audio is not None:
            target_audio = target_audio.permute(0, 3, 2, 1)  # [B, C, N, T]

        crfBatch, ctxBatch, recon_audio = self.process_frames_batch(inputs)

        loss_spec = torch.tensor(0.0, device=device)
        loss_wmse = 0.0
        if target_audio is not None:
            target_audio = target_audio[..., : recon_audio.shape[-1]]

            loss_spec = multi_resolution_stft_loss(
                recon_audio=recon_audio,
                target_audio=target_audio,
                n_fft=self.n_fft,
                hop_length=512,
            )
            log_wmse = LogWMSE(
                audio_length=recon_audio.shape[-1] / self.fs,
                sample_rate=self.fs,
                return_as_loss=True,
            )
            loss_wmse = log_wmse(
                inputs,
                recon_audio,
                target_audio,
            )

        # prepare groundtruth
        intervalsBatch = []
        velocityBatch = []
        ofRefinedGTBatch = []
        ofPresenceGTBatch = []
        for notes in notesBatch:
            data = prepareIntervals(
                notes, self.hop_size / self.fs, self.target_midi_pitch
            )
            intervalsBatch.append(data["intervals"])
            velocityBatch.append(sum(data["velocity"], []))
            ofRefinedGTBatch.append(sum(data["endPointRefine"], []))
            ofPresenceGTBatch.append(sum(data["endPointPresence"], []))

        intervalsBatch_flatten = sum(intervalsBatch, [])
        assert len(intervalsBatch_flatten) == B * len(self.target_midi_pitch)

        # tmp = torch.Tensor(sum(intervalsBatch_flatten, []))
        # print(tmp.max())

        pathScore = crfBatch.evalPath(intervalsBatch_flatten)
        logZ = crfBatch.computeLogZ()
        logProb = pathScore - logZ
        logProb = logProb.view(B, -1)

        # then fetch the attrbute features for all intervals

        nIntervalsAll = sum([len(_) for _ in intervalsBatch_flatten])

        if nIntervalsAll > 0:
            (
                ctx_a_all,
                ctx_b_all,
                symIdx_all,
                scatterIdx_all,
            ) = self.fetchIntervalFeaturesBatch(ctxBatch, intervalsBatch)

            attributeInput = torch.cat(
                [ctx_a_all, ctx_b_all, ctx_a_all * ctx_b_all], dim=-1
            )

            # prepare groundtruth for velocity

            velocityBatch = sum(velocityBatch, [])
            ofRefinedGTBatch = sum(ofRefinedGTBatch, [])
            ofPresenceGTBatch = sum(ofPresenceGTBatch, [])

            logitsVelocity = self.velocityPredictor(attributeInput)
            logitsVelocity = F.log_softmax(logitsVelocity, dim=-1)

            velocityBatch = torch.tensor(velocityBatch, dtype=torch.long, device=device)

            logProbVelocity = torch.gather(
                logitsVelocity, dim=-1, index=velocityBatch.unsqueeze(-1)
            ).squeeze(-1)

            ofRefinedGTBatch = torch.tensor(
                ofRefinedGTBatch, device=device, dtype=torch.float
            )
            ofPresenceGTBatch = torch.tensor(
                ofPresenceGTBatch, device=device, dtype=torch.float
            )

            # shift it to [0,1]
            # print("GT:", ofRefinedGTBatch)

            ofRefinedGTBatch = ofRefinedGTBatch * 0.99 + 0.5

            ofValue, ofPresence = self.refinedOFPredictor(attributeInput).chunk(
                2, dim=-1
            )

            # ofValue = F.logsigmoid(ofValue)

            ofDist = torch.distributions.ContinuousBernoulli(logits=ofValue)

            logProbOF = ofDist.log_prob(ofRefinedGTBatch).sum(-1)

            ofPresenceDist = torch.distributions.Bernoulli(logits=ofPresence)

            logProbOFPresence = ofPresenceDist.log_prob(ofPresenceGTBatch).sum(-1)

            # scatter them back
            logProb = logProb.view(-1)

            logProb = logProb.scatter_add(
                -1, scatterIdx_all, logProbVelocity + logProbOF + logProbOFPresence
            )

        logProb = logProb.view(B, -1)

        return logProb, (loss_spec, loss_wmse)

    def compute_stats_mireval(self, inputs, notes_batch):
        B = inputs.shape[0]
        inputs = inputs.transpose(-1, -2)

        notesEstBatch, _, _ = self.transcribeFrames(inputs)

        assert len(notes_batch) == len(notesEstBatch)

        # metricsBatch = [Evaluation.compareTranscription(est, gt) for est, gt in zip(notesEstBatch, notesBatch)  ]

        # aggregate by  batch count
        metricsBatch = []

        nEstTotal = 0
        nGTTotal = 0
        nCorrectTotal = 0

        for est, gt in zip(notesEstBatch, notes_batch):
            metrics = compareTranscription(est, gt)
            p, r, f, _ = metrics["note+offset"]

            nGT = metrics["nGT"]
            nEst = metrics["nEst"]

            nCorrect = r * nGT

            nEstTotal += nEst
            nGTTotal += nGT
            nCorrectTotal += nCorrect

        stats = {
            "nGT": nGTTotal,
            "nEst": nEstTotal,
            "nCorrect": nCorrectTotal,
        }

        return stats

    def compute_stats(self, inputs, notes_batch):
        B = inputs.shape[0]
        inputs = inputs.transpose(-1, -2)

        device = inputs.device

        crfBatch, ctxBatch, _ = self.process_frames_batch(inputs)

        path = crfBatch.decode()

        # print(sum([len(p) for p in path]))
        intervalsBatch = []
        velocityBatch = []
        ofRefinedGTBatch = []
        for notes in notes_batch:
            data = prepareIntervals(
                notes, self.hop_size / self.fs, self.target_midi_pitch
            )
            intervalsBatch.append(data["intervals"])
            velocityBatch.append(sum(data["velocity"], []))
            ofRefinedGTBatch.append(sum(data["endPointRefine"], []))

        intervalsBatch_flatten = sum(intervalsBatch, [])
        assert len(intervalsBatch_flatten) == B * len(self.target_midi_pitch)

        # then compare intervals and path
        assert len(path) == len(intervalsBatch_flatten)

        # print(sum([len(p) for p in intervalsBatch_flatten]), "intervalsGT")

        statsAll = [
            compareBracket(l1, l2) for l1, l2 in zip(path, intervalsBatch_flatten)
        ]

        nGT = sum([_[0] for _ in statsAll])
        nEst = sum([_[1] for _ in statsAll])
        nCorrect = sum([_[2] for _ in statsAll])

        # omit pedal
        statsFramewiseAll = [
            compareFramewise(l1, l2) for l1, l2 in zip(path, intervalsBatch_flatten)
        ]

        nGTFramewise = sum([_[0] for _ in statsFramewiseAll])
        nEstFramewise = sum([_[1] for _ in statsFramewiseAll])
        nCorrectFramewise = sum([_[2] for _ in statsFramewiseAll])

        # then make forced predictions about velocity and refined onset offset

        (
            ctx_a_all,
            ctx_b_all,
            symIdx_all,
            scatterIdx_all,
        ) = self.fetchIntervalFeaturesBatch(ctxBatch, intervalsBatch)

        attributeInput = torch.cat(
            [
                ctx_a_all,
                ctx_b_all,
                ctx_a_all * ctx_b_all,
            ],
            dim=-1,
        )

        logitsVelocity = self.velocityPredictor(attributeInput)
        pVelocity = F.softmax(logitsVelocity, dim=-1)

        # MSE
        w = torch.arange(128, device=device)
        velocity = (pVelocity * w).sum(-1)

        ofValue, _ = self.refinedOFPredictor(attributeInput).chunk(2, dim=-1)
        # ofValue = torch.sigmoid(ofValue)-0.5
        ofDist = torch.distributions.ContinuousBernoulli(logits=ofValue)

        ofValue = (ofDist.mean - 0.5) / 0.99
        ofValue = torch.clamp(ofValue, -0.5, 0.5)

        velocityBatch = sum(velocityBatch, [])
        velocityBatch = torch.tensor(velocityBatch, dtype=torch.long, device=device)

        ofRefinedGTBatch = sum(ofRefinedGTBatch, [])
        ofRefinedGTBatch = torch.tensor(
            ofRefinedGTBatch, device=device, dtype=torch.float
        )
        # compare p velocity with ofValue

        # ofValue-of
        seOF = (ofValue - ofRefinedGTBatch).pow(2).sum()
        seVelocity = (velocity - velocityBatch).pow(2).sum()

        # print(ofValue[0], ofRefinedGTBatch[0])
        # print(ofValue[-1], ofRefinedGTBatch[-1])

        stats = {
            "nGT": nGT,
            "nEst": nEst,
            "nCorrect": nCorrect,
            "nGTFramewise": nGTFramewise,
            "nEstFramewise": nEstFramewise,
            "nCorrectFramewise": nCorrectFramewise,
            "seVelocityForced": seVelocity.item(),
            "seOFForced": seOF.item(),
        }

        return stats

    def fetchIntervalFeaturesBatch(self, ctxBatch, intervalsBatch):
        # ctx: [N, SYM, T, D]
        ctx_a_all = []
        ctx_b_all = []
        symIdx_all = []
        scatterIdx_all = []
        device = ctxBatch.device
        T = ctxBatch.shape[-2]

        for idx, curIntervals in enumerate(intervalsBatch):
            nIntervals = len(sum(curIntervals, []))
            if nIntervals > 0:
                symIdx = torch.tensor(
                    listToIdx(curIntervals), dtype=torch.long, device=device
                )
                symIdx_all.append(symIdx)

                scatterIdx_all.append(idx * len(self.target_midi_pitch) + symIdx)

                indices = torch.tensor(
                    sum(curIntervals, []), dtype=torch.long, device=device
                )
                # print(len(symIdx), len(indices[:,0]))

                ctx_a = (
                    ctxBatch[idx]
                    .flatten(0, 1)
                    .index_select(dim=0, index=indices[:, 0] + symIdx * T)
                )
                ctx_b = (
                    ctxBatch[idx]
                    .flatten(0, 1)
                    .index_select(dim=0, index=indices[:, 1] + symIdx * T)
                )

                ctx_a_all.append(ctx_a)
                ctx_b_all.append(ctx_b)

        ctx_a_all = torch.cat(ctx_a_all, dim=0)
        ctx_b_all = torch.cat(ctx_b_all, dim=0)
        symIdx_all = torch.cat(symIdx_all, dim=0)
        scatterIdx_all = torch.cat(scatterIdx_all, dim=0)

        return ctx_a_all, ctx_b_all, symIdx_all, scatterIdx_all

    def transcribeFrames(
        self,
        inputs,
        forcedStartPos=None,
        velocityCriteron="hamming",
        onsetBound=None,
        lastFrameIdx=None,
    ):
        device = inputs.device
        nBatch = inputs.shape[0]
        crfBatch, ctxBatch, recon_audio = self.process_frames_batch(inputs)
        nSymbols = len(self.target_midi_pitch)
        n_frames = inputs.shape[1] // self.hop_size + 1

        if lastFrameIdx is None:
            lastFrameIdx = n_frames - 1

        path = crfBatch.decode(forcedStartPos=forcedStartPos, forward=False)

        assert nSymbols * nBatch == len(path)

        # also get the last position for each path for forced decoding
        if onsetBound is not None:
            path = [[e for e in _ if e[0] < onsetBound] for _ in path]

        # then predict attributes associated with frames

        # obtain segment features

        nIntervalsAll = sum([len(_) for _ in path])
        # print("#e:", nIntervalsAll)

        intervalsBatch = []
        for idx in range(nBatch):
            curIntervals = path[idx * nSymbols : (idx + 1) * nSymbols]
            intervalsBatch.append(curIntervals)

        if nIntervalsAll == 0:
            # nothing detected, return empty
            return (
                [[] for _ in range(nBatch)],
                [0 for _ in range(len(path))],
                recon_audio,
            )

        # then predict the attribute set

        (
            ctx_a_all,
            ctx_b_all,
            symIdx_all,
            scatterIdx_all,
        ) = self.fetchIntervalFeaturesBatch(ctxBatch, intervalsBatch)

        attributeInput = torch.cat(
            [
                ctx_a_all,
                ctx_b_all,
                ctx_a_all * ctx_b_all,
            ],
            dim=-1,
        )

        logitsVelocity = self.velocityPredictor(attributeInput)
        pVelocity = F.softmax(logitsVelocity, dim=-1)

        # MSE
        if velocityCriteron == "mse":
            w = torch.arange(128, device=device)
            velocity = (pVelocity * w).sum(-1)
        elif velocityCriteron == "match":
            # TODO: Minimal risk
            # predict velocity, readout by minimizing the risk
            # 0.1 is usually the tolerance for the velocity, so....

            # It will never make so extreme predictions

            # create the risk matrix
            w = torch.arange(128, device=device)

            # [Predicted, Actual]

            tolerance = 0.1 * 128
            utility = ((w.unsqueeze(1) - w.unsqueeze(0)).abs() < tolerance).float()

            r = pVelocity @ utility

            velocity = torch.argmax(r, dim=-1)

        elif velocityCriteron == "hamming":
            # return the mode
            velocity = torch.argmax(pVelocity, dim=-1)

        elif velocityCriteron == "mae":
            # then this time return the median
            pCum = pVelocity.cumsum(-1)
            tmp = (pCum - 0.5) > 0
            w2 = torch.arange(128, 0.0, -1, device=device)

            velocity = torch.argmax(tmp * w2, dim=-1)

        else:
            raise Exception("Unrecognized criterion: {}".format(velocityCriteron))

        ofValue, ofPresence = self.refinedOFPredictor(attributeInput).chunk(2, dim=-1)
        # ofValue = torch.sigmoid(ofValue)-0.5
        ofDist = torch.distributions.ContinuousBernoulli(logits=ofValue)

        ofValue = (ofDist.mean - 0.5) / 0.99
        ofValue = torch.clamp(ofValue, -0.5, 0.5)

        ofPresence = ofPresence > 0

        # print(velocity)
        # print(ofValue)

        # generate the final result

        # parse the list of path to (begin, end, midipitch, velocity)

        velocity = velocity.cpu().detach().tolist()
        ofValue = ofValue.cpu().detach().tolist()
        ofPresence = ofPresence.cpu().detach().tolist()

        assert len(velocity) == len(ofValue)
        assert len(velocity) == nIntervalsAll

        nCount = 0

        notes = [[] for _ in range(nBatch)]

        frameDur = self.hop_size / self.fs

        # the last offset
        lastP = []

        for idx in range(nBatch):
            curIntervals = intervalsBatch[idx]

            for j, eventType in enumerate(self.target_midi_pitch):
                lastEnd = 0
                curLastP = 0

                for k, aInterval in enumerate(curIntervals[j]):
                    # print(aInterval, eventType, velocity[nCount], ofValue[nCount])
                    isLast = k == (len(curIntervals[j]) - 1)

                    curVelocity = velocity[nCount]

                    curOffset = ofValue[nCount]
                    start = (aInterval[0] + curOffset[0]) * frameDur
                    end = (aInterval[1] + curOffset[1]) * frameDur

                    # ofPresence prediction is only used to distinguish the corner case that either onset or offset happens exactly on the first/last frame.

                    hasOnset = (aInterval[0] > 0) or ofPresence[nCount][0]
                    hasOffset = (aInterval[1] < lastFrameIdx) or ofPresence[nCount][1]

                    assert aInterval[0] >= 0
                    # print(aInterval[0], aInterval[1], nFrame)
                    start = max(start, lastEnd)
                    end = max(end, start + 1e-8)
                    lastEnd = end
                    curNote = Note(
                        start=start,
                        end=end,
                        pitch=eventType,
                        velocity=curVelocity,
                        hasOnset=hasOnset,
                        hasOffset=hasOffset,
                    )

                    notes[idx].append(curNote)

                    if hasOffset:
                        curLastP = aInterval[1]
                    # if hasOnset and hasOffset:
                    # curLastP = aInterval[1]

                    nCount += 1

                lastP.append(curLastP)

            notes[idx].sort(key=lambda x: (x.start, x.end, x.pitch))

        return notes, lastP, recon_audio

    def transcribe(
        self,
        x,
        stepInSecond=None,
        segmentSizeInSecond=None,
        discardSecondHalf=False,
        mergeIncompleteEvent=True,
    ):
        if stepInSecond is None and segmentSizeInSecond is None:
            stepInSecond = self.segmentHopSizeInSecond
            segmentSizeInSecond = self.segmentSizeInSecond

        x = x.transpose(-1, -2)

        padTimeBegin = segmentSizeInSecond - stepInSecond

        x = F.pad(
            x, (math.ceil(padTimeBegin * self.fs), math.ceil(self.fs * (padTimeBegin)))
        )

        nSample = x.shape[-1]

        eventsAll = []

        eventsByType = defaultdict(list)
        startFrameIdx = math.floor(padTimeBegin * self.fs / self.hop_size)
        startPos = [startFrameIdx] * len(self.target_midi_pitch)

        stepSize = math.ceil(stepInSecond * self.fs / self.hop_size) * self.hop_size
        segmentSize = math.ceil(segmentSizeInSecond * self.fs)

        total_length = x.shape[-1]  # 元波形長 (padding 後)
        recon_buffer = torch.zeros(
            self.num_channels,
            self.num_stems,
            total_length,
            device=x.device,
            dtype=x.dtype,
        )
        window_buffer = torch.zeros_like(recon_buffer)  # 窓の重なりを計算するため
        ola_window = torch.hann_window(segmentSize, device=x.device, dtype=x.dtype)
        ola_window = ola_window[None, None, :]  # (1, 1, segmentSize)

        for i in range(0, nSample, stepSize):
            # t1 = time.time()

            j = min(i + segmentSize, nSample)
            # print(i, j)

            beginTime = (i) / self.fs - padTimeBegin
            # print(beginTime)

            curSlice = x[:, i:j]
            if curSlice.shape[-1] < segmentSize:
                # pad to the segmentSize
                curSlice = F.pad(curSlice, (0, segmentSize - curSlice.shape[-1]))

            lastFrameIdx = round(segmentSize / self.hop_size)
            # # print(curSlice.shape)
            # # print(startPos)
            # startPos = None
            if discardSecondHalf:
                onsetBound = stepSize
            else:
                onsetBound = None

            curEvents, lastP, recon_audio = self.transcribeFrames(
                curSlice.unsqueeze(0),
                forcedStartPos=startPos,
                velocityCriteron="hamming",
                onsetBound=onsetBound,
                lastFrameIdx=lastFrameIdx,
            )
            recon_audio = recon_audio[0]  # (C, N, segmentSize)
            curEvents = curEvents[0]

            # セグメント長と実際の recon_audio 長が違う場合に備えてクロップ
            seg_len = recon_audio.shape[-1]  # (= segmentSize)
            end_idx = min(i + seg_len, recon_buffer.shape[-1])
            valid = end_idx - i  # この区間だけ書き込める
            win = ola_window[..., :seg_len]
            recon_buffer[..., i:end_idx] += recon_audio[..., :valid] * win[..., :valid]
            window_buffer[..., i:end_idx] += win[..., :valid]

            startPos = []
            for k in lastP:
                startPos.append(max(k - int(stepSize / self.hop_size), 0))

            # # shift all notes by beginTime
            for e in curEvents:
                e.start += beginTime
                e.end += beginTime

                e.start = max(e.start, 0)
                e.end = max(e.end, e.start)
                # print(e.start, e.end, e.pitch, e.hasOnset, e.hasOffset)

            for e in curEvents:
                if mergeIncompleteEvent:
                    if len(eventsByType[e.pitch]) > 0:
                        last_e = eventsByType[e.pitch][-1]

                        # test if e overlap with the last event
                        if e.start < last_e.end:
                            if e.hasOnset:  # and e.hasOffset:
                                eventsByType[e.pitch][-1] = e
                            else:
                                # merge two events
                                eventsByType[e.pitch][-1].hasOffset = e.hasOffset
                                eventsByType[e.pitch][-1].end = max(e.end, last_e.end)
                                # eventsByType[e.pitch][-1].end = max(e.end, last_e.end)

                            continue

                if e.hasOnset:
                    eventsByType[e.pitch].append(e)

            eventsAll.extend(curEvents)

        # handling incomplete events in the last segment
        for eventType in eventsByType:
            if len(eventsByType[eventType]) > 0:
                eventsByType[eventType][-1].hasOffset = True

        # flatten all events
        eventsAll = sum(eventsByType.values(), [])

        # post filtering
        eventsAll = [n for n in eventsAll if n.hasOffset]

        eventsAll = resolveOverlapping(eventsAll)

        # 窓で割って規格化
        mask = window_buffer > 0
        recon_buffer[mask] = recon_buffer[mask] / window_buffer[mask]

        # パディング除去
        pad_len = math.ceil(padTimeBegin * self.fs)
        recon_buffer = recon_buffer[..., pad_len:-pad_len]

        return eventsAll, recon_buffer
