import os
import random

from torch.utils.tensorboard import SummaryWriter
import torch
from .model import TransKun
from . import Data
import copy
import time
import numpy as np
import math

from .train_utils import *
import argparse

import moduleconf


def collect_grad_stats(model, learning_rate: float | None = None, eps: float = 1e-12):
    stats = []
    for parameter_name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue

        grad_data = parameter.grad.detach()
        param_data = parameter.detach()

        grad_rms = grad_data.pow(2).mean().sqrt().item()
        param_rms = param_data.pow(2).mean().sqrt().item()
        grad_l2 = grad_data.norm().item()
        param_l2 = param_data.norm().item()

        if learning_rate is not None:
            relative_update = learning_rate * grad_rms / (param_rms + eps)
        else:
            relative_update = float("nan")

        stats.append(
            {
                "name": parameter_name,
                "grad_rms": grad_rms,
                "grad_l2": grad_l2,
                "param_rms": param_rms,
                "param_l2": param_l2,
            }
        )
    return stats


def train(workerId, filename, runSeed, args):
    device = torch.device(
        "cuda:" + str(workerId % torch.cuda.device_count())
        if torch.cuda.is_available()
        else "cpu"
    )
    torch.cuda.set_device(device)
    random.seed(workerId + int(time.time()))
    # torch.autograd.set_detect_anomaly(True)
    np.random.seed(workerId + int(time.time()))
    torch.manual_seed(workerId + int(time.time()))
    torch.cuda.manual_seed(workerId + int(time.time()))

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # obtain the Model Module
    confManager = moduleconf.parseFromFile(args.modelConf)
    transkun_model = confManager["Model"].module.TransKun
    conf = confManager["Model"].config
    model: TransKun

    if workerId == 0:
        # if the saved file does not exist
        if not os.path.exists(filename):
            print("initializing the model...")

            (
                startEpoch,
                startIter,
                model,
                lossTracker,
                best_state_dict,
                optimizer,
                lrScheduler,
            ) = initializeCheckpoint(
                transkun_model,
                device=device,
                max_lr=args.max_lr,
                weight_decay=args.weight_decay,
                nIter=args.nIter,
                conf=conf,
            )

            save_checkpoint(
                filename,
                startEpoch,
                startIter,
                model,
                lossTracker,
                best_state_dict,
                optimizer,
                lrScheduler,
            )

    (
        startEpoch,
        startIter,
        model,
        lossTracker,
        best_state_dict,
        optimizer,
        lrScheduler,
    ) = load_checkpoint(transkun_model, conf, filename, device)
    print("#{} loaded".format(workerId))

    if workerId == 0:
        print("loading dataset....")

    datasetPath = args.datasetPath
    datasetPicklePath = args.datasetMetaFile_train
    datasetPicklePath_val = args.datasetMetaFile_val

    dataset = Data.DatasetMaestro(datasetPath, datasetPicklePath)
    datasetVal = Data.DatasetMaestro(datasetPath, datasetPicklePath_val)

    print("#{} loaded".format(workerId))

    if workerId == 0:
        writer = SummaryWriter(filename + ".log")

    globalStep = startIter
    # create dataloader

    # this iterator should be constructed each time
    batchSize = args.batchSize
    loss_spec_weight = conf.loss_spec_weight

    if args.hopSize is None:
        hopSize = conf.segmentHopSizeInSecond
    else:
        hopSize = args.hopSize

    if args.chunkSize is None:
        chunkSize = conf.segmentSizeInSecond
    else:
        chunkSize = args.chunkSize

    gradNormHist = MovingBuffer(initValue=40, maxLen=10000)

    augmentator = None
    if args.augment:
        augmentator = Data.AugmentatorAudiomentations(
            sampleRate=44100, noiseFolder=args.noiseFolder, convIRFolder=args.irFolder
        )

    for epoc in range(startEpoch, 1000000):
        # average length will be chunkSize
        dataIter = Data.DatasetMaestroIterator(
            dataset,
            hopSize,
            chunkSize,
            seed=epoc * 100 + runSeed,
            augmentator=augmentator,
            notesStrictlyContained=False,
        )

        dataloader = torch.utils.data.DataLoader(
            dataIter,
            batch_size=batchSize,
            collate_fn=Data.collate_fn_batching,
            num_workers=args.dataLoaderWorkers,
            shuffle=True,
            drop_last=True,
            prefetch_factor=max(4, args.dataLoaderWorkers),
        )

        lossAll = []
        globalStepWarmupCutoff = globalStep + 500

        for idx, batch in enumerate(dataloader):
            if workerId == 0:
                currentLR = [p["lr"] for p in optimizer.param_groups][0]
                writer.add_scalar(f"Optimizer/lr", currentLR, globalStep)

            computeStats = False
            if idx % 40 == 0:
                computeStats = True

            t1 = time.time()

            model.train()
            optimizer.zero_grad()

            totalBatch = torch.zeros(1).cuda()
            totalLoss = torch.zeros(1).cuda()
            totalLen = torch.zeros(1).cuda()

            totalGT = torch.zeros(1).cuda()
            totalEst = torch.zeros(1).cuda()
            totalCorrect = torch.zeros(1).cuda()

            totalGTFramewise = torch.zeros(1).cuda()
            totalEstFramewise = torch.zeros(1).cuda()
            totalCorrectFramewise = torch.zeros(1).cuda()

            totalSEVelocity = torch.zeros(1).cuda()
            totalSEOF = torch.zeros(1).cuda()

            notesBatch = batch["notes"]
            audioSlices = batch["audioSlices"].to(device)
            target_audio = batch["target_audio"].to(device)
            audioLength = audioSlices.shape[1] / model.conf.fs

            logp, (loss_wave, loss_wmse) = model.log_prob(
                audioSlices, notesBatch, target_audio=target_audio
            )
            loss_recon = (loss_wave * 1000.0 + loss_wmse) * loss_spec_weight
            loss_seq = -logp.sum(-1).mean()
            loss = loss_seq + loss_recon

            (loss / 50).backward()

            # 勾配の確認
            # print(collect_grad_stats(model))

            totalBatch = totalBatch + 1
            totalLen = totalLen + audioLength
            totalLoss = totalLoss + loss.detach() - loss_recon.detach()

            if computeStats:
                with torch.no_grad():
                    model.eval()
                    stats = model.compute_stats(audioSlices, notesBatch)
                    stats2 = model.compute_stats_mireval(audioSlices, notesBatch)

            totalGT = totalGT + stats2["nGT"]
            totalEst = totalEst + stats2["nEst"]
            totalCorrect = totalCorrect + stats2["nCorrect"]
            totalGTFramewise = totalGTFramewise + stats["nGTFramewise"]
            totalEstFramewise = totalEstFramewise + stats["nEstFramewise"]
            totalCorrectFramewise = totalCorrectFramewise + stats["nCorrectFramewise"]
            totalSEVelocity = totalSEVelocity + stats["seVelocityForced"]
            totalSEOF = totalSEOF + stats["seOFForced"]

            loss = totalLoss / totalLen

            # adaptive gradient clipping
            curClipValue = gradNormHist.getQuantile(args.gradClippingQuantile)

            totalNorm = torch.nn.utils.clip_grad_norm_(model.parameters(), curClipValue)

            gradNormHist.step(totalNorm.item())

            optimizer.step()

            try:
                if globalStep > globalStepWarmupCutoff:
                    lrScheduler.step()
            except:
                # continue after the final iteration
                pass

            if workerId == 0:
                t2 = time.time()
                print(
                    "epoch:{} progress:{:0.3f} step:{}  loss:{:0.4f} log_wmse:{:0.4f} gradNorm:{:0.2f} clipValue:{:0.2f} time:{:0.2f} ".format(
                        epoc,
                        idx / len(dataloader),
                        globalStep,
                        loss.item(),
                        loss_wmse.item(),
                        totalNorm.item(),
                        curClipValue,
                        t2 - t1,
                    )
                )
                writer.add_scalar(f"Loss/train", loss.item(), globalStep)
                writer.add_scalar(f"Optimizer/gradNorm", totalNorm.item(), globalStep)
                writer.add_scalar(f"Optimizer/clipValue", curClipValue, globalStep)
                writer.add_scalar(f"Loss/train_log_wmse", loss_wmse.item(), globalStep)
                if computeStats:
                    num_ground_truth = totalGT.item() + 1e-4
                    num_estimated = totalEst.item() + 1e-4
                    num_correct = totalCorrect.item() + 1e-4
                    if num_estimated == 0:
                        precision = 0.0
                    else:
                        precision = num_correct / num_estimated
                    if num_ground_truth == 0:
                        recall = 0.0
                    else:
                        recall = num_correct / num_ground_truth
                    f1 = 2 * precision * recall / (precision + recall)
                    print(
                        "nGT:{} nEst:{} nCorrect:{}".format(
                            num_ground_truth, num_estimated, num_correct
                        )
                    )

                    writer.add_scalar(f"Loss/train_f1", f1, globalStep)
                    writer.add_scalar(f"Loss/train_precision", precision, globalStep)
                    writer.add_scalar(f"Loss/train_recall", recall, globalStep)

                    nGTFramewise = totalGTFramewise.item() + 1e-4
                    nEstFramewise = totalEstFramewise.item() + 1e-4
                    nCorrectFramewise = totalCorrectFramewise.item() + 1e-4
                    if nEstFramewise == 0.0:
                        precisionFrame = 0.0
                    else:
                        precisionFrame = nCorrectFramewise / nEstFramewise
                    if nGTFramewise == 0.0:
                        recallFrame = 0.0
                    else:
                        recallFrame = nCorrectFramewise / nGTFramewise
                    f1Frame = (
                        2
                        * precisionFrame
                        * recallFrame
                        / (precisionFrame + recallFrame)
                    )

                    mseVelocity = totalSEVelocity.item() / num_ground_truth
                    mseOF = totalSEOF.item() / num_ground_truth

                    writer.add_scalar(f"Loss/train_f1_frame", f1Frame, globalStep)
                    writer.add_scalar(
                        f"Loss/train_precision_frame", precisionFrame, globalStep
                    )
                    writer.add_scalar(
                        f"Loss/train_recall_frame", recallFrame, globalStep
                    )
                    writer.add_scalar(
                        f"Loss/train_mse_velocity", mseVelocity, globalStep
                    )
                    writer.add_scalar(f"Loss/train_mse_OF", mseOF, globalStep)
                    print("f1:{} precision:{} recall:{}".format(f1, precision, recall))
                    print(
                        "f1Frame:{} precisionFrame:{} recallFrame:{}".format(
                            f1Frame, precisionFrame, recallFrame
                        )
                    )
                    print("mseVelocity:{} mseOF:{}".format(mseVelocity, mseOF))

                if math.isnan(loss.item()):
                    exit()
                lossAll.append(loss.item())

                if idx % 2000 == 1999:
                    save_checkpoint(
                        filename,
                        epoc + 1,
                        globalStep + 1,
                        model,
                        lossTracker,
                        best_state_dict,
                        optimizer,
                        lrScheduler,
                    )
                    print("saved")

            globalStep += 1
            # torch.cuda.empty_cache()

        if workerId == 0:
            print("Validating...")
        # let's do validation
        torch.cuda.empty_cache()

        dataIterVal = Data.DatasetMaestroIterator(
            datasetVal,
            hopSizeInSecond=conf.segmentHopSizeInSecond,
            chunkSizeInSecond=chunkSize,
            notesStrictlyContained=False,
            seed=runSeed + epoc * 100,
        )
        dataloaderVal = torch.utils.data.DataLoader(
            dataIterVal,
            batch_size=2 * batchSize,
            collate_fn=Data.collate_fn,
            num_workers=args.dataLoaderWorkers,
            shuffle=True,
        )

        model.eval()
        valResult = doValidation(model, dataloaderVal, device=device)

        nll = valResult["meanNLL"]
        f1 = valResult["f1"]

        # lrScheduler.step(nll)
        torch.cuda.empty_cache()

        if workerId == 0:
            lossAveraged = sum(lossAll) / len(lossAll)
            lossAll = []
            lossTracker["train"].append(lossAveraged)
            lossTracker["val"].append(f1)

            print("result:", valResult)

            for key in valResult:
                writer.add_scalar("val/" + key, valResult[key], epoc)

            if f1 >= max(lossTracker["val"]) * 1.00:
                print("best updated")
                best_state_dict = copy.deepcopy(model.state_dict())

            save_checkpoint(
                filename,
                epoc + 1,
                globalStep + 1,
                model,
                lossTracker,
                best_state_dict,
                optimizer,
                lrScheduler,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Perform Training")
    parser.add_argument("saved_filename")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--datasetPath", required=True)
    parser.add_argument("--datasetMetaFile_train", required=True)
    parser.add_argument("--datasetMetaFile_val", required=True)

    parser.add_argument("--batchSize", default=4, type=int)
    parser.add_argument("--hopSize", required=False, type=float)
    parser.add_argument("--chunkSize", required=False, type=float)
    parser.add_argument("--dataLoaderWorkers", default=2, type=int)
    parser.add_argument("--gradClippingQuantile", default=0.8, type=float)

    parser.add_argument("--max_lr", default=5e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--nIter", default=500000, type=int)
    parser.add_argument(
        "--modelConf", required=True, help="the path to the model conf file"
    )
    parser.add_argument("--augment", action="store_true", help="do data augmentation")
    parser.add_argument("--noiseFolder", required=False)
    parser.add_argument("--irFolder", required=False)

    args = parser.parse_args()
    saved_filename = args.saved_filename

    runSeed = int(time.time())

    train(0, saved_filename, runSeed, args)
