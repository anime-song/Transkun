def computeParamSize(module):
    total_params = sum(p.numel() for p in module.parameters())

    # Convert to millions
    total_params_millions = total_params / 1e6
    return total_params_millions


def checkpointByPass(f, *args):
    return f(*args)


def checkpointSequentialByPass(f, n, *args):
    return f(*args)


def listToIdx(l):
    batchIndices = [idx for idx, curList in enumerate(l) for _ in curList]

    return batchIndices
