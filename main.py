import argparse
import copy
import os
import random

import numpy as np
import torch
from tqdm import trange
from dgl import random as dgl_random

from datasets import dataloader
from networks.init import init_model
from optim import Corrgraphtrainer


def set_seed(seed: int):
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl_random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_args():
    parser = argparse.ArgumentParser(description="GEL for Graph Anomaly Detection")

    # dataset
    parser.add_argument("--dataset", type=str, default="Disney")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="~/datasets/GAD/mat",
        help="directory containing .mat datasets",
    )

    # trials / seed
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="base random seed; trial seed = seed + trial_id; -1 means random",
    )

    # model
    parser.add_argument("--module", type=str, default="GEL")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--correlation", type=str, default="CCA")
    parser.add_argument("--lamda", type=float, default=0.001)
    parser.add_argument("--gma", type=int, default=1)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--pe_dim", type=int, default=15)
    parser.add_argument("--n_epochs", type=int, default=2000)
    parser.add_argument("--n_hidden", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=3)

    # optim
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # graph
    parser.add_argument("--self_loop", action="store_true", default=True)
    parser.add_argument("--no_self_loop", dest="self_loop", action="store_false")
    parser.add_argument("--norm", action="store_true", default=False)
    parser.add_argument("--normal_class", type=int, default=0)
    parser.add_argument("--datamode", type=str, default="AD")

    # device
    parser.add_argument("--gpu", type=int, default=-1)

    # early stop
    parser.add_argument("--no_early_stop", action="store_true", default=False)
    parser.add_argument("--patience", type=int, default=60)

    args = parser.parse_args()
    args.early_stop = not args.no_early_stop
    return args


def run_one_trial(args, trial_id: int):
    trial_args = copy.deepcopy(args)

    if args.seed >= 0:
        trial_args.seed = args.seed + trial_id

    set_seed(trial_args.seed)

    data = dataloader.loader_anomaly(trial_args)
    model = init_model(trial_args, data["input_dim"], len(data["features"]))

    auc, ap, best_epoch = Corrgraphtrainer.train(trial_args, data, model)
    return {
        "trial": trial_id + 1,
        "seed": trial_args.seed,
        "auc": auc,
        "ap": ap,
        "best_epoch": best_epoch,
    }


def main():
    args = build_args()

    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    results = []
    for trial_id in trange(args.num_trials, desc="Trial", position=0, leave=True):
        result = run_one_trial(args, trial_id)
        results.append(result)

    aucs = np.array([r["auc"] for r in results], dtype=float)
    aps = np.array([r["ap"] for r in results], dtype=float)
    best_epochs = np.array([r["best_epoch"] for r in results], dtype=int)

    print("\n========== Final Results ==========")
    print(f"Dataset      : {args.dataset}")
    print(f"Data dir     : {os.path.expanduser(args.data_dir)}")
    print(f"Num trials   : {args.num_trials}")
    print(f"Base seed    : {args.seed}")
    print("-----------------------------------")
    for r in results:
        print(
            f"Trial {r['trial']:02d} | "
            f"Seed {r['seed']:>4} | "
            f"AUC {r['auc']:.4f} | "
            f"AP {r['ap']:.4f} | "
            f"Best epoch {r['best_epoch']}"
        )
    print("-----------------------------------")
    print(f"AUC mean/std : {aucs.mean():.4f} ± {aucs.std():.4f}")
    print(f"AP  mean/std : {aps.mean():.4f} ± {aps.std():.4f}")
    print(f"Best epoch   : mean {best_epochs.mean():.1f}, max {best_epochs.max()}")
    print("===================================\n")


if __name__ == "__main__":
    main()
