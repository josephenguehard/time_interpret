import os
import shutil

from argparse import ArgumentParser


def main(experiment: str):
    if experiment == "main":
        with open("results.csv", "w") as fp:
            fp.write(
                "Seed,Fold,Explainer,AUP,AUR,Information,Entropy,AUROC,AUPRC\n"
            )
    elif experiment == "extremal_mask_params":
        with open("bayes_mask_params.csv", "w") as fp:
            fp.write("Metric,Distribution,Hard,Model,Eps\n")
    else:
        raise NotImplementedError

    if os.path.exists("lightning_logs"):
        shutil.rmtree("lightning_logs")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        required=True,
        help="Name of the experiment. Either 'main' or 'extremal_mask_params'.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(experiment=args.experiment)
