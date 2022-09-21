import os
import shutil


def main():
    with open("results.csv", "w") as fp:
        fp.write(
            "Seed,Topk,Weight_fn,Explainer,Accuracy,Comprehensiveness,Cross Entropy,Log Odds,Sufficiency,"
            "Sensitivity,Lipschitz,Sensitivity Acc,Sensitivity Comp,Sensitivity CE,Sensitivity LOdds,Sensitivity Suff,"
            "Lipschitz Acc,Lipschitz Comp,Lipschitz CE,Lipschitz LOdds,Lipschitz Suff,\n"
        )

    if os.path.exists("lightning_logs"):
        shutil.rmtree("lightning_logs")


if __name__ == "__main__":
    main()
