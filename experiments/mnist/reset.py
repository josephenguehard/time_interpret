import os
import shutil


def main():
    with open("results.csv", "w") as fp:
        fp.write(
            "Seed,Mode,Topk,Explainer,Accuracy Comp,Accuracy Suff,Comprehensiveness,Cross Entropy Comp,"
            "Cross Entropy Suff,Log Odds,Sufficiency,Sensitivity,Lipschitz,\n"
        )

    if os.path.exists("lightning_logs"):
        shutil.rmtree("lightning_logs")


if __name__ == "__main__":
    main()
