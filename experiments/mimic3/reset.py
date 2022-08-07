import os
import shutil


def main():
    with open("results.csv", "w") as fp:
        fp.write(
            "Fold,Seed,Explainer,Accuracy,Comprehensiveness,Cross Entropy,Log Odds,Sufficiency\n"
        )

    if os.path.exists("lightning_logs"):
        shutil.rmtree("lightning_logs")


if __name__ == "__main__":
    main()
