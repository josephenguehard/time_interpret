import os
import shutil


def main():
    with open("results.csv", "w") as fp:
        fp.write("Seed,Noise,Activations,Explainer,Purity\n")

    if os.path.exists("lightning_logs"):
        shutil.rmtree("lightning_logs")

    if os.path.exists("figures"):
        shutil.rmtree("figures")
    os.makedirs("figures")


if __name__ == "__main__":
    main()
