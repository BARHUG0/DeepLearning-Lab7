import argparse

from .common import RESULTS, directories, environment, save_json


def main():
    parser = argparse.ArgumentParser(description="Reproduce the embeddings lab one stage at a time.")
    parser.add_argument("stage", choices=("prepare", "pilot", "embeddings", "arithmetic", "classification", "final", "all"))
    arguments = parser.parse_args()
    directories()
    save_json(RESULTS / "environment.json", environment())
    from .preprocessing import prepare
    from .sgns import pilot
    from .experiments import arithmetic, train_embeddings
    from .classification import classification, final_evaluation

    stages = {
        "prepare": prepare, "pilot": pilot, "embeddings": train_embeddings,
        "arithmetic": arithmetic, "classification": classification, "final": final_evaluation,
    }
    if arguments.stage == "all":
        for function in stages.values():
            function()
    else:
        stages[arguments.stage]()


if __name__ == "__main__":
    main()
