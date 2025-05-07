import os, sys, fire

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_experiments.text.enwik8.evaluate import evaluate_model
if __name__ == "__main__":
    fire.Fire(evaluate_model)
