import fire
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_experiments.text.imdb.evaluate import evaluate_model

if __name__ == "__main__":
    fire.Fire(evaluate_model)