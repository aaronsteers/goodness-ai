import argparse
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from goodness_ai.crew import GoodnessCrew

load_dotenv()

questions: dict = yaml.safe_load(Path("./questions.yaml").read_text())


def run():
    # Get input from argparse
    question_name: str = sys.argv[1] if len(sys.argv) > 1 else "trolley_car_as_worker"

    if question_name not in questions["questions"]:
        nl = "\n"
        print(
            f"Question not found: {question_name}.\n"
            f"Known question names: \n- {f'{nl}- '.join(sorted(list(questions['questions'].keys())))}"
        )
        sys.exit(1)

    inputs = questions["questions"][question_name]
    GoodnessCrew().crew().kickoff(inputs=inputs)
