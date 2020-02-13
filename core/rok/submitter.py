from pathlib import Path
from time import strftime

import pandas as pd
import wandb


def main():
    project = 'CodeSearchNet'
    name = 'CSN-%s' % strftime('%Y-%m-%d')
    wandb.init(project=project, name=name)

    in_path = Path.cwd().parent / 'model_predictions.csv'
    out_path = Path(wandb.run.dir) / 'model_predictions.csv'
    df = pd.read_csv(in_path)
    df.to_csv(out_path, index=False)


if __name__ == '__main__':
    main()
