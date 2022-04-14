# LAMBDA
<img width="100%" src="https://imgur.com/0G3VKle.gif"><img width="100%" src="https://imgur.com/zdyuRdN.gif">

A repository for the implementation of the paper **Constrained Policy Optimization via Bayesian World Models** (Yarden As, Ilnura Usmanova, Sebastian Curi, Andreas Krause, ICLR 2022). Please see our [paper (arXiv)](https://arxiv.org/abs/2201.09802) for further details.
To cite, please use:
```
@article{as2022constrained,
  title={Constrained Policy Optimization via Bayesian World Models},
  author={As, Yarden and Usmanova, Ilnura and Curi, Sebastian and Krause, Andreas},
  journal={arXiv preprint arXiv:2201.09802},
  year={2022}
}
```

## Idea
By taking a Bayesian perspective, LAMBDA learns a world model and uses it to generate sequences using different posterior samples of the world model parameters. Following that, it chooses and learns from the optimistic sequences how to solve the task and from the pessimistic sequences how to adhere to safety restrictions.

<img width="100%" src="https://imgur.com/W5n1wuV.gif">

## Running and plotting
Install dependencies (this may take more than an hour):
```
conda create -n lambda python=3.6
conda activate lambda
pip3 install .
```
Run experiments:
```
python3 experiments/train.py --log_dir results/point_goal2/314 --environment sgym_Safexp-PointGoal2-v0 --total_training_steps 1000000 --safety
```
Plot:
```
python3 experiments/plot.py --data_path results/
```
where the script expects the following directory tree structure:
```
results
├── algo1
│   └── environment1
│       └── experiment1
│       └── ...
│   └── ...
└── algo2
    └── environment1
        ├── experiment1
        └── experiment2
```

## Acknowledgement
[Dreamer](https://github.com/danijar/dreamer) codebase which served as a starting point for this github repo
