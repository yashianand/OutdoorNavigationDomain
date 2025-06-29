# Outdoor Navigation Domain

Main entry point for running the outdoor navigation domain:
```
python run.py
```

This will
- Initialize the environment. Environment is defined in `outdoorEnv.py`.
- Run policy iteration. Code for policy iteration is in `solvers.py`.
- Generate trajectories using the computed policy. Code for generating trajectories is in `helper_functions.py` (`simulate_trajectory` function).

Note: The repository currently uses standard Python libraries and does not require any additional installations. The code is designed to be run in a Python 3 environment.