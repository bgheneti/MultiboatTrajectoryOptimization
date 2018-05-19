# Multiboat Trajectory Optimization
Multiboat minimum makespan formation planning in three stages.

- Goal assignment with the Hungarian algorithm with initial state to goal high order  norm costs
- Linear and shape-based interpolation for trajectory initialization
- Direct transcription trajectory optimization using pydrake with SNOPT SQP solver

Final project for [6.832 - Underactuated Robotics](http://underactuated.csail.mit.edu/)

See Youtube for an [explanation video](https://youtu.be/kd0PPfe8hwg).

# Contents

- `.py` files include code for the above 3 stages, as well as for producing visualizations
- `final_project_visualizations.ipynb` contains code for running experiments on the .py files and saving them to `/results`
- `final_project_visualizations.experiments` for viewing result files as tables, displaying and saving boat animations, and plotting various graphs.
- `/results` contains experiment results
- `/animations` contains various experiment animations
- `/icp` contains the iterative closest point (ICP) implementation obtained from [@ClayFlanigan](https://github.com/ClayFlannigan/icp).

# Dependencies

- Main dependency is [drake](https://github.com/RobotLocomotion/drake)
- Install other python dependencies with `pip install sklearn tabular`





