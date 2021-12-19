# Multiboat Trajectory Optimization

## shapeshifting branch

### Trajectory Planning for the Shapeshifting of Autonomous Surface Vessels
#### Gheneti, Banti; Park, Shinkyu; Kelly, Ryan; Meyers, Drew; Leoni, Pietro; Ratti, Carlo; Rus, Daniela L

- c-space computation for a vessel made of rectangles moving around another vessel made of rectangles, using shapely
- trajectory optimization algorithm for solving collision free trajectories to shapeshift in the c-space, using pydrake
- also includes features in the master branch described below

[paper presented at the International Symposium on Multi-Robot and Multi-Agent Systems](https://dspace.mit.edu/handle/1721.1/137050)

[demo video](https://www.youtube.com/watch?v=9JNuBHQdF0U)

## master branch

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





