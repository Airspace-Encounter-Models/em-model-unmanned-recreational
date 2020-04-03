# em-model-hobbydrone

Repository for generating hobby drone trajectories and writing them to output CSV files. Code originally written by E. R. Mueller and interface was written by S. M. Katz. Hobby drone trajectories are generated by sampling a Bayesian network trained on telemetry log files of multi-rotor aircraft trajectories. The flight data was from DroneShare.com, which was a website in which hobbyists could upload their telemetry log files. Data from over 75,000 flights was publicly available for download. For more information about this model, see the following citation:

E. R. Mueller and M. J. Kochenderfer, “Simulation comparison of collision avoidance algorithms for small multi-rotor aircraft,” in AIAA Modeling and Simulation Technologies Conference, 2016, p. 3674

This code was developed on Julia1.1.

## Quick Start Guide
Ensure that the following packages are installed in your current version of Julia: `BayesNets`, `LightGraphs`, `DataFrames`, `CSV`, and `JLD2`. To install any of these packages that have not yet been installed, simply run:

```julia
include("startup.jl")
```

The following lines will generate a trajectory file titled "test.csv" with data for a hobby drone trajectory with a time step of 1 second. 

```julia
include("HobbyDroneInterface.jl")
generate_trajectory_file(1.0, "test.csv")
```

The default trajectory time length is 120 seconds. Use the `tMax` variable to generate trajectories of a different length. The follow line will generate a trajectory that is 100 seconds in length.

```julia
generate_trajectory_file(1.0, "test.csv", tMax=100.0)
```

If you want to generate more than 1 trajectory per file, you can add this as the last required argument. The following line will generate a file called "test.csv" with 3 trajectories.

```julia
generate_trajectory_file(1.0, "test.csv", 3)
```

The first column of the resulting CSV file will be the time in seconds and the following three columns will be the x-, y-, and z-position respectively in feet. To test on your system, include the `RUN_HD.jl` script. It should generate a file in the output folder that matches the "hd_traj_ex.csv" file in the examples folder.

## Key Functions
`generate_HD(initBN::BayesNet, tranBN::BayesNet)` - function that interfaces with the original hobby drone code to generate a trajectory. The keyword arguments allow for setting things like initial evidence, time step, maximum time, and initial time.

`convert_to_position(stHistory::Array{IntruderState})` - converts output of hobby drone model to just get the trajectory positions

## File Descriptions
`setup.jl` - file that installs required packages that you do not currently have

`HobbyDroneInterface.jl` - Wrapper code for hobby drone model. Contains functions for generating trajectories and writing to files.

`CreateHobbyDroneTrajectories.jl` - file containing code for sampling the Bayesian network to generate a trajectory.

`iBN_data.jld2` - contains the initial network for the dynamic Bayesian network model

`tBN_data.jld2` - contains the transition network for the dynamic Bayesian network model

`run_hd.jl` - sample script for generating hobby drone trajectory files

`sample_hd_traj.csv` - sample model output for one hobby drone trajectory of length 120 seconds
