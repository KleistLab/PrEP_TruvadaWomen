# prophylaxis estimator
>This repo contains the codes that were used to generate the 
> results of the top-down and bottom-up approaches.  

## Table of Contents
- [Top-down](#Top-down)
- [Bottom-up](#Bottom-up)

## Top-down
In the folder 'top_down' there are several scripts for different purposes:
* simulation_clinical.py: contains functions for the simulation of clinical trials. 
* run_simulation.py: call function in simulation_clinical.py, run the simulations and return 
the number of infections for each hypothesis. 

## Bottom-up
The folder 'bottom_up' contains scripts and packages for the 'bottom-up' modelling 
in the paper. 
* There is a package  called 'Vectorized_clean' to compute the prophylactic efficacy 
in a vectorized way, i.e. this package can compute the PrEP efficacy trajectory of multiple 
regimens for multiple individuals in a single run. For detailed usage of this package you can 
check the examples in 'bottom_up/Vectorized_clean/example.ipynb'. 
* PrEP efficacy for different hypotheses ... ...  


