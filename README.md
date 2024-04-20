# Quantum Krylov algorithm for the Lattice Schwinger model

This repository contains code used to produce results in

[L. W. Anderson, M. Kiffner, T. O'Leary, J. Crain, D. Jaksch, Solving lattice gauge theories using the quantum Krylov 
algorithm and qubitization. arXiv:2403.08859.](https://arxiv.org/abs/2403.08859)

We ask that anyone using this code in a publication cites the above article. 

This repository simulates the effect of 
performing quantum subspace expansion (QSE) algorithm using a Krylov basis, in the presence of finite shot noise, in 
order to compute groundstate energy of the single flavour lattice Schwinger model. We estimate and extrapolate 
the number of quantum resources required to implement this using a qubitisation procedure based on a novel linear 
combination of unitaries approach as described in the above paper.

This code implements versions of 
partitioned quantum subspace expansion [1] and thresholded subspace expansion as described by Epperly et al. [2].

## Recreating plots from the paper

The plots from the above article can be recreated (with minor cosmetic changes) using the following files

- Figures 3 and 8 can be recreated using
  `` python experiments/subspace_expansion/full_qse_from_file.py --directory experiments/results/saved_overlaps_and_groundstates_mu=1.5_x=0.5_k=0.0_no_large_states``

- Figures 4, 5, 7 and 9 can be recreated using

  ``python experiments/subspace_expansion/full_qse_with_noise.py --load-output experiments/subspace_expansion/2023-11-24_150000_pqse_outputs_final_result.csv``.

  This loads a saved version of the data resulting from applying PQSE to saved overlaps with noise added.

- Figures similar to 10 and 11 can be recreated using

    ``python experiments/subspace_expansion/full_qse_with_noise.py --directory experiments/results/saved_overlaps_and_groundstates_mu=1.5_x=0.5_k=0.0_no_large_states --partitioning False --thresholding True``

  This runs TQSE on a set of saved overlaps and applies simulated shot noise. Since the process of adding noise is 
  random (optional seed as command line argument), this will give slightly different output and fitting results than 
  those shown in Figures 10 and 11 of the paper.

- Figure 6 can be recreated using

  ``python costing/gate_costs/plot_cost.py``.

## Performing your own simulations and resource analysis

To perform your own resource analysis, the work flow is as follows.

1. For each set of system parameters (typically you'll want to run this for various system sizes `n`) as runtime 
    arguments, Run ``experiments/subspace_expansion/generate_overlaps.py`` to generate a set of overlaps required for 
    matrices H and S within the QSE procedure. 
2. Run ``experiments/subspace_expansion/calculate_groundstatess.py`` for the same set of parameters to generate true 
   groundstate energies for comparison.
3. Move all of the outputs from the above experiments into the same directory.
4. Run ``experiments/subspace_expansion/full_qse_with_noise.py``, using the directory
    you created above as command line argument ``--directory``. This will use the above overlaps, apply shot noise 
    depending on the number of calls (currently hardcoded, change manually), generate matrices S and H and apply the 
    QSE algorithm. Other command line arguments can be used to change settings of the QSE algorithm used (notably to 
    toggle between partitioned quantum subspace expansion and thresholded quantum subspace expansion.

For testing purposes, I recommend using the `--number-noise-instances` to reduce the number of noise instance (default 
100)) to reduce runtime of `experiments/subspace_expansion/full_qse_with_noise.py`. Additionally, the number of 
different calls to the block encoding procedure is hard coded to quite a large number of different values in this file.
Manually removing some value from the variable `total_calls_to_qubitisation_procedure` will also reduce compute time.

## References

[1] [T. O'Leary, L. W. Anderson, D. Jaksch, M. Kiffner. Partitioned Quantum Subspace Expansion. arXiv:2403.08868](https://arxiv.org/abs/2403.08868)

[2] [E. N. Epperly, L. Lin, Y. Nakatsukasa. A theory of quantum subspace diagonalization. SIAM Journal on Matrix Analysis and Applications, 43(3), 1263-1290 (2022)](https://arxiv.org/abs/2110.07492)

## Contact

For any queries, contact Lewis at lewis.anderson@physics.ox.ac.uk (before December 2023) or 
lewis.anderson@ibm.com (December 2023 onwards) 

## License and disclaimer

This code uses an Apache 2.0 license.

This is research quality code and may not be actively maintained. There may be bugs.