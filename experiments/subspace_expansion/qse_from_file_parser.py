from argparse import ArgumentParser
from distutils.util import strtobool


class QSEFromFileParser(ArgumentParser):
    def __init__(self):
        super().__init__(description="loading overlaps and groundstates to perform QSE")
        self.add_argument("--directory",
                          default="experiments/results/saved_overlaps_and_groundstates_mu=1.5_x=0.5_k=0.0_no_large_states",
                          type=str, help="Directory to load files from.")
        self.add_argument("--value-of-merit", default="energy", type=str, choices=["fidelity", "energy"],
                          help="Whether to calculate and plot energy error or fidelity.")
        self.add_argument("--number-noise-instances", type=int, default=100)
        self.add_argument("--noise_seed", type=int, default=None)
        self.add_argument("--failure-tolerance", type=float, default=0.0)
        self.add_argument("--thresholding", type=lambda x: bool(strtobool(x)), default=True)
        self.add_argument("--partitioning", type=lambda x: bool(strtobool(x)), default=False)
        self.add_argument("--load-output", type=str,
                          default="")
