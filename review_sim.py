import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
# from time import time
import time
from pathlib import Path

from utils.sweep import run_sweep

DATA_PATH = Path('data') / '300, PD, 0.001.csv'

if __name__ == '__main__':
    start = time.perf_counter()
    df = run_sweep(theta_values=[0.5], M=300, N=300, Dr=0.5, Dg=0.5, epsilon=0.001, grid_step=0.1, n_jobs=6, outfile=DATA_PATH)
    end = time.perf_counter()
    print(f"執行時間: {end - start} 秒")

