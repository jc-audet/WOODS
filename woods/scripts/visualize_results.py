import os
import json
import argparse

from woods.lib import utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Visualize the results from a training run')
    parser.add_argument('mode', nargs='?', type=str, default=['plot', 'print'])
    parser.add_argument('--result_path', type=str, required=True)
    flags = parser.parse_args()

    # Print result table in the same style as training (Epoch and step time are unavailable)
    if 'print' in flags.mode:
        utils.print_results(flags.result_path)

    # Plot progression of loss and acc through training step
    if 'plot' in flags.mode:
        utils.plot_results(flags.result_path)
    