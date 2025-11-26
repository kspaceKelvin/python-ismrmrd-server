import os

import numpy as np
from helpers import MAX, mrd_data, checkers, write_example


def test_invert(tmp_path):
    data = checkers(4,4,4)
    examples = write_example(data, tmp_path)
    in_file = examples["mrd"]
    out_file = tmp_path / "out.h5"

    os.environ["INFILE"] = str(in_file)
    os.environ["OUTFILE"] = str(out_file)
    os.system("./single-shot.sh")

    # Read mrd files and extract numpy matrix
    in_data = mrd_data(in_file)
    out_data = mrd_data(out_file)
    abs_diff = np.abs(out_data - in_data)
    assert in_data[0, 0, 0] == MAX, "input as expected is 0"
    assert out_data[0, 0, 0] == 0, "output is inverted"
    assert np.all(abs_diff == MAX), "all voxels inverted"
