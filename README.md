# StratigraPy

StratigraPy is a Python package for stratigraphic modeling based on [Landlab](https://github.com/landlab/landlab).

## Installation

You can directly install StratigraPy from pip:

    pip install stratigrapy

Or from GitHub using pip:

    pip install git+https://github.com/grongier/stratigrapy.git

## Usage

Basic use:

```
import numpy as np
from pybarsim import BarSim2D
import matplotlib.pyplot as plt

# Set the parameters
run_time = 10000.
barsim = BarSim2D(np.linspace(1000., 900., 200),
                  np.array([(0., 950.), (run_time, 998.)]),
                  np.array([(0., 25.), (run_time, 5.)]),
                  spacing=100.)
# Run the simulation
barsim.run(run_time=10000., dt_fair_weather=15., dt_storm=1.)
# Interpolate the outputs into a regular grid
barsim.regrid(900., 1000., 0.5)
# Compute the mean grain size
barsim.finalize(on='record')
# Plot the median grid size in the regular grid
barsim.record_['Mean grain size'].plot(figsize=(12, 4))
plt.show()
```

For a more complete example, see...

## Citation

If you use StratigraPy in your research, please cite the original article(s) describing the method(s) you used (see the docstrings for the references).

## Credits

This software was written by:

| [Guillaume Rongier](https://github.com/grongier) <br>[![ORCID Badge](https://img.shields.io/badge/ORCID-A6CE39?logo=orcid&logoColor=fff&style=flat-square)](https://orcid.org/0000-0002-5910-6868)</br> |
| :---: |

## License

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest in the program StratigraPy written by the Author(s). Prof.dr.ir. S.G.J. Aarninkhof, Dean of the Faculty of Civil Engineering and Geosciences

&#169; 2025, Guillaume Rongier

This work is licensed under a MIT OSS licence, see [LICENSE](LICENSE) for more information.
