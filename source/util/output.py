"""
Copyright 2018 ICG, Graz University of Technology

This file is part of MURAUER.

MURAUER is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

MURAUER is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with MURAUER.  If not, see <http://www.gnu.org/licenses/>.
"""

from pycrayon import CrayonClient

    
def create_crayon_logger(exp_name, port=8889):
    """
    """
    # Connect Crayon Logger (TensorBoard "wrapper") to the server
    cc = CrayonClient(hostname="localhost", port=port)
    tb_log_exp_name = exp_name
    # Remove previous experiment
    try:
        cc.remove_experiment(tb_log_exp_name)
    except ValueError:
        # experiment doesn't already exist - nothing to be done here
        print("Experiment '{}' didn't exist already (nothing to be done).".format(\
                tb_log_exp_name))
    # Create a new experiment
    tb_log = cc.create_experiment(tb_log_exp_name)
    return tb_log
    