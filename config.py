"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Model settings. Overwrite default values defined in default_config.py here.
==========================================================================================================================================================================="""

from zen_garden.model import Config
import os

# create a config
config = Config()

## Analysis - Default dictionary
analysis = config.analysis
## Solver - Default dictionary
solver = config.solver

## Analysis - settings update compared to default values
analysis["dataset"] = os.path.join(os.path.dirname(__file__), "ZEN-Model_HP")
analysis["objective"] = "total_cost"#"regional_minimum_cost" #"total_cost"
# use greenfield or brownfield approach
analysis["use_capacities_existing"] = True

## Solver - settings update compared to default values
solver["name"] = "gurobi" # free solver
solver['DualReductions'] = 0
solver["analyze_numerics"] = True
solver["immutable_unit"] = ["hour","km"]

