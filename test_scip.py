import math
import re
import random
import time
from netlist import Terminal, Block, Net
from Tree import Tree
from pyscipopt import Model as MD
from docplex.mp.model import Model
import pathlib
from floorplaner import FloorPlaner

random.seed(0)



if __name__ == '__main__':
	result_directory = './results/HARD/'
	pathlib.Path(result_directory).mkdir(parents=True, exist_ok=True)
	instance_name = 'n200'
	print('case: hard ' + instance_name)
	fp = FloorPlaner()
	fp.readBlocks('./GSRCbench/HARD/' + instance_name + '.blocks')
	fp.readPins('./GSRCbench/HARD/' + instance_name + '.pl')
	fp.readNets('.//GSRCbench/HARD/' + instance_name + '.nets')
	fp.ILP_floorplan_scip(instance_name=instance_name, result_directory=result_directory)

	# fp.alpha = 0.5
	# fp.init()
	# fp.SA()
