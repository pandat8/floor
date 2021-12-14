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
	block_types = ['HARD', 'SOFT']
	block_type = block_types[1]
	modes = ['sol-limit', 'time-limit']
	mode = modes[0]
	print('mode : ', mode)
	instance_id = 'n10'
	instance_name = block_type + '_' + instance_id

	result_directory = './results/' + block_type + '/'
	pathlib.Path(result_directory).mkdir(parents=True, exist_ok=True)

	print('case: ', instance_name)
	fp = FloorPlaner()
	fp.readBlocks('./GSRCbench/' + block_type + '/' + instance_id + '.blocks')
	fp.readPins('./GSRCbench/' + block_type + '/' + instance_id + '.pl')
	fp.readNets('./GSRCbench/' + block_type + '/' + instance_id + '.nets')

	fp.ILP_floorplan_scip(instance_name=instance_name, result_directory=result_directory, mode=mode)

	# fp.alpha = 0.5
	# fp.init()
	# fp.SA()
