import math
import re
import random
import time
from netlist import Terminal, Block, Net
from Tree import Tree
from pyscipopt import Model as MD
from docplex.mp.model import Model
import pathlib

random.seed(0)

class FloorPlaner:
	def __init__(self):
		self.BoundW = 0
		self.BoundH = 0
		self.A_norm = 0
		self.W_norm = 0
		self.alpha  = 0.5
		self.NumHardRectangularBlocks = 0
		self.NumSoftRectangularBlocks = 0
		self.NumTerminals = 0
		self.numNets = 0
		self.runtime = 0

		self.BlockArea = 0
		self.ChipX, self.ChipY = (0,0)
		self.illegal = True

		self.BlockDic = {}
		self.TerminalDic = {}

		self.BlockList = []
		self.NetList = []

		self.tree = None
		self.BoundW = 0
		self.BoundH = 0

	def readBlocks(self, filename):
		with open(filename) as f:
			lines = f.readlines()
		lines = [line.split() for line in lines]
		for line in lines:
			if len(line) == 0:
				continue
			elif line[0] == 'NumSoftRectangularBlocks':
				self.NumSoftRectangularBlocks = int(line[2])
			elif line[0] == 'NumHardRectilinearBlocks':
				self.NumHardRectilinearBlocks = int(line[2])
			elif line[0] == 'NumTerminals':
				self.NumTerminals = int(line[2])
			else:
				if line[1] == 'terminal':
					# x = int(line[2])
					# y = int(line[3])
					self.TerminalDic[line[0]] = Terminal(line[0])
				elif line[1] == 'hardrectilinear':
					name = line[0]
					type = line[1]

					width = int(line[7].strip('(),'))
					height = int(line[8].strip('(),'))
					b = Block(name, type, width, height)
					self.BlockDic[name] = b
					self.BlockList.append(b)
					self.TerminalDic[name] = b.t
					self.BlockArea += width*height

	def readPins(self, filename):
		with open(filename) as f:
			lines = f.readlines()
		lines = [line.split() for line in lines]
		for line in lines:
			if len(line) == 0:
				continue
			elif line[0][0] == 'p':
				terminal_name = line[0]
				terminal = self.TerminalDic[terminal_name]
				terminal.setX(int(line[1]))
				terminal.setY(int(line[2]))

	def readNets(self, filename):
		with open(filename) as f:
			for line in f:
				line = line.split()
				if len(line) == 0:
					continue
				elif line[0] == 'NumNets':
					self.NumNets = int(line[2])
				elif line[0] == 'NetDegree':
					degree = int(line[2])
					n = Net()
					for i in range(degree):
						name = f.readline().split()[0]
						n.terminalList.append(self.TerminalDic[name])
						# print("Pin {}: x={}, y={},".format(name, self.TerminalDic[name].x, self.TerminalDic[name].y))
					self.NetList.append(n)

	def Observation(self):
		deadspace = self.ChipX * self.ChipY / self.BlockArea -1
		WidthRatio  = 1-self.ChipX / self.BoundW
		HeightRatio = 1-self.ChipY / self.BoundH
		#Area = self.ChipX * self.ChipY / self.A_norm
		#Wire = self.totalwirelength()  / self.W_norm
		#return (deadspace, WidthRatio, HeightRatio, Area, Wire)
		return (deadspace, WidthRatio, HeightRatio)

	def writeResult(self, filename):
		with open(filename,'w') as f:
			self.ChipW, self.ChipH = self.tree.pack()
			f.write(str(self.getCost(0))+'\n')
			f.write(str(self.totalwirelength())+'\n')
			f.write(str(self.ChipW*self.ChipH)+'\n')
			f.write(str(self.ChipW)+' '+str(self.ChipH)+'\n')
			f.write(str(self.runtime)+'\n')
			for b in self.BlockList:
				bX = b.x + (b.h if b.r else b.w)
				bY = b.y + (b.w if b.r else b.h)
				f.write(b.t.name+' '+str(b.x)+' '+str(b.y)+' '+str(bX)+' '+str(bY)+'\n')

	def totalwirelength(self):
		wirelen = 0
		for n in self.NetList:
			wirelen += n.wirelength()
		return wirelen

	def init(self, NumMove=100):
		self.tree = Tree(self.BlockList)
		self.A_norm = 0
		self.W_norm = 0
		for i in range(NumMove):
			for p in range(5):
				self.tree.Pertubation()
			(self.ChipX, self.ChipY) = self.tree.pack()
			self.A_norm += self.ChipX*self.ChipX
			self.W_norm += self.totalwirelength()
		self.A_norm = self.A_norm/100.0
		self.W_norm = self.W_norm/100.0

	def getCost(self, Mode=1):  #Mode_0:(area,wirelength) ; Mode_1: normalized+illegal panelty
		(self.ChipX, self.ChipY) = self.tree.pack()
		if self.illegal:
			self.illegal = (self.ChipX > self.BoundW) or (self.ChipY > self.BoundH)
		if Mode == 0:
			return self.alpha*self.ChipX*self.ChipY + (1-self.alpha)*(self.totalwirelength())
		elif Mode == 1:
			cost = self.alpha*self.ChipX*self.ChipY/self.A_norm + (1-self.alpha)*(self.totalwirelength()/self.W_norm)
			cost += (self.ChipX > self.BoundW)*(self.ChipX-self.BoundW) + (self.ChipY > self.BoundH)*(self.ChipY-self.BoundH)
			return cost

	def SA(self):
		start = time.time()
		self.init()
		lastcost = self.getCost()
		bestcost = lastcost
		bestTree = self.tree.copy()
		T = 5000.0
		Coolrate = 0.99
		uphill = 0; heatup = 0
		while T > 0.0001:
			for m in range(self.NumHardRectangularBlocks * 2):
				self.tree.Pertubation()
				newcost = self.getCost(1)
				delta = newcost - lastcost
				if delta < 0:
					lastcost = newcost
					if newcost < bestcost:
						bestcost = newcost
						bestTree = self.tree.copy()
				elif random.random() < math.exp(-10.0*delta/T):
					lastcost = newcost
					if uphill > self.NumHardRectangularBlocks:
						uphill += 1
						break
				else:
					self.tree.Reverse()
			if T < 0.1 and not self.illegal:
				T = 5000
			else:
				T = T*Coolrate
			print('             \rT = '+str(T)+'\r',end='',flush=True)
		self.tree = bestTree.copy()
		self.runtime = time.time() - start
		(self.ChipX, self.ChipY) = self.tree.pack()
		print('--- SA() ---------------------------------------')
		print('Bounding Box: '+str(self.ChipX)+','+str(self.ChipY))
		print('cost_norm: '+str(self.getCost())+'\t cost: '+str(self.getCost(0)))

	def ILP_floorplan(self, obj_type='area', instance_name='n', result_directory='./results/'):
		WIDTH_REC = int((self.BlockArea / 0.5) ** 0.5)
		HEIGHT_REC = WIDTH_REC

		Enlarge_Fac = 1.0

		# Create model
		mdl = Model()

		# Create array of variables for subsquares
		vx = [mdl.integer_var(lb=0, ub=max(WIDTH_REC, HEIGHT_REC), name="x_"+ block.t.name + "_" + str(i)) for i, block in enumerate(self.BlockList)]
		vy = [mdl.integer_var(lb=0, ub=max(WIDTH_REC, HEIGHT_REC), name="y_" + block.t.name + "_" + str(i)) for i, block in enumerate(self.BlockList)]
		vz = [mdl.integer_var(lb=0, ub=1, name="z_" + block.t.name + "_" + str(i)) for i, block in enumerate(self.BlockList)] # 0 non-rotate. 1 rotate
		vw = [[mdl.integer_var(lb=0, ub=1, name="w_{}.{}".format(i, j)) for j, block in enumerate(self.BlockList)] for i, block in enumerate(self.BlockList)]
		H = mdl.integer_var(lb=0, ub=HEIGHT_REC, name="H")

		# Create dependencies between variables (add them one by one)
		# list of width and height for blocks
		BLOCK_WIDTH_LIST = []
		BLOCK_HEIGHT_LIST = []
		BLOCK_NAME_LIST = []
		n_blocks = len(self.BlockList)
		for i in range(n_blocks):
			BLOCK_WIDTH_LIST.append(self.BlockList[i].w)
			BLOCK_HEIGHT_LIST.append(self.BlockList[i].h)
			BLOCK_NAME_LIST.append(self.BlockList[i].t.name)

		for i in range(n_blocks):
			for j in range(n_blocks):
				mdl.add_constraint((vx[i] + (1 - vz[i]) * BLOCK_WIDTH_LIST[i] + vz[i] * BLOCK_HEIGHT_LIST[i] <= WIDTH_REC))
		for i in range(n_blocks):
			for j in range(n_blocks):
				mdl.add_constraint((vy[i] + (1 - vz[i]) * BLOCK_HEIGHT_LIST[i] + vz[i] * BLOCK_WIDTH_LIST[i] <= H))

		for i in range(n_blocks - 1):
			for j in range(i + 1, n_blocks):
				mdl.add_constraint((vx[i] + (1 - vz[i]) * BLOCK_WIDTH_LIST[i] + vz[i] * BLOCK_HEIGHT_LIST[i] <= vx[j] + max(
					WIDTH_REC, HEIGHT_REC) * (vw[i][j] + vw[j][i])))
		for i in range(n_blocks - 1):
			for j in range(i + 1, n_blocks):
				mdl.add_constraint((vx[i] - (1 - vz[j]) * BLOCK_WIDTH_LIST[j] - vz[j] * BLOCK_HEIGHT_LIST[j] >= vx[j] - max(
					WIDTH_REC, HEIGHT_REC) * (1 + vw[i][j] - vw[j][i])))
		for i in range(n_blocks - 1):
			for j in range(i + 1, n_blocks):
				mdl.add_constraint((vy[i] + (1 - vz[i]) * BLOCK_HEIGHT_LIST[i] + vz[i] * BLOCK_WIDTH_LIST[i] <= vy[j] + max(
					WIDTH_REC, HEIGHT_REC) * (1 - vw[i][j] + vw[j][i])))
		for i in range(n_blocks - 1):
			for j in range(i + 1, n_blocks):
				mdl.add_constraint((vy[i] - (1 - vz[j]) * BLOCK_HEIGHT_LIST[j] - vz[j] * BLOCK_WIDTH_LIST[j] >= vy[j] - max(
					WIDTH_REC, HEIGHT_REC) * (2 - vw[i][j] - vw[j][i])))

		# Set up the objective
		# objective
		if obj_type == 'area':
			mdl.minimize(H)
		# elif obj == 'ENERGY':
		#
		#     obj = mdl.minimize( mdl.sum( vol_commu[i][j] * ( vx[i]/vx[j] + vx[j]/vx[i] + vy[i]/vy[j] + vy[j]/vy[i])  \
		#                                  for i in range(n_blocks) for j in range(n_blocks) if i!= j and vol_commu[i][j]) )
		# else:
		#     fac = 3000
		#     obj = mdl.minimize( fac*H + mdl.sum( vol_commu[i][j] * ( vx[i]/vx[j] + vx[j]/vx[i] + vy[i]/vy[j] + vy[j]/vy[i])  \
		#                                  for i in range(n_blocks) for j in range(n_blocks) if i!= j and vol_commu[i][j]) )
		# mdl.add(obj)
		# -----------------------------------------------------------------------------
		# Solve the model and display the result
		# -----------------------------------------------------------------------------

		# Solve model
		TIME_LIMIT = 600
		# N_SOL_LIMIT = 1
		mdl.parameters.timelimit = TIME_LIMIT
		# mdl.parameters.mip.limits.solutions = N_SOL_LIMIT

		print('Cplex started...')
		print("Solving model....")
		mdl.solve(log_output=True)
		msol = mdl.solution
		# print(msol)
		print(mdl.solve_details)
		print("solve time =", mdl.solve_details.time)
		print("Estimated width length: ", self.totalwirelength())
		# msol.get_value_dict()

		if msol:
			import matplotlib.pyplot as plt
			import matplotlib.cm as cm
			from matplotlib.patches import Polygon
			import matplotlib.ticker as ticker

			# Plot external square
			print("Plotting squares....")
			fig, ax = plt.subplots()
			# plt.plot((0, 0), (0, HEIGHT_REC/Enlarge_Fac), (WIDTH_REC/Enlarge_Fac, HEIGHT_REC/Enlarge_Fac), (WIDTH_REC/Enlarge_Fac, 0))
			plt.xlim((0, WIDTH_REC / Enlarge_Fac))
			plt.ylim((0, HEIGHT_REC / Enlarge_Fac))

			H_val = msol.get_value(H)
			TOTAL_HEIGHT = H_val
			print("Total Area is {}".format(TOTAL_HEIGHT * WIDTH_REC / Enlarge_Fac / Enlarge_Fac))
			print("BlockArea is {}".format(self.BlockArea))
			for i in range(n_blocks):
				# Display square i
				sx, sy, sz = msol.get_value(vx[i]), msol.get_value(vy[i]), msol.get_value(vz[i])

				# update block.x (block.y) and block.terminal.x (block.terminal.y)
				self.BlockList[i].setX(sx)
				self.BlockList[i].setY(sy)

				# transform (rotation)
				if sz:
					# exchange block width and height if rotate
					w = self.BlockList[i].w
					self.BlockList[i].setWidth(self.BlockList[i].h)
					self.BlockList[i].setHeight(w)

					sx1, sx2, sy1, sy2 = sx, sx + BLOCK_HEIGHT_LIST[i], sy, sy + BLOCK_WIDTH_LIST[i]
				else:
					sx1, sx2, sy1, sy2 = sx, sx + BLOCK_WIDTH_LIST[i], sy, sy + BLOCK_HEIGHT_LIST[i]

				sx1, sx2, sy1, sy2 = sx1 / Enlarge_Fac, sx2 / Enlarge_Fac, sy1 / Enlarge_Fac, sy2 / Enlarge_Fac
				poly = Polygon([(sx1, sy1), (sx1, sy2), (sx2, sy2), (sx2, sy1)], fc=cm.Set2(float(i) / n_blocks))
				ax.add_patch(poly)
				# Display identifier of square i at its center
				ax.text(float(sx1 + sx2) / 2, float(sy1 + sy2) / 2, BLOCK_NAME_LIST[i], ha='center', va='center')
			# ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
			# ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
			plt.margins(0)
			plt.title( instance_name + ' cplex')
			fig.savefig(result_directory + instance_name + "_cplex.png")
			plt.show()

			print("Estimated width length: ", self.totalwirelength())

	def ILP_floorplan_scip(self, obj_type='area', instance_name='n', result_directory='./results/'):
		WIDTH_REC = int((self.BlockArea / 0.5) ** 0.5)
		HEIGHT_REC = WIDTH_REC

		Enlarge_Fac = 1.0

		# Create model
		model = MD(instance_name)

		# Create array of variables for subsquares
		vx = [model.addVar(vtype='I', lb=0, ub=max(WIDTH_REC, HEIGHT_REC), name="x_"+ block.t.name + "_" + str(i)) for i, block in enumerate(self.BlockList)]
		vy = [model.addVar(vtype='I', lb=0, ub=max(WIDTH_REC, HEIGHT_REC), name="y_" + block.t.name + "_" + str(i)) for i, block in enumerate(self.BlockList)]
		vz = [model.addVar(vtype='I', lb=0, ub=1, name="z_" + block.t.name + "_" + str(i)) for i, block in enumerate(self.BlockList)] # 0 non-rotate. 1 rotate
		vw = [[model.addVar(vtype='B', lb=0, ub=1, name="w_{}.{}".format(i, j)) for j, block in enumerate(self.BlockList)] for i, block in enumerate(self.BlockList)]
		H = model.addVar(vtype='I', lb=0, ub=HEIGHT_REC, name="H")

		# Create dependencies between variables (add them one by one)
		# list of width and height for blocks
		BLOCK_WIDTH_LIST = []
		BLOCK_HEIGHT_LIST = []
		BLOCK_NAME_LIST = []
		n_blocks = len(self.BlockList)
		for i in range(n_blocks):
			BLOCK_WIDTH_LIST.append(self.BlockList[i].w)
			BLOCK_HEIGHT_LIST.append(self.BlockList[i].h)
			BLOCK_NAME_LIST.append(self.BlockList[i].t.name)

		for i in range(n_blocks):
			for j in range(n_blocks):
				model.addCons((vx[i] + (1 - vz[i]) * BLOCK_WIDTH_LIST[i] + vz[i] * BLOCK_HEIGHT_LIST[i] <= WIDTH_REC))
		for i in range(n_blocks):
			for j in range(n_blocks):
				model.addCons((vy[i] + (1 - vz[i]) * BLOCK_HEIGHT_LIST[i] + vz[i] * BLOCK_WIDTH_LIST[i] <= H))

		for i in range(n_blocks - 1):
			for j in range(i + 1, n_blocks):
				model.addCons((vx[i] + (1 - vz[i]) * BLOCK_WIDTH_LIST[i] + vz[i] * BLOCK_HEIGHT_LIST[i] <= vx[j] + max(
					WIDTH_REC, HEIGHT_REC) * (vw[i][j] + vw[j][i])))
		for i in range(n_blocks - 1):
			for j in range(i + 1, n_blocks):
				model.addCons((vx[i] - (1 - vz[j]) * BLOCK_WIDTH_LIST[j] - vz[j] * BLOCK_HEIGHT_LIST[j] >= vx[j] - max(
					WIDTH_REC, HEIGHT_REC) * (1 + vw[i][j] - vw[j][i])))
		for i in range(n_blocks - 1):
			for j in range(i + 1, n_blocks):
				model.addCons((vy[i] + (1 - vz[i]) * BLOCK_HEIGHT_LIST[i] + vz[i] * BLOCK_WIDTH_LIST[i] <= vy[j] + max(
					WIDTH_REC, HEIGHT_REC) * (1 - vw[i][j] + vw[j][i])))
		for i in range(n_blocks - 1):
			for j in range(i + 1, n_blocks):
				model.addCons((vy[i] - (1 - vz[j]) * BLOCK_HEIGHT_LIST[j] - vz[j] * BLOCK_WIDTH_LIST[j] >= vy[j] - max(
					WIDTH_REC, HEIGHT_REC) * (2 - vw[i][j] - vw[j][i])))

		# Set up the objective
		# objective
		if obj_type == 'area':
			model.setObjective(H, "minimize")
		# elif obj == 'ENERGY':
		#
		#     obj = mdl.minimize( mdl.sum( vol_commu[i][j] * ( vx[i]/vx[j] + vx[j]/vx[i] + vy[i]/vy[j] + vy[j]/vy[i])  \
		#                                  for i in range(n_blocks) for j in range(n_blocks) if i!= j and vol_commu[i][j]) )
		# else:
		#     fac = 3000
		#     obj = mdl.minimize( fac*H + mdl.sum( vol_commu[i][j] * ( vx[i]/vx[j] + vx[j]/vx[i] + vy[i]/vy[j] + vy[j]/vy[i])  \
		#                                  for i in range(n_blocks) for j in range(n_blocks) if i!= j and vol_commu[i][j]) )
		# mdl.add(obj)
		# -----------------------------------------------------------------------------
		# Solve the model and display the result
		# -----------------------------------------------------------------------------

		# Solve model

		N_SOL_LIMIT = 1
		# TIME_LIMIT = 3600
		model.setParam('limits/solutions', N_SOL_LIMIT)
		# model.setParam('limits/time', TIME_LIMIT)

		print("SCIP started...")
		print("Solving model...")
		model.optimize()
		msol = model.getBestSol()
		print("Solve status: ", model.getStatus())
		# print(msol)
		print(model.getObjVal())
		print("solve time =", model.getSolvingTime())
		# msol.get_value_dict()

		if msol:

			import matplotlib.pyplot as plt
			import matplotlib.cm as cm
			from matplotlib.patches import Polygon
			import matplotlib.ticker as ticker

			# Plot external square
			print("Plotting squares....")
			fig, ax = plt.subplots()
			# plt.plot((0, 0), (0, HEIGHT_REC/Enlarge_Fac), (WIDTH_REC/Enlarge_Fac, HEIGHT_REC/Enlarge_Fac), (WIDTH_REC/Enlarge_Fac, 0))
			plt.xlim((0, WIDTH_REC / Enlarge_Fac))
			plt.ylim((0, HEIGHT_REC / Enlarge_Fac))

			H_val = model.getSolVal(msol, H)
			TOTAL_HEIGHT = H_val
			print("Total Area is {}".format(TOTAL_HEIGHT * WIDTH_REC / Enlarge_Fac / Enlarge_Fac))
			print("BlockArea is {}".format(self.BlockArea))
			print('')
			for i in range(n_blocks):
				# Display square i
				sx, sy, sz = model.getSolVal(msol, vx[i]), model.getSolVal(msol, vy[i]), model.getSolVal(msol, vz[i])

				self.BlockList[i].setX(sx)
				self.BlockList[i].setY(sy)
				# transform (rotation)
				if sz:
					# exchange block width and height if rotate
					w = self.BlockList[i].w
					self.BlockList[i].setWidth(self.BlockList[i].h)
					self.BlockList[i].setHeight(w)

					sx1, sx2, sy1, sy2 = sx, sx + BLOCK_HEIGHT_LIST[i], sy, sy + BLOCK_WIDTH_LIST[i]
				else:
					sx1, sx2, sy1, sy2 = sx, sx + BLOCK_WIDTH_LIST[i], sy, sy + BLOCK_HEIGHT_LIST[i]

				sx1, sx2, sy1, sy2 = sx1 / Enlarge_Fac, sx2 / Enlarge_Fac, sy1 / Enlarge_Fac, sy2 / Enlarge_Fac
				poly = Polygon([(sx1, sy1), (sx1, sy2), (sx2, sy2), (sx2, sy1)], fc=cm.Set2(float(i) / n_blocks))
				ax.add_patch(poly)
				# Display identifier of square i at its center
				ax.text(float(sx1 + sx2) / 2, float(sy1 + sy2) / 2, BLOCK_NAME_LIST[i], ha='center', va='center')
			# ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
			# ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
			plt.margins(0)
			plt.title(instance_name + ' scip')
			fig.savefig(result_directory + instance_name + "_scip.png")
			plt.show()

			print("Estimated width length: ", self.totalwirelength())

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
