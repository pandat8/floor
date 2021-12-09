# Author Name: Manqing Mao; Email address: Manqing.Mao@asu.edu
import math
from docplex.mp.model import Model
import docplex.cp.utils_visu as visu
import pandas as pd
#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------
CONFIG = 'download'
SCHEDULER = 'ETF'
RATE = '150'
# Size of the total area

HEIGHT_REC = 150
WIDTH_REC = 100
obj = 'AREA'
system = 'baseline'

if system == 'baseline':
    NUM_RESOURCES = 12
    SIZE_WIDTH = [36, 36, 36, 36, 18, 18, 18, 18, 36, 36, 36, 36]
    SIZE_HEIGHT = [22, 22, 22, 22, 11, 11, 11, 11, 22, 22, 22, 22]
    RESOUCE_dict = {0:'A72-0', 1:'A72-1', 2:'A72-2', 3:'A72-3', \
                    4:'A53-0', 5:'A53-1', 6:'A53-3', 7:'A53-4', \
                    8:'Cache-1', 9:'Cache-2', 10:'Cache-3', 11:'Cache-4'}
elif system == '1FFT+1DAP+1M':
    NUM_RESOURCES = 15
    SIZE_WIDTH = [36, 36, 36, 36, 18, 18, 18, 18, 36, 36, 36, 36, 25, 8, 17]
    SIZE_HEIGHT = [22, 22, 22, 22, 11, 11, 11, 11, 22, 22, 22, 22, 15, 5, 10]
    RESOUCE_dict = {0:'A72-0', 1:'A72-1', 2:'A72-2', 3:'A72-3', \
                    4:'A53-0', 5:'A53-1', 6:'A53-3', 7:'A53-4', \
                    8:'Cache-1', 9:'Cache-2', 10:'Cache-3', 11:'Cache-4', \
                    12:'FFT', 13:'DAP', 14:'Memory',}
elif system == '2FFT+2DAP+1M':
    NUM_RESOURCES = 17
    SIZE_WIDTH = [36, 36, 36, 36, 18, 18, 18, 18, 36, 36, 36, 36, 25, 25, 8, 8, 17]
    SIZE_HEIGHT = [22, 22, 22, 22, 11, 11, 11, 11, 22, 22, 22, 22, 15, 15, 5, 5, 10]
    RESOUCE_dict = {0:'A72-0', 1:'A72-1', 2:'A72-2', 3:'A72-3', \
                    4:'A53-0', 5:'A53-1', 6:'A53-3', 7:'A53-4', \
                    8:'Cache-1', 9:'Cache-2', 10:'Cache-3', 11:'Cache-4', \
                    12:'FFT-0', 13:'FFT-1', 14:'DAP-0', 15:'DAP-1', 16:'Memory'}
elif system == '2FFT+3DAP+1M':
    NUM_RESOURCES = 18
    SIZE_WIDTH = [36, 36, 36, 36, 18, 18, 18, 18, 36, 36, 36, 36, 25, 25, 8, 8, 8, 17]
    SIZE_HEIGHT = [22, 22, 22, 22, 11, 11, 11, 11, 22, 22, 22, 22, 15, 15, 5, 5, 5, 10]
    RESOUCE_dict = {0:'A72-0', 1:'A72-1', 2:'A72-2', 3:'A72-3', \
                    4:'A53-0', 5:'A53-1', 6:'A53-3', 7:'A53-4', \
                    8:'Cache-1', 9:'Cache-2', 10:'Cache-3', 11:'Cache-4', \
                    12:'FFT-0', 13:'FFT-1', 14:'DAP-0', 15:'DAP-1', 16:'DAP-2', 17:'Memory'}
    df = pd.read_fwf("inputs/"+CONFIG+"/trace_"+SCHEDULER+"_"+RATE+"/matrix_traffic_"+SCHEDULER+"_"+RATE+".rpt")   # read file   --- MODIFY HERE

    
if system == 'practical':
    HEIGHT_REC = 100
    WIDTH_REC = 82
    NUM_RESOURCES = 14
    SIZE_WIDTH = [36, 36, 36, 25, 8, 8, 17, 18, 18, 18, 18, 8, 8, 8]
    SIZE_HEIGHT = [22, 22, 22, 15, 5, 5, 10, 11, 11, 11, 11, 5, 5, 5]
    RESOUCE_dict = {0:'A72', 1:'A53-0', 2:'A53-1', 3:'FFT', 4:'DAP-0', 5:'DAP-1', 6:'Memory', \
                    7:'Cache-1', 8:'Cache-2', 9:'Cache-3', 10:'Cache-4', 11:'FIR', 12:'DMA', 13:'NIC'}
    df = pd.read_fwf("inputs/"+CONFIG+"/trace_"+SCHEDULER+"_"+RATE+"/matrix_traffic_"+SCHEDULER+"_"+RATE+"_"+"new.rpt")   # read file   --- MODIFY HERE
'''
if system == 'test':
    HEIGHT_REC = 80
    WIDTH_REC = 80
    NUM_RESOURCES = 8
    SIZE_WIDTH = [36, 18, 25, 25, 8, 8, 8, 17]
    SIZE_HEIGHT = [22, 11, 15, 15, 5, 5, 5, 10]
    RESOUCE_dict = {0:'A72-0', 1:'A53-0', 2:'FFT-0', 3:'FFT-1', 4:'DAP-0', 5:'DAP-1', 6:'DAP-2', 7:'Memory'}
'''
# Real sizes of the hardware resources
'''
SIZE_WIDTH = [1.79, 0.89, 0.89, 1.24, 'DAP-0', 'DAP-1', 'Memory', 1.81, 1.81, 1.81, 1.81]
SIZE_HEIGHT = [1.08, 0.54, 0.54, 0.75, 'DAP-0', 'DAP-1', 'Memory', 1.08, 1.08, 1.08, 1.08]
'''
area_estimate = sum([w*h for w,h in zip(SIZE_WIDTH, SIZE_HEIGHT)])
WIDTH_REC = int((area_estimate/0.9)**0.5)
Enlarge_Fac = 1.0


# df = df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)   # drop unnamed
# df = df.drop(range(10))   # drop the first 10 lines
# vol = df.values.tolist()
# vol_commu = [list(map(int,i)) for i in vol]
'''
vol_commu = [[0 for _ in range(NUM_RESOURCES)] for _ in range(NUM_RESOURCES)]
vol_commu[0][1] = 10000
vol_commu[0][2] = 10000
'''
#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

# Create model
mdl = Model()

# Create array of variables for subsquares
vx = [mdl.integer_var(lb=0, ub=max(WIDTH_REC,HEIGHT_REC), name="x" + str(i)) for i in range(NUM_RESOURCES)]
vy = [mdl.integer_var(lb=0, ub=max(WIDTH_REC,HEIGHT_REC), name="y" + str(i)) for i in range(NUM_RESOURCES)]
vz = [mdl.integer_var(lb=0, ub=1, name="z" + str(i)) for i in range(NUM_RESOURCES)]   # 0 non-rotate. 1 rotate
vxy = [[mdl.integer_var(lb=0, ub=1, name="xy_{}.{}".format(i, j)) for j in range(NUM_RESOURCES)] for i in range(NUM_RESOURCES)]
H = mdl.integer_var(lb=0, ub=HEIGHT_REC, name="H")


# Create dependencies between variables (add them one by one)
for i in range(NUM_RESOURCES):
    for j in range(NUM_RESOURCES):
        mdl.add_constraint((vx[i] + (1-vz[i])*SIZE_WIDTH[i] + vz[i]*SIZE_HEIGHT[i] <= WIDTH_REC))
for i in range(NUM_RESOURCES):
    for j in range(NUM_RESOURCES):
        mdl.add_constraint((vy[i] + (1-vz[i])*SIZE_HEIGHT[i] + vz[i]*SIZE_WIDTH[i] <= H))

for i in range(NUM_RESOURCES-1):
    for j in range(i+1, NUM_RESOURCES):
        mdl.add_constraint((vx[i] + (1-vz[i])*SIZE_WIDTH[i] + vz[i]*SIZE_HEIGHT[i] <= vx[j] + max(WIDTH_REC,HEIGHT_REC)*(vxy[i][j]+vxy[j][i])))
for i in range(NUM_RESOURCES-1):
    for j in range(i+1, NUM_RESOURCES):
        mdl.add_constraint((vx[i] - (1-vz[j])*SIZE_WIDTH[j] - vz[j]*SIZE_HEIGHT[j] >= vx[j] - max(WIDTH_REC,HEIGHT_REC)*(1+vxy[i][j]-vxy[j][i])))
for i in range(NUM_RESOURCES-1):
    for j in range(i+1, NUM_RESOURCES):
        mdl.add_constraint((vy[i] + (1-vz[i])*SIZE_HEIGHT[i] + vz[i]*SIZE_WIDTH[i] <= vy[j] + max(WIDTH_REC,HEIGHT_REC)*(1-vxy[i][j]+vxy[j][i])))
for i in range(NUM_RESOURCES-1):
    for j in range(i+1, NUM_RESOURCES):
        mdl.add_constraint((vy[i] - (1-vz[j])*SIZE_HEIGHT[j] - vz[j]*SIZE_WIDTH[j] >= vy[j] - max(WIDTH_REC,HEIGHT_REC)*(2-vxy[i][j]-vxy[j][i])))
        


# Set up the objective
"""
obj = mdl.minimize( mdl.sum( vol_commu[i][j] * ( mdl.max( [mdl.end_of(vx[i]) - mdl.end_of(vx[j]), mdl.end_of(vx[j]) - mdl.end_of(vx[i])] )\
                                                 + mdl.max( [mdl.start_of(vy[i]) - mdl.end_of(vy[j]), mdl.start_of(vy[j]) - mdl.end_of(vy[i])] ) ) \
                             for i in range(NUM_RESOURCES) for j in range(NUM_RESOURCES) if not vol_commu[i][j]) )

obj = mdl.minimize( mdl.sum( vol_commu[i][j] * ( mdl.max( [mdl.end_of(vx[i]) - mdl.end_of(vx[j]), mdl.end_of(vx[j]) - mdl.end_of(vx[i])] )\
                                                 + mdl.min([mdl.min([mdl.max( [mdl.start_of(vy[i]) - mdl.end_of(vy[j]), 0] ), mdl.max( [mdl.start_of(vy[j]) - mdl.end_of(vy[i]), 0] )]), 1]) ) \
                             for i in range(NUM_RESOURCES) for j in range(NUM_RESOURCES) if not vol_commu[i][j]) )

"""
# objective
if obj == 'AREA':
    obj = mdl.minimize(H)
# elif obj == 'ENERGY':
#
#     obj = mdl.minimize( mdl.sum( vol_commu[i][j] * ( vx[i]/vx[j] + vx[j]/vx[i] + vy[i]/vy[j] + vy[j]/vy[i])  \
#                                  for i in range(NUM_RESOURCES) for j in range(NUM_RESOURCES) if i!= j and vol_commu[i][j]) )
# else:
#     fac = 3000
#     obj = mdl.minimize( fac*H + mdl.sum( vol_commu[i][j] * ( vx[i]/vx[j] + vx[j]/vx[i] + vy[i]/vy[j] + vy[j]/vy[i])  \
#                                  for i in range(NUM_RESOURCES) for j in range(NUM_RESOURCES) if i!= j and vol_commu[i][j]) )
# mdl.add(obj)
#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------

# Solve model
print("Solving model....")
mdl.parameters.timelimit = 20
msol = mdl.solve()
print("Solution: ")
# msol.get_value_dict()

if msol and visu.is_visu_enabled():
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.patches import Polygon
    import matplotlib.ticker as ticker

    # Plot external square
    print("Plotting squares....")
    fig, ax = plt.subplots()
    #plt.plot((0, 0), (0, HEIGHT_REC/Enlarge_Fac), (WIDTH_REC/Enlarge_Fac, HEIGHT_REC/Enlarge_Fac), (WIDTH_REC/Enlarge_Fac, 0))
    plt.xlim((0, WIDTH_REC/Enlarge_Fac))
    plt.ylim((0, HEIGHT_REC/Enlarge_Fac))

    H_val = msol.get_value(H)
    TOTAL_HEIGHT = H_val
    print("Total Area is {}".format(TOTAL_HEIGHT*WIDTH_REC/Enlarge_Fac/Enlarge_Fac))
    for i in range(NUM_RESOURCES):
        # Display square i
        sx, sy, sz = msol.get_value(vx[i]), msol.get_value(vy[i]), msol.get_value(vz[i])

        # transform (rotation)
        if sz:
            sx1, sx2, sy1, sy2 = sx, sx+SIZE_HEIGHT[i], sy, sy+SIZE_WIDTH[i]
        else:
            sx1, sx2, sy1, sy2 = sx, sx+SIZE_WIDTH[i], sy, sy+SIZE_HEIGHT[i]

        sx1, sx2, sy1, sy2 = sx1/Enlarge_Fac, sx2/Enlarge_Fac, sy1/Enlarge_Fac, sy2/Enlarge_Fac
        poly = Polygon([(sx1, sy1), (sx1, sy2), (sx2, sy2), (sx2, sy1)], fc=cm.Set2(float(i) / NUM_RESOURCES))
        ax.add_patch(poly)
        # Display identifier of square i at its center
        ax.text(float(sx1 + sx2)/ 2, float(sy1 + sy2)/ 2, RESOUCE_dict[i], ha='center', va='center')
        #ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        #ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    plt.margins(0)
    fig.savefig("./outputs/flex/"+CONFIG+"_"+SCHEDULER+"_"+RATE+"_"+system+".png")
    plt.show()


