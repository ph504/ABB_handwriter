# in this file we convert the standard image matrix sequence 
# (pixels with chronogical order according to drawing reference and optimizations and predictions)
# to a robot frame coordinate system that which is necessary to command the robot movement.
import pandas as pd
import ppg_init as ppg

seq_dir = '..\\data\\pathgen out\\'
seq_fname = 'path_Num_'
for i in range(ppg.NO_DIGITS):
    ### swap columns
    # print('----------------- i =', i)
    tdf = pd.read_csv(seq_dir+
        seq_fname+str(i)+'.csv')

    # change matrix form to cartesian coordination.
    colList = list(tdf.columns)
    colList[0], colList[1] =  colList[1], colList[0]
    tdf = tdf[colList]
    tdf.rename(columns={'x': 'y',
                      'y': 'x'},
          inplace=True, errors='raise')
    tdf['y'] = ppg.LOW_RESOLUTION_IMG_SIZE - tdf['y'] - 1
    # print(tdf, '\n')
    
    # colList = list(tdf.columns)
    # colList[0], colList[1] =  colList[1], colList[0]
    # tdf = tdf[colList]
    
    # tdf.rename(columns={'x': 'y',
    #                   'y': 'x'},
    #            inplace=True, errors='raise')
    # tdf['y'] *= -1
    ### moving center to the center.
    tdf += -4.5 
    # print('before scaling', '\n')
    # print(tdf,'\n')
    ### mapping to the real world coordinate system 
    scalar = ppg.PAPER_SIZE / (ppg.LOW_RESOLUTION_IMG_SIZE + 2*ppg.LOW_RESOLUTION_IMG_MARGIN)
    # tdf += 1
    tdf *= scalar
    # print('after scaling', '\n')
    # print('scalar = ', scalar, '\n')
    # print(tdf,'\n')
    
    colList = list(tdf.columns)
    colList[0], colList[1] =  colList[1], colList[0]
    tdf = tdf[colList]
    tdf.rename(columns={'x': 'y',
                      'y': 'x'},
          inplace=True, errors='raise')
    
    tdf['y'] *= -1
    # print(tdf,'\n')
    tdf['x'] += ppg.ROBOT_PAPER_CX
    tdf['y'] += ppg.ROBOT_PAPER_CY
    # print(tdf,'\n')
    ### save to file
    tdf.to_csv('..\\data\\path(t) numbers\\patht_Num_'+str(i)+'.csv')