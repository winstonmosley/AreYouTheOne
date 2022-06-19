# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import numpy.ma as ma
import itertools
import time
import imageio
import os

from plot_heatmap import heatmap, annotate_heatmap




#scale number of contestants, mostly for testing
n = 11
pgrid = ma.array(np.random.random((n, n)))

#data labels
b = ["A","B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
g = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
b = b[:n]
g = g[:n]

displaySteps = True
knownSol = False
filenames = []

def lightBeams(testCombo, trueCombo, couples):
    global pgrid
    
    #number correctly guessed
    numCorrect = np.sum(trueCombo == testCombo)
    print('\t ', numCorrect, ' correct pairs')
    
    #eliminate all impossible combos
    numTestMatch = np.sum(couples == testCombo,axis=1)
    mask = numTestMatch == numCorrect
    possCouples = couples[mask]

    #check likelihood of each combos based on remaining options
    for i in range(n):
        for j in range(n):
            pgrid[i,j] = np.sum(possCouples[:,j] == (i+1),axis=0) #num of occurances
    pgrid = pgrid/len(possCouples) #divide by total
    pgrid.mask = pgrid == 1 #mask known values

    pf = possCouples.astype(np.float16) #cast possCouples as float
    for i in range(pf.shape[0]):
        for j in range(pf.shape[1]):
            k = possCouples[i,j]-1
            pf[i,j] = pgrid.data[k,j] #set each array value to relative probability
    cumulpf = np.sum(pf,axis=1) #sum each row
    ind = np.argmax(cumulpf) #pull index of high total probability combination
    
    if displaySteps:
        fig, ax = plt.subplots()
        im, cbar = heatmap(pgrid.data, b, g, ax=ax, cmap="Purples")
        def func(x, pos):
            return "{:.2f}".format(x).replace("0.0", "").replace("0.", "").replace("1.00", "")
        annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=12,
                             textcolors=("white", "white"))
        fig.tight_layout()
        ax.set_title('Probability matrix, step %dA' % np.ceil(ii))
        
        # create file name and append it to a list
        filename = f'{ii}.png'
        filenames.append(filename)
        
        # save frame
        plt.savefig(filename, bbox_inches='tight')
        plt.show()
    
    return possCouples, ind

def truthBooth(pair, trueCombo, couples):
    global pgrid
    
    #true match?
    match = trueCombo[pair[1]] == 1+pair[0]
    print('\t True match?', match)
    
    #eliminate all impossible combos
    if match == True:
        mask = couples[:,pair[1]] == 1+pair[0]
    else:
        mask = couples[:,pair[1]] != 1+pair[0]
    possCouples = couples[mask]

    #check likelihood of each combos based on remaining options
    for i in range(n):
        for j in range(n):
            pgrid[i,j] = np.sum(possCouples[:,j] == (i+1),axis=0) #num of occurances
    pgrid = pgrid/len(possCouples) #divide by total
    pgrid.mask = pgrid == 1 #mask known values

    pf = possCouples.astype(np.float16) #cast possCouples as float
    for i in range(pf.shape[0]):
        for j in range(pf.shape[1]):
            k = possCouples[i,j]-1
            pf[i,j] = pgrid.data[k,j] #set each array value to relative probability
    cumulpf = np.sum(pf,axis=1) #sum each row
    ind = np.argmax(cumulpf) #pull index of high total probability combination
    
    if displaySteps:
        fig, ax = plt.subplots()
        im, cbar = heatmap(pgrid.data, b, g, ax=ax, cmap="Purples")
        def func(x, pos):
            return "{:.2f}".format(x).replace("0.0", "").replace("0.", "").replace("1.00", "")
        annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=12,
                             textcolors=("white", "white"))
        fig.tight_layout()
        ax.set_title('Probability matrix, step %dB' % np.ceil(ii))
        
        # create file name and append it to a list
        filename = f'{ii}.png'
        filenames.append(filename)
        
        # save frame
        plt.savefig(filename, bbox_inches='tight')
        plt.show()
    
    return possCouples, ind



ts = time.time()



#generate all possible combinations
perms = list(itertools.permutations(range(1,1+n)))
t1 = time.time() - ts

#convert to numpy array for easier calculation
couples0 = np.array(perms, dtype=np.uint8)
t2 = time.time() - ts

# initial TRUE combination (random)
trueCombo = couples0[np.random.choice(len(perms)-1,1),:][0,:]
print('True combo is:  ', trueCombo)
print('\t total possible permutations: ', "%5d" % len(couples0))

#first guess, and make it 1D
pairsLB = couples0[np.random.choice(len(perms)-1,1),:][0,:]



couplesInt2 = couples0
ii = 0
#call func to iterate through possible solutions & plot progress
while (len(couplesInt2) > 1) & (ii < 15):
    ii += .5
    print('in step ', ii, 'lightbeam pairs selected ', pairsLB)
    couplesInt1, ind = lightBeams(pairsLB, trueCombo, couplesInt2)
    print('\t possible combinations left: ', len(couplesInt1))
    
    if len(couplesInt1) == 1: #already found correct solution
        break

    
    ii += .5
    pairTB = np.unravel_index(np.argmax(pgrid), pgrid.shape)
    print('in step ', ii, 'truthbooth pair selected ', pairTB)
    couplesInt2, ind = truthBooth(pairTB, trueCombo, couplesInt1)
    print('\t possible combinations left: ', len(couplesInt2))
    
    pairsLB = couplesInt2[ind,:] #select best combination for next LB iteration
    
#duplicate last frame
filenames.append(f'{ii}.png')

#set final couples
if ii % 1 == 0:
    couples1 = couplesInt2
else:
    couples1 = couplesInt1

print('\n\nSolution found?', len(couples1) == 1)
print('\t in ', ii, ' iterations, %5.3f seconds' % (time.time()-ts))

# build gif
with imageio.get_writer('Prob_matrix_AYTO.gif', mode='I', duration = 0.3) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        
# Remove files
for filename in set(filenames):
    os.remove(filename)
