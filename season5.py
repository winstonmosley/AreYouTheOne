# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import itertools
import time

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def lightBeams(testCouples, numCorrect, posCouples):
    numTestMatch = np.sum(posCouples == testCouples,axis=1)
    mask = numTestMatch == numCorrect
    return mask

def truthbooth(couple, tf, possCouples):
    if tf == True:
        mask = possCouples[:,couple[0]-1] == couple[1]
    else:
        mask = possCouples[:,couple[0]-1] != couple[1]
    return mask



b = ["1. Andre","2. Derrick", "3. Edward", "4. Hayden", "5. Jaylan",
              "6. Joey", "7. Michael", "8. Mike", "9. Osvaldo", "10. Ozzy", "11. Tyler"]

g = ["1. Alicia", "2. Carolina", "3. Casandra", "4. Gianna", "5. Hannah",
           "6. Kam", "7. Kari", "8. Karthryn", "9. Shannon", "10. Taylor", "11. Tyranny"]










ts = time.time()



sl1 = [1, 6, 5, 10, 7, 3, 8, 2, 4, 11, 9];
slr1 = 2;

tb1 = [4, 4]
tbr1 = False

sl2 = [2, 6, 8, 7, 1, 5, 9, 10, 3, 4, 11];
slr2 = 0;

tb2 = [1, 1]
tbr2 = False

sl3 = [8, 4, 5, 10, 2, 3, 1, 6, 11, 7, 9];
slr3 = 4;

tb3 = [2, 9]
tbr3 = False

sl4 = [3, 4, 1, 2, 10, 8, 7, 6, 11, 9, 5];
slr4 = 4;

tb4 = [11, 9]
tbr4 = False

sl5 = [8, 4, 5, 9, 10, 3, 7, 6, 11, 1, 2];
slr5 = 4;

tb5 = [6, 3]
tbr5 = True

sl6 = [8, 4, 10, 9, 5, 3, 7, 6, 11, 1, 2];
slr6 = 4;

tb6 = [5, 10]
tbr6 = False

sl7 = [10, 4, 1, 8, 7, 3, 2, 6, 11, 9, 5];
slr7 = 4;

tb7 = [10, 1]
tbr7 = False

#sl8 = [9, 4, 1, 10, 11, 3, 7, 6, 5, 2, 8];
#slr8 = 5;

n=11





sl = [sl1, sl2, sl3, sl4, sl5, sl6, sl7] #, sl8]
slr = [slr1, slr2, slr3, slr4, slr5, slr6, slr7] #, slr8]
tb = [tb1, tb2, tb3, tb4, tb5, tb6, tb7]
tbr = [tbr1, tbr2, tbr3, tbr4, tbr5, tbr6, tbr7]

sln = len(sl)
tbn = len(tb)


pgrid = np.random.random((n, n))

perms = list(itertools.permutations(range(1,1+n)))
t1 = time.time() - ts
array0 = np.array(perms, dtype=np.uint8)
t2 = time.time() - ts

arrayB = array0
for i in range(sln):
    maskA = lightBeams(sl[i], slr[i], arrayB)
    arrayA = arrayB[maskA]
    
    if i == sln-1 and sln > tbn:
        arrayB = arrayA
    else:
        maskB = truthbooth(tb[i], tbr[i], arrayA)
        arrayB = arrayA[maskB]

arrayF = arrayB

#how likely is each pair
for i in range(n):
    for j in range(n):
        pgrid[i,j] = np.sum(arrayF[:,j] == (i+1),axis=0)
pgrid = pgrid/len(arrayF)

pf = arrayF.astype(np.float16)

for i in range(pf.shape[0]):
    for j in range(pf.shape[1]):
        k = arrayF[i,j]-1
        pf[i,j] = pgrid[k,j]
cumulpf = np.sum(pf,axis=1)
ind = np.argmax(cumulpf)


print('Most likely combination is: ', arrayF[ind,:])
for i in range(n):
    print(g[i], 'with ', b[arrayF[ind,i]-1])


    

#hcheck = np.sum(pgrid,axis=0)
#vcheck = np.sum(pgrid,axis=1)






b = b[:n]
g = g[:n]
fig, ax = plt.subplots()
im, cbar = heatmap(pgrid, b, g, ax=ax, cmap="Purples")
def func(x, pos):
    return "{:.2f}".format(x).replace("0.0", "").replace("0.", "").replace("1.00", "")
annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=12,
                     textcolors=("white", "white"))
fig.tight_layout()
plt.show()
















print('\n\n\n')
print('total possible permutations: ', "%10.3E" % len(array0))

t3 = time.time() - ts
print('list generation took: ', t1, 'seconds')
print('array generation took: ', t2, 'seconds')
print('comparison operations took: ', t3, 'seconds')

