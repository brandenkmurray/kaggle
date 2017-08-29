# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 20:45:31 2016

@author: branden
"""
import numpy as np
import pandas as pd
import gc
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib


# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})



ts1Trans = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/data_trans/ts2Trans_v11.csv") 
pca = PCA(n_components= 100, whiten=True)
t1nn_pca = pca.fit_transform(ts1Trans.iloc[:,3:]).astype(np.float32)

#(n_components=2, perplexity=30.0, early_exaggeration=4.0, learning_rate=1000.0, n_iter=1000, n_iter_without_progress=30, min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0, random_state=None, method='barnes_hut', angle=0.5)

ts1_sub = t1nn_pca[1:500,]

model = TSNE(n_components=2, random_state=0, perplexity=30.0, verbose=1)
tsne_proj = model.fit_transform(t1nn_pca) 




def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts