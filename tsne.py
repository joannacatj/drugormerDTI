import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
dataset=pd.read_csv(r'G:\序列预测相关\模型\DrugormerDTI\file\embeddings_human.csv')
#dataset=dataset.dropna()
dataset=1.0*(dataset - dataset.mean())/dataset.std()
dataset=dataset.fillna(0)
#data=dataset.drop('label',axis=1)
#data_zs = 1.0*(data - data.mean())/data.std()
#label=dataset['label']
tsne = TSNE(random_state=105,n_components=2)
tsne.fit_transform(dataset)  # 用标准化数据进行数据降维
tsne = pd.DataFrame(tsne.embedding_)
estimator = KMeans(n_clusters=2, random_state=777)  # 构造聚类器,设定随机种子
estimator.fit(tsne)  # 聚类
label=estimator.labels_
import numpy as np

from sklearn.manifold import TSNE
# For the UCI ML handwritten digits dataset
from sklearn.datasets import load_digits

# Import matplotlib for plotting graphs ans seaborn for attractive graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns

palette = np.array(sns.color_palette("pink", 6))
print(palette)
def plot(x, colors):
    # Choosing color palette
    # https://seaborn.pydata.org/generated/seaborn.color_palette.html

    #print(palette)
    palette=np.array([[0.98595925,0.72930411,0.74042291],[0.73094963,0.83947712,0.92132257]])
    #print(palette)
    # pastel, husl, and so on

    # Create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int8)])
    ax.legend()
    '''
    # Add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
        txts.append(txt)
    '''
    plt.savefig(r'tsne-GPCR_att.png', dpi=120)
    return f, ax

# There are 10 classes (0 to 9) with alomst 180 images in each class
# The images are 8x8 and hence 64 pixels(dimensions)

# Place the arrays of data of each digit on top of each other and store in X
X = np.vstack([dataset[label==i] for i in range(2)])
# Place the arrays of data of each target digit by the side of each other continuosly and store in Y
Y = np.hstack([label[label==i] for i in range(2)])

# Implementing the TSNE Function - ah Scikit learn makes it so easy!
final = TSNE(perplexity=30).fit_transform(X)
# Play around with varying the parameters like perplexity, random_state to get different plots

plot(final, Y)
