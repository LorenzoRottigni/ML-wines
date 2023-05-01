import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def debug_DF(X, filename = 'graph.png'):
    print(X)

    df = pd.DataFrame(X, columns=['phenols', 'magnesium'])

    g = sns.scatterplot(data=df, x='phenols', y='magnesium')

    # set the x and y axis limits in order to prevent the library to resize the graph
    # g.set(xlim=(-10, 200), ylim=(-10, 200))

    # plt.show()
    plt.savefig(filename)
