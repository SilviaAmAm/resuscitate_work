import sys
sys.path.insert(0,'/Volumes/Transcend/repositories/YAMLP/SciFlow')
import ImportData as imp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("white")
sns.set_context("poster")

datasets_X = ["/Volumes/Transcend/repositories/trainingNN/dataSets/XYQ/X_pbe.csv", "/Volumes/Transcend/repositories/trainingNN/dataSets/XYQ/X_cc.csv", "/Volumes/Transcend/repositories/trainingNN/dataSets/XYQ/X_b3lyp.csv"]
datasets_y = ["/Volumes/Transcend/repositories/trainingNN/dataSets/XYQ/Y_pbe.csv", "/Volumes/Transcend/repositories/trainingNN/dataSets/XYQ/Y_cc.csv", "/Volumes/Transcend/repositories/trainingNN/dataSets/XYQ/Y_b3lyp.csv"]

for i, name in enumerate(["PBE", "B3LYP", "CC"]):

    X = imp.loadX(datasets_X[i])
    y = imp.loadY(datasets_y[i])

    print(len(X))

    df_X = pd.DataFrame(X)
    columns = list(range(0,28,4))

    df_clean = df_X.drop(columns, axis=1)
    print(df_X.shape, df_clean.shape)

    X_clean = df_clean.as_matrix()
    print(X_clean.shape)

    X_clus = X_clean
    y_clus = y*2625.5

    df_X_clus = pd.DataFrame(X_clus)
    df_X_clus.columns=['C1x', 'C1y', 'C1z', 'H1x', 'H1y', 'H1z', 'H2x', 'H2y', 'H2z', 'H3x', 'H3y', 'H3z', 'H4x', 'H4y', 'H4z', 'C2x', 'C2y', 'C2z', 'Nx', 'Ny', 'Nz']
    df_X_clus['CC_dist'] = np.sqrt((df_X_clus.C1x - df_X_clus.C2x)**2 + (df_X_clus.C1y - df_X_clus.C2y)**2 + (df_X_clus.C1z - df_X_clus.C2z)**2)
    df_X_clus['ene'] = y_clus.reshape((len(y_clus),1))

    x = df_X_clus.as_matrix(columns=['CC_dist'])

    fig, ax = plt.subplots(figsize=(12,8))
    plt.scatter(x,y_clus, alpha=0.4, s=20)
    # sns.lmplot('CC_dist','ene', data=df_X_clus, scatter_kws={"s": 20, "alpha": 0.6}, fit_reg=False)
    ax.set_xlabel('CC distance (Angstroms)')
    ax.set_ylabel("Energy (kJ/mol)")
    plt.tight_layout()
    plt.savefig("../plots/"+name+"_data.png", dpi=200)
    # plt.show()

