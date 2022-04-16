#Đào Thuỳ Dương - 1851061470
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier

X1 = [[2.37319011,1.71875981], [1.51261889,1.40558943], [2.4696794,2.02144973], [1.78736889,1.29380961], [1.81231157, 1.56119497], [2.03717355,1.93397133], [1.53790057,1.87434722], [2.29312867,2.76537389], [1.38805594,1.86419379], [1.57279694, 0.9070734]]
y1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
X2 = [[3.42746579,0.71254431], [4.24760864,2.39846497], [3.33595491,1.61731637], [3.69420104,1.94273986], [4.53897645,2.54957308], [3.3071994,0.19362396], [4.13924705,2.09561534], [4.47383468,2.41269466], [4.00512009,1.89290099], [4.28205624,1.79675607]]
y2 = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
X = np.array(X1 + X2)
y = y1 + y2



clf = SVC(kernel='linear', C=1E10)
clf.fit(X, y)

def plot_svc_decision_function(clf, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = clf.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(clf.support_vectors_[:, 0],
                   clf.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='brg');
plot_svc_decision_function(clf)
plt.show()



print('Du doan mau moi la: ', clf.predict([[5.28205624,2.79675607]]))

