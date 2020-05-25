import logreg_cd as lrcd
from gimme_data import gimme_data
from plotter import plotter


# variables
n, d, m = 1000, 20, 10
myeps = 1e-3

# to test effects of changing size of dataset
[X, u, y] = gimme_data(n, d, m)

loss_list = []
err_list = []

# call algorithm first time
# X is data, y are labels (?)
# epsilon is accuracy to stop
[wr, l, e] = lrcd.logistic_regression_cd(X, y, eps=myeps)
loss_list.append(l)
err_list.append(e)

# run algo again without a replacement
# supossed to run faster like this
[wn, l, e] = lrcd.logistic_regression_cd(X, y, eps=myeps, replace=0)
loss_list.append(l)
err_list.append(e)

# plot stuff
plotter(loss_list, 'Loss')
plotter(err_list, 'Error')
