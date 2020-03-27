import matplotlib.pyplot as plt

def plotter(vlist, lname):
  vr = vlist[0]
  vn = vlist[1]
  plt.figure(lname)
  plt.plot(range(1, 1+len(vr)), vr,
           range(1, 1+len(vn)), vn,
           linewidth=2, linestyle='-', marker='o')
  plt.legend(('rep', 'nor'))
  plt.grid()
  _ = plt.xlabel('Epoch', fontsize=14)
  _ = plt.ylabel(lname, fontsize=14)
  return
