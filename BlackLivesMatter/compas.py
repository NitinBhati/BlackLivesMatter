"""Computing fairness measures on COMPAS data set."""

import argparse
import math
import numpy as np
import pandas as pd
import pylab
import fairness
import fairness_plotter

import application
import flask
import pandas as pd




# if __name__ != '__main__':
#   pylab.ion()
# filename = app.filename
# print filename
DATA_DIR='C:/Users/Nitin_Bhati/fairness-Compas/flaskr/uploads/'
# print "Hello world"
def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument('--data_location',
                 type=str,
                 default=DATA_DIR)
  p.add_argument('-t', '--test',
                 action='store_true',
                 default=False)
  p.add_argument('--file_type',
                 action='store',
                 default='pdf')
  p.add_argument('-p', '--plots',
                 action='store')
  return p.parse_args()
# print "Hello world 2"
# Should we use the location-based version?
# filename = app.filename
races = 'White Black'.split()# files = dict(scores=filename)
i=''
j=''
k=''
l=''
def cleanup_frame(frame):
  """Make the columns have better names, and ordered in a better order"""
  frame.race = frame.race.replace({'Caucasian': 'White', 'African-American': 'Black'})
  frame = frame[(frame.c_charge_degree != 'O') &  # Ordinary traffic offences
                (~frame.score_text.isnull()) &
                (frame.is_recid != -1) &         #Missing data
                (frame.days_b_screening_arrest.abs() <= 30)]  # Possibly wrong offense
  frame = frame[frame.race.isin(races)]
  # i=j
  global i,j,k,l
  # j= frame.ix[:,i]
  # print j
  i = frame[sensitive_attr_no]
  # Unique_values = i.unique()
  # print Unique_values
  j = frame[true_outcome_no]
  k = frame[direct_attr_no]
  l = frame[pred_outcome_no]
  # print i
  # print j
  # print k
  # print l

  return (frame)
# races = i.split()

# print "Hello world 3"
def parse_data(data_dir=DATA_DIR):
  """Parse sqf data set."""
  # files = dict(scores=filename)
  frame = cleanup_frame(pd.DataFrame.from_csv(data_dir+files['scores']))
  # print frame
  return frame

def data_to_table(data):
  obj = parse_data()
  return data.groupby([sensitive_attr_no,true_outcome_no,direct_attr_no]).count()[pred_outcome_no]

def get_bounds(lo=0, hi=0): #mi=0
  # n = lo + hi + mi
  n= lo + hi
  if (n==0):
    for i in range(0,n+1):
      break
  else:
    print n
    p = hi * 1. / n
    print p
    std = (n*p * (1-p))**.5
    print std
    return (hi * 1./n, std * 1./n)

def table_to_rates(table):
  answer = {}
  for race in table.index.levels[0]:
    vals_for_race = table[race]
    print vals_for_race
    scores = vals_for_race.index.levels[0]
    # print scores
    answer[race] = np.array([get_bounds(*vals_for_race[s]) for s in scores]).T
  # print answer
  return answer

def plot_table(table):
  pylab.clf()
  rates = table_to_rates(table)
  for race in rates:
    means, stds = rates[race]
    print means
    x = np.arange(len(means))+1
    l,=pylab.plot(x, means, label=race, marker='o')
    pylab.fill_between(x, means-stds, means+stds, alpha=0.2, color=l.get_color())
  pylab.xlabel('Score decile')
  pylab.ylabel('Recidivism rate')
  pylab.legend(loc=0)

def get_datapair(table, interpolate=True):
  cdfs = pd.DataFrame(columns=races)
  perf = pd.DataFrame(columns=races)
  for race in races:
    goods = table[race][:,0][::-1]
    bads = table[race][:,1][::-1]
    perf[race] = goods * 1. / (bads + goods)
    cdfs[race] = (goods + bads).cumsum() * 1./ (goods + bads).sum()
  perf.index = perf.index
  cdfs.index = cdfs.index

  if interpolate:
    idx = np.arange(1, 11, 1./2**5)[::-1]
    cdfs = cdfs.reindex(idx)
    cdfs.values[0] = 0
    cdfs = cdfs.interpolate()
    perf = perf.reindex(idx, method='bfill')
  return (cdfs, perf)

def get_plotter_from_table(table):
  cdfs, perf = get_datapair(table)
  totals = {r:table[r].sum() for r in races}
  fdata = fairness.FairnessData(cdfs, perf, totals)
  labels = dict(success='non-recidivism',
                success_people = 'non-recidivist',
                score='COMPAS score',
                longscore='COMPAS recidivism score',
                fail_people='recidivist',
                classifier_outcome='below threshold'
  )
  plotter = fairness_plotter.FairnessPlotter(fdata, labels)
  return plotter
def get_plotter_table():
  data = parse_data()
  table = data_to_table(data)
  plotter = get_plotter_from_table(table)
  return plotter, table
# print "Hello world 4"
#  filename = 2" "
# global filename
# filename1 = ''
files = 0
sensitive_attr_no = 0
true_outcome_no = 0
pred_outcome_no = 0
direct_attr_no = 0
def main(filename, sensitive_attr_col,true_outcome,predicted_outcome,direct_attribute):
  global filename1
  # global filename
  filename1 = filename
  # print filename1
  global files
  global sensitive_attr_no
  global true_outcome_no
  global pred_outcome_no
  global direct_attr_no
  sensitive_attr_no = sensitive_attr_col
  true_outcome_no = true_outcome
  pred_outcome_no = predicted_outcome
  direct_attr_no = direct_attribute
  print sensitive_attr_no
  print true_outcome_no
  print pred_outcome_no
  print direct_attr_no
  files = dict(scores=filename1)

  args=parse_args()
  fairness_plotter.rc('lines', markersize='8')

  plotter, table = get_plotter_table()
  plot_table(table)
  plotter.addfunc('marginal.performance', lambda: pylab.xticks(np.arange(1.5, 11.5), range(1, 11)))
  plotter.addfunc('marginal.cdf', lambda: pylab.xticks(np.arange(2, 12), range(1, 11)))
  plotter.addfunc('marginal.cdf', lambda: pylab.xlim((1, 11)))
  plotter.addfunc('marginal.performance', lambda: pylab.ylim((0, 100)))
  plotter.marked_indices = range(1, 11)
  plotter.plot_figures(2./3, None if args.test else 'figs/compas-',
                       args.plots.split() if args.plots else None,
                       plot_kwargs=dict(roc=dict(xlim=(0.1, 0.6), ylim=(0.3, 0.8), s=2)),
                       file_type=args.file_type,
  )
  fairness_plotter.return_resultset()
  # return data_to_table(data)
  # print "Hello world end"
# if __name__ == '__main__':
#   main(parse_args())
