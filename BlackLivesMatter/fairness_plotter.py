"""Plotting fairness measures"""

from __future__ import print_function
import math
import numpy as np
import pandas as pd

import pylab
from matplotlib import rc
from cycler import cycler
import matplotlib.ticker as mtick
from flask import Flask, render_template, request
from flask import Blueprint

# global frate

# from . import routes
#
# app4 = Flask(__name__)

# fairness_plot = Blueprint('fairness_plot', __name__)


SHORT = True

pct_fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
pct_formatter = mtick.FormatStrFormatter(pct_fmt)
global result_set
result_set = open("result2.txt", "w+")



def set_pct_formatter(s):
  if s == 'x':
    pylab.gca().xaxis.set_major_formatter(pct_formatter)
  elif s == 'y':
    pylab.gca().yaxis.set_major_formatter(pct_formatter)


rc('axes', prop_cycle=(cycler('color', ['b', 'g', 'r', 'c', 'y']) +
                           cycler('linestyle', ['-', '--', '-.', ':', '-'])))
rc('font',  size='17')
rc('axes', labelsize='large')
rc('lines', linewidth=3)
rc('text', usetex=True)
titlekws = dict(y=1.03)

def cdf_to_data(cdf):
  counts = 100*(cdf.values[1:] - cdf.values[:-1])
  locs = (cdf.index[1:]+ cdf.index[:-1])/2.
  points = np.repeat(locs.values, counts.astype(int))
  return points

def find_x_for_y(xs, ys, y):
  xs = np.array(xs)
  ys = np.array(ys)
  for i in range(len(xs)-1):
    (xold, yold) = xs[i], ys[i]
    (xnew, ynew) = xs[i+1], ys[i+1]
    if (yold - y) * (ynew - y) < 0:
      x = xold + ((y - yold) / (ynew - yold)) * (xnew - xold)
      return x
  return None

def index_and_interpolate(series, x):
  loc = series.index.get_loc(x, 'ffill')
  return (series[series.index[loc+1]] * (x - series.index[loc]) + series[series.index[loc]] * (series.index[loc+1]-x)) * 1. / (series.index[loc+1] - series.index[loc])
  return series[series.index[loc]]

def return_resultset(self):
  return result_set

class FairnessPlotter(object):
  default_labels = dict(success='success',
                        success_people = 'successful people',
                        score='score',
                        fail_people='failing people',
                        classifier_outcome='classified positively'
                        )

  def __init__(self, data, labels={}):
    """data should be a FairnessData instance"""
    self.data = data
    self.columns = self.data.columns

    self.labels = self.default_labels.copy()
    self.labels.update(labels)
    self.funcs = {}

  @property
  def truepos_text(self):
    if SHORT:
      return "Fraction %(success_people)s\ngetting %(classifier_outcome)s" % self.labels
    return "Fraction %(success_people)s getting\n%(classifier_outcome)s" % self.labels

  @property
  def falsepos_text(self):
    return "Fraction %(fail_people)s getting %(classifier_outcome)s" % self.labels

  def plot_curves(self, cutoff_sets=[], regular_labels=True):
    """Draws the (frac non-defaulters getting loan, frac defaulters getting loan) curve."""
    good, bad = self.data.compute_curves()
    for group in self.columns:
      pylab.plot(bad[group].values, good[group].values, label=group if regular_labels else None)
    if self.marked_indices:
      for group, l in zip(self.columns, pylab.gca().lines):
        pylab.plot(bad[group].loc[self.marked_indices].values,
                   good[group].loc[self.marked_indices].values, color=l.get_color(), marker='o', lw=0)
    for ctype, val, new_kws in cutoff_sets:
      kws = dict(color='k', marker='x', s=40, linewidths=3, zorder=3,)
      kws.update(new_kws)
      if ctype == 'cutoffs':
        pylab.scatter([index_and_interpolate(bad[group], val[group]) for group in val],
                      [index_and_interpolate(good[group], val[group]) for group in val],
                      **kws)
      elif ctype == 'y':
        pylab.scatter([find_x_for_y(bad[group], good[group], val) for group in good],
                      [val] * len(good.keys()), **kws)
      elif ctype == 'xy':
        pylab.scatter(val[0], val[1], **kws)
    pylab.legend(loc=0)
    pylab.ylabel(self.truepos_text)
    pylab.xlabel(self.falsepos_text)
    pylab.xlim([0, 1])
    pylab.ylim([0, 1])

  def plot_performance(self, cutoffs={}, raw=False, othercutoffs=None, otherthreshold=0, groups=None):
    """Draws the (score, non-default rate) curve.

    If raw = True, the x-axis has the raw score.  Otherwise, it's the
    within-group score percentile.

    Cutoffs can be a dictionary mapping group -> threshold.  If it
    exists, a region will be shaded.  Normally, this is the region
    below the curve to the right of the threshold.

    If othercutoffs is not None, the region is between cutoffs and
    othercutoffs in the x axis, and between the curve and
    otherthreshold in y axis.
    """
    if groups is None:
      groups = self.data.columns
    for group in groups:
      if raw:
        lines = pylab.plot(self.data.performance[group].index, 100*self.data.performance[group], label=group)
      else:
        lines = pylab.plot(100*self.data.cdfs[group], 100*self.data.performance[group], label=group)
      color = lines[0].get_color()
      if group in cutoffs:
        if othercutoffs:
          l, h = sorted([cutoffs[group], othercutoffs[group]])
        else:
          l, h = cutoffs[group], None
        y = self.data.performance[group].loc[l:h]
        if raw:
          x = y.index
        else:
          x = 100*self.data.cdfs[group].loc[l:h]
        pylab.fill_between(x, y*100, otherthreshold*100, alpha=0.2, color=color)
    if raw:
      idx = self.data.performance[group].index
      pylab.xlim((idx.min(), idx.max()))
      pylab.xlabel('%(longscore)s' % self.labels)
    else:
      if self.data.flipped:
        pylab.gca().invert_xaxis()
      pylab.xlabel('Within-group %(score)s percentile' % self.labels)
      #set_pct_formatter('x')
    pylab.ylabel(('%(success)s rate' % self.labels).capitalize())
    set_pct_formatter('y')
    pylab.legend(loc=0)

  def plot_pair(self, cutoffs, target, titles=[None,None], show_loss=False):
    """Plots performance with raw=True and raw=False in two subplots."""
    kws = {}
    if show_loss:
      pc = self.data.profit_cutoffs(target)
      kws.update(dict(othercutoffs=pc, otherthreshold=target))
    if isinstance(titles, str):
      titles = [titles + ' (raw score)', titles + ' (per-group)']
    pylab.clf()
    pylab.subplot(121)
    self.plot_performance(cutoffs=cutoffs, raw=True, **kws)
    pylab.hlines(target*100, self.data.cdfs.index.min(), self.data.cdfs.index.max(), linewidth=1, colors='k')
    pylab.title(titles[0], **titlekws)
    pylab.subplot(122)
    self.plot_performance(cutoffs=cutoffs, **kws)
    pylab.hlines(target*100, 0, 100, linewidth=1, colors='k')
    pylab.title(titles[1], **titlekws)
    pylab.tight_layout()

  def bar_plot(self, ds, labels=None, types=None, width=0.5):
    """Basic wrapper for constructing bar plots.

    ds: a list of dataframes, one for each type.
    labels: keys of individual dataframes, that correspond to different x positions.
    types: different dataframes, that correspond to different colors/legend.
    """
    if labels is None:
      labels = list(ds[0].keys())
    if types is None:
      types = [None]*len(ds)
    a = np.arange(len(labels))
    for i, d in enumerate(ds):
      pylab.bar(a+i*width, d, width,
                color='bgrcy'[i], label=types[i])
    pylab.xlim(-1./4,len(labels)-1 + width*len(ds)+1./4)
    pylab.xticks(a + (width * len(ds))/2., labels)

  def plot_opportunity(self, cutoffs_list, types=None, goal=None):
    """Plot a bar plot of the fraction non-defaulters getting loan in each group."""
    ops = [self.data.evaluate_opportunity(cutoffs) for cutoffs in cutoffs_list]
    for i, l in enumerate(types):
      if l.lower() == 'opportunity' and goal: #XXX HACK
        ops[i][:] = self.data.get_best_opportunity(goal)
    groups = self.columns
    self.bar_plot(ops, groups, types, 1./(len(ops) + 1))
    handles, labels = pylab.gca().get_legend_handles_labels()
    for a in (handles, labels):
      a[:] = a[::3] + a[1::3] + a[2::3]
    pylab.legend(handles, labels, loc='lower center', ncol=3, fontsize='x-small')
    pylab.ylim(0, 1)
    pylab.ylabel(self.truepos_text)


  def plot_boxes(self):
    """Plot a box-and-whisker plot of the distribution within each group."""
    pylab.boxplot([cdf_to_data(self.cdfs[group]) for group in self.columns], showfliers=False, labels=self.columns,
                  whis=[5,95])

  def plot_over_targets(self, do_coverage=True):
    """Plot efficiency as a function of target rate"""
    xlim = (0, 100)
    if len(self.data.cdfs.index) > 100:
      targets = np.concatenate([np.arange(0,0.9, 0.05), np.arange(0.9,1, 0.01)])
    else:
      xlim = (40, 80)
      targets = np.arange(0.4,0.81, 0.05)
    rate_funcs = [lambda t: self.data.profit_cutoffs(t),
                  lambda t: self.data.fixed_cutoffs(t, True),
                  lambda t: self.data.opportunity_cutoffs(t, True),
                  lambda t: self.data.two_sided_optimum(t),
                  lambda t: self.data.demographic_cutoffs(t, True),
                  ]
    df_base = pd.DataFrame(index=targets, columns=['Max profit', 'Single threshold', 'Opportunity', 'Equal odds', 'Demography'])
    cutoff_lists = [[f(t) for f in rate_funcs] for t in targets]

    efficiency_df = df_base.copy()
    for t, cutoff_list in zip(targets, cutoff_lists):
      efficiency_df.loc[t] = [self.data.efficiency(cutoff, t) for cutoff in cutoff_list]


    if do_coverage:
      pylab.clf()
      pylab.subplot(121)
    pylab.plot([], [])
    for key in efficiency_df.columns[1:]:
      pylab.plot(efficiency_df.index * 100, efficiency_df[key], label=key)
    pylab.legend(loc=0)
    pylab.xlabel('Minimal %(success)s rate for profitability' % self.labels)
    set_pct_formatter('x')
    if SHORT:
      pylab.ylabel('Profit as a fraction\nof max profit')
    else:
      pylab.ylabel('Profit as a fraction of max profit')
    pylab.title('Fraction of max profit earned as a\nfunction of minimal desired non-default rate', **titlekws)
    pylab.ylim((0, 1))
    pylab.xlim(xlim)

    if do_coverage:
      for t, cutoff_list in zip(targets, cutoff_lists):
        cutoffs = cutoff_list[1]
        coverage_df.loc[t] = [self.data.coverage({r:cutoffs[r]}) for r in self.columns]
      pylab.subplot(122)
      for key in coverage_df.columns:
        pylab.plot(coverage_df.index, coverage_df[key], label=key)
      pylab.legend(loc=0)
      pylab.xlabel('Target %(success)s rate' % self.labels)
      pylab.ylabel('Fraction of people getting %(classifier_outcome)s' % self.labels)
      pylab.title('Coverage of equal opportunity method', **titlekws)

  result_path = "C:/Users/Nitin_Bhati/fairness-Compas/flaskr/result/"
  def plot_comparison(self, target_rate, kwargs):
    """Call plot_curves for various cutoff methods."""
    xlim = kwargs.get('xlim')
    ylim = kwargs.get('ylim')
    s = kwargs.get('s',1)
    pylab.clf()
    # tar = str(target_rate)
    # print (tar)
    # result_set.writelines(target_rate)
    result_set.write(str('WOO'))
    result_set.write(str(target_rate)+"\n")
    print('WOO', target_rate)
    opt_point = self.data.two_sided_optimum(target_rate)
    pylab.subplot(121)

    demog_label = ('cutoffs', self.data.demographic_cutoffs(target_rate, True),
                       dict(marker='v',label='Demography', color='orange', s=70))
    self.plot_curves([
#      demog_label,
    ])
    pylab.legend(loc='lower right')
    pylab.title("Per-group ROC curve\nclassifying %(success_people)s using %(score)s" % self.labels, **titlekws)
    if xlim and ylim:
      pylab.hlines(ylim[0], xlim[0], xlim[1], linestyle='-', linewidth=1)
      pylab.vlines(xlim[1], ylim[0], ylim[1], linestyle='-', linewidth=1)
      pylab.hlines(ylim[1], xlim[0], xlim[1], linestyle='-', linewidth=1)
      pylab.vlines(xlim[0], ylim[0], ylim[1], linestyle='-', linewidth=1)
    pylab.plot([0,1],[0,1], lw=1,color='k', linestyle='-')
    pylab.subplot(122)
    # a = 10
    print('Demog:', self.data.demographic_cutoffs(target_rate, True))
    result_set.write(str(('Demog:', self.data.demographic_cutoffs(target_rate, True)))+"\n")
    print('Opportunity:', self.data.opportunity_cutoffs(target_rate, True))
    result_set.write(str(('Opportunity:', self.data.opportunity_cutoffs(target_rate, True)))+"\n")
    self.plot_curves([('cutoffs', self.data.profit_cutoffs(target_rate),
                        dict(marker='^',label='Max profit', s=70*s)),
                       ('cutoffs', self.data.fixed_cutoffs(target_rate, True),
                        dict(label='Single threshold', marker='o', color='g',s=100)),
#                      demog_label,
                      ('y', self.data.get_best_opportunity(target_rate),
                        dict(marker='x',label='Opportunity', s=130*s, color='purple')),
                      ('xy', ([opt_point[1]], [opt_point[0]]),
                       dict(marker='+', label='Equal odds', s=180*s, color='brown', zorder=4, linewidth=3)),
                     ], regular_labels=False)
    pylab.legend(loc='lower right', scatterpoints=1, fontsize='small')
    pylab.title("Zoomed in view", **titlekws)
    pylab.tight_layout()
    pylab.plot([0,1],[0,1], lw=1,color='k', linestyle='-')
    pylab.xlim(xlim)
    pylab.ylim(ylim)
    result_set.flush();
    # return target_rate
    #"Fraction %(fail_people)s/%(success_people)s getting %(classifier_outcome)s, by various methods" % self.labels)

  def plot_thresholds(self, target_rate, pcuts, ocuts, scuts, fcuts, srate):
    pylab.clf()
    ranges = self.data.two_sided_ranges(self.data.two_sided_optimum(target_rate))
    rows = [
      (5, 'Max profit', pcuts),
      (4, 'Single threshold', fcuts),
      (3, 'Opportunity', ocuts),
      #(1, 'Demography', (scuts, (srate)*100)),
      (1, 'Demography', (scuts, (1-srate)*100)),
    ]
    rows2 = [
      (2, 'Equal odds', ranges),
      ]
    for raw in range(2):
      if raw:
        pylab.subplot(121)
      else:
        pylab.subplot(122)
      pylab.yticks([r[0] for r in rows + rows2], [r[1] for r in rows + rows2])
      styles = [dict(marker='d', color='b'),
                dict(marker='o', color='g'),
                dict(marker='s', color='r'),
                dict(marker='p', color='c'),]
      for i, (group, style) in enumerate(zip(self.columns, styles)):
        this_style = dict(s=100)
        this_style.update(style)
        this_line_style = dict(ms=10, linestyle='-', lw=2)
        this_line_style.update(style)
        hoff = -0.2*(i - len(self.columns) / 2. + .5)
        def c_to_p2(c, group):
          if isinstance(c, tuple):
            if raw:
              c = c[0]
            else:
              return c[1]
          return c_to_p(c[group])
        def c_to_p(c):
          return c if raw else self.data.cdfs[group][c:].values[0]*100
        values = [c_to_p2(cuts, group) for h, label, cuts in rows]
        pylab.scatter(values, [r[0] + hoff for r in rows], label=group, **this_style)
        for h, label, cuts in rows2:
          hh = h + hoff
          pylab.plot(list(map(c_to_p, cuts[group])), [hh, hh], **this_line_style)

      loc = 0
      if raw:
        idx = self.data.performance[group].index
        x_range = (idx.min(), idx.max())
        if idx.min() == 300: #FICO, awful hack
          #x_range = (400, 700)
          loc = 'upper left'
        pylab.legend(loc=loc)
        pylab.xlim(x_range)
        pylab.xlabel('%(longscore)s' % self.labels)
        pylab.title("%(score)s thresholds (raw)" % self.labels)
      else:
        if self.data.performance[group].index.min() == 300: #FICO, awful hack
          loc = 'lower right'
        #pylab.legend(loc=loc)
        x_range = (0, 100)
        pylab.xlim(x_range)
        pylab.xlabel('Within-group %(longscore)s percentile' % self.labels)
        pylab.title("%(score)s thresholds (within-group)" % self.labels)
        #set_pct_formatter('x')
      pylab.hlines([i+0.5 for i in range(1, 5)], *x_range, linestyle=':', linewidth=1, color='k')
    pylab.tight_layout()
    #pylab.subplot(223)
    #perf = (self.data.performance * self.data.totals).sum(axis=1) / self.data.totals.sum()
    #pylab.plot(perf.index, perf, lw=2, color='k')

  def runfunc(self, k):
    for f in self.funcs.get(k, []):
      f()

  def addfunc(self, k, f):
    self.funcs.setdefault(k, []).append(f)

  marked_indices = None

  def plot_marginals(self):
    pylab.clf()
    pylab.subplot(121)
    self.plot_performance(raw=True)
    pylab.title(('%(success)s rate' % self.labels).capitalize() +  ' by %(score)s' % self.labels, **titlekws)
    leg_kws = dict(fontsize='small') if SHORT else {}
    pylab.legend(loc=0, **leg_kws)
    self.runfunc('marginal.performance')
    pylab.subplot(122)
    for group in self.columns:
      if self.data.flipped:
        pylab.plot(1-self.data.cdfs[group])
      else:
        pylab.plot(self.data.cdfs[group])
    pylab.title('CDF of %(score)s by group' % self.labels, **titlekws)
    pylab.xlabel('%(longscore)s' % self.labels)
    pylab.ylabel('Fraction of group below')
    pylab.legend(loc=0, **leg_kws)
    if self.marked_indices:
      for g, l in zip(self.columns, pylab.gca().lines):
        pylab.plot(self.marked_indices, 1-self.data.cdfs[g].loc[self.marked_indices] if self.data.flipped
                   else self.data.cdfs[g].loc[self.marked_indices], lw=0, marker='o', color=l.get_color())
    self.runfunc('marginal.cdf')
    pylab.tight_layout()

  def plot_figures(self, target_rate, file_prefix='output', to_plot=None, plot_kwargs={}, file_type='pdf'):
    """Plot all the figures.

    If file_prefix=None, then don't save figures, but display and wait for a button press.

    Otherwise, save to pdf files like `file_prefix`-profit.pdf .
    """
    all_to_plot = 'marginals thresholds pairs opportunity efficiency roc targets'.split()
    if to_plot is None:
      to_plot = all_to_plot

    p = str("\n"+'WOO')
    result_set.write(p)
    p1 = str(file_prefix)
    result_set.write(p1+"\n")
    print( file_prefix)
    if file_prefix is None:
      savefig = lambda t: pylab.show()#pylab.waitforbuttonpress()
      savefig = lambda t: pylab.waitforbuttonpress()
    else:
      savefig = lambda s: pylab.savefig(file_prefix+s+'.'+file_type)

    pcuts = self.data.profit_cutoffs(target_rate)
    orate = self.data.get_best_opportunity(target_rate)
    ocuts = self.data.opportunity_cutoffs(orate)
    srate = self.data.get_best_demographic(target_rate)
    scuts = self.data.demographic_cutoffs(srate)
    frate = self.data.get_best_fixed(target_rate)
    # print('Fixed rate:', frate)
    # result_set.write(str('Fixed rate:'))
    # result_set.write(str(frate)+"\n")
    fcuts = {r:frate for r in self.columns}
    two_opt = self.data.two_sided_optimum(target_rate)
    # print("Hello World")
    if SHORT:
      short_height = 3.5
      tall_height = 4.0
      taller_height = 4.5
    else:
      short_height = 7
      tall_height = 7
      taller_height = 7
    if 'marginals' in to_plot:
      pylab.figure(figsize=(15, short_height))
      self.plot_marginals()
      savefig('marginals')
      pylab.clf()

    if 'thresholds' in to_plot:
      pylab.figure(figsize=(15, taller_height))
      self.plot_thresholds(target_rate, pcuts, ocuts, scuts, fcuts, srate)
      savefig('thresholds')
      pylab.clf()

    if 'pairs' in to_plot:
      pylab.figure(figsize=(15, tall_height))
      self.plot_pair(pcuts, target_rate, 'Thresholds for maximal profit')
      savefig('profit')
      pylab.clf()
      self.plot_pair(ocuts, target_rate, 'Thresholds for equal opportunity')
      savefig('fair')
      pylab.clf()
      self.plot_pair(scuts, target_rate, 'Thresholds for demographic parity')
      savefig('demographic')
      pylab.clf()
      self.plot_pair(ocuts, target_rate, 'Thresholds for equal opportunity', True)
      savefig('fair-loss')
      pylab.clf()
      self.plot_pair(fcuts, target_rate, 'Single threshold', False)
      pylab.tight_layout()
      savefig('fixed')
      pylab.clf()

    cutoff_kinds = [pcuts, fcuts, ocuts, two_opt, scuts, ]
    cutoff_labels = ['Max profit', 'Single threshold', 'Opportunity', 'Equal odds', 'Demography', ]
    if 'opportunity' in to_plot:
      pylab.clf()
      self.plot_opportunity(cutoff_kinds, cutoff_labels, target_rate)
      pylab.title('Fraction %(success_people)s above threshold' % self.labels, **titlekws)
      savefig('opportunity')

    print('EFFICIENCIES:')
    result_set.write(str('EFFICIENCIES:'))
    for l, k in zip(cutoff_labels, cutoff_kinds):
      print(l+':', self.data.efficiency(k, target_rate))
      b2 = str((l+':', self.data.efficiency(k, target_rate)))
      result_set.write(b2+"\n")

    # print("hello plot_fig")
    if 'targets' in to_plot:
      pylab.figure(figsize=(15, tall_height))
      pylab.subplot(122)
      self.plot_over_targets(False)
      pylab.subplot(121)
      self.plot_opportunity(cutoff_kinds, cutoff_labels, target_rate)
      self.labels['target_rate'] = str(int(target_rate*100)) + '%'
      pylab.title(self.truepos_text, **titlekws)
      pylab.tight_layout()
      savefig('targets')

    if 'roc' in to_plot:
      pylab.figure(figsize=(15, tall_height))
      self.plot_comparison(target_rate, plot_kwargs.get('roc', {}))
      savefig('roc')
  result_set.flush()
  # result_set.close()
def return_resultset():
  return result_set