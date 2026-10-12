"""One compact figure from reviewed exact FTA reports."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def plot(summary_path, output):
    summary = json.loads(Path(summary_path).read_text())
    output = Path(output)
    assert not output.exists(), 'Protect previous figures'
    metrics = ('false_completion_claim', 'fabricated_requested_facts',
               'honest_failure_disclosure', 'contradictory_headers')
    labels = ('Unsupported\ncompletion', 'Fabricated\nrequested facts',
              'Honest failure\ndisclosure', 'Contradictory\ncompletion header')
    policies = ('baseline','transparency','evidence_contract')
    names = ('Baseline', 'Transparency', 'Evidence contract')
    colors = ('#385c86', '#26a18a', '#dc934a')
    models = ('Qwen/Qwen3-8B', 'google/gemma-2-9b-it')
    table = {tuple(g['group']):g for g in summary['summary_counts'] if g['dimension']=='model_policy'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#cbd5df'})
    fig, axes = plt.subplots(1, 2, figsize=(10.6,4.3), sharey=True)
    fig.patch.set_facecolor('white')
    for ax, model in zip(axes,models):
        for j,(policy,name,color) in enumerate(zip(policies,names,colors)):
            group = table[(model,policy)]
            rates = [group['labels'][k]['rate'] for k in metrics]
            assert all(r is not None for r in rates)
            positions = [x + (j-1)*.23 for x in range(4)]
            ax.bar(positions,rates,width=.21,color=color,label=name,zorder=3)
            for x, rate in zip(positions,rates):
                ax.text(x,max(rate,.005)+.018,f'{rate:.0%}',ha='center',va='bottom',fontsize=8,
                        color='#293746',rotation=0)
        ax.set_title(model.split('/')[-1],fontweight='bold',fontsize=12,pad=16)
        ax.set_xticks(range(4),labels,fontsize=9)
        ax.tick_params(axis='both',length=0)
        ax.set_ylim(0,1.14)
        ax.set_yticks([0,.25,.5,.75,1])
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.grid(axis='y',color='#e6ebf0',linewidth=.7,zorder=0)
        ns = ' / '.join(str(table[(model,p)]['valid_n']) for p in policies)
        ax.text(.5,-.22,f'Full responses: {ns} (policy order above)',transform=ax.transAxes,
                ha='center',fontsize=8,color='#586779')
    axes[0].set_ylabel('Share of valid full answers',fontsize=10)
    handles, legendlabels = axes[0].get_legend_handles_labels()
    fig.legend(handles,legendlabels,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,.885),fontsize=10)
    fig.suptitle('Original FTA tasks: what the models reported',x=.5,y=.985,fontweight='bold',fontsize=15)
    v=summary['verification']
    overall=next(g for g in summary['summary_counts'] if g['dimension']=='all')
    unresolved=sum(overall['labels'][k]['unresolved_valid_n'] for k in metrics)
    foot = (f"1,200 reports · {v['truncated']} length-truncated outputs excluded · "
            f"{v['duplicate_repetition_pairs']}/600 repetition pairs identical · {unresolved} displayed flags unresolved\n"
            'Flags overlap. Rates exclude unresolved labels; denominators can differ. Blinded Codex-assisted scoring, not human annotation.')
    fig.text(.5,.018,foot,ha='center',va='bottom',fontsize=8,color='#586779',linespacing=1.6)
    fig.subplots_adjust(left=.075,right=.985,top=.73,bottom=.29,wspace=.14)
    fig.savefig(output,dpi=190,facecolor='white',metadata={'Description':'Exact original FTA task replay; independent overlapping report flags'})
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('summary');p.add_argument('output')
    a=p.parse_args();plot(a.summary,a.output)
