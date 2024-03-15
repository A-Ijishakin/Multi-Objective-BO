import torch 
import numpy as np 
from methods import load_pickles, problem, N_ITER, BATCH_SIZE, ci, to_cpu 
import matplotlib.pyplot as plt 
from matplotlib.cm import ScalarMappable

qparego, qehvi, qnehvi, random, dragonfly = load_pickles(files=['qparego.pkl', 'qehvi.pkl', 
                                                         'qnehvi.pkl', 'random.pkl', 'dragonfly.pkl']) 

hvs_qparego, hvs_qehvi, hvs_qnehvi, hvs_random, hvs_dragonfly = qparego['hvs'], qehvi['hvs'], qnehvi['hvs'], random['hvs'], dragonfly['hvs']
train_obj_true_random, train_obj_true_qparego = qparego['train_obj_true'], qparego['train_obj_true']
train_obj_true_qehvi, train_obj_true_qnehvi = qehvi['train_obj_true'], qnehvi['train_obj_true'] 
train_obj_true_dragonfly = dragonfly['train_obj_true'] 


iters = np.arange(N_ITER + 1) * BATCH_SIZE


log_hv_difference_qparego = np.log10(problem.max_hv - np.asarray(hvs_qparego))
log_hv_difference_qehvi = np.log10(problem.max_hv - np.asarray(hvs_qehvi))
log_hv_difference_qnehvi = np.log10(problem.max_hv - np.asarray(hvs_qnehvi))
log_hv_difference_rnd = np.log10(problem.max_hv - np.asarray(hvs_random)) 
log_hv_difference_dragonfly = np.log10(problem.max_hv - np.asarray(hvs_dragonfly)) 

fig, ax = plt.subplots(1, 1, figsize=(8, 6)) 
ax.errorbar(
    iters,
    log_hv_difference_rnd.mean(axis=0),
    yerr = ci(log_hv_difference_rnd), 
    label="Sobol",
    linewidth=1.5,
) 

ax.errorbar(
    iters,
    log_hv_difference_qparego.mean(axis=0),
    yerr = ci(log_hv_difference_qparego), 
    label="qNParEGO",
    linewidth=1.5,
)
ax.errorbar(
    iters,
    log_hv_difference_qehvi.mean(axis=0),
    yerr = ci(log_hv_difference_qehvi), 
    label="qEHVI",
    linewidth=1.5,
)
ax.errorbar(
    iters, 
    log_hv_difference_qnehvi.mean(axis=0),
    yerr=ci(log_hv_difference_qnehvi), 
    label="qNEHVI",
    linewidth=1.5,
) 
ax.errorbar(
    iters,
    log_hv_difference_dragonfly.mean(axis=0),
    yerr = ci(log_hv_difference_dragonfly), 
    label="Dragonfly",
    linewidth=1.5,
) 
ax.set(
    xlabel="number of observations (beyond initial points)",
    ylabel="Log Hypervolume Difference",
)
ax.legend(loc="lower left") 

plt.savefig('MOO/images/hv_difference.png') 

fig, axes = plt.subplots(1, 5, figsize=(23, 7), sharex=True, sharey=True)
algos = ["Sobol", "qNParEGO", "qEHVI", "qNEHVI", 'Dragonfly']
cm = plt.cm.get_cmap("viridis")

batch_number = torch.cat(
    [
        torch.zeros(1 * (problem.dim + 2)),
        torch.arange(1, N_ITER + 1).repeat(BATCH_SIZE, 1).t().reshape(-1),
    ]
).numpy()
for i, train_obj in enumerate(
    (
        to_cpu(torch.stack(train_obj_true_random)).mean(axis=0),
        to_cpu(torch.stack(train_obj_true_qparego)).mean(axis=0),
        to_cpu(torch.stack(train_obj_true_qehvi)).mean(axis=0),
        to_cpu(torch.stack(train_obj_true_qnehvi)).mean(axis=0),
        to_cpu(torch.stack(train_obj_true_dragonfly)).mean(axis=0) 
    )
):
    sc = axes[i].scatter(
        train_obj[:, 0],
        train_obj[:, 1],
        c=batch_number,
        alpha=0.8,
    )
    axes[i].set_title(algos[i])
    axes[i].set_xlabel("Objective 1")
axes[0].set_ylabel("Objective 2")
norm = plt.Normalize(batch_number.min(), batch_number.max())
sm = ScalarMappable(norm=norm, cmap=cm)
sm.set_array([])
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax) 
plt.savefig('MOO/images/Objectives.png') 


breakpoint 