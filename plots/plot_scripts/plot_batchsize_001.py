import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sns.set()
sns.set_context("poster")
sns.set_style("white")

data = np.load("../outputs/ch4cn_batchsize_001.npz")
batch_sizes, scores, wall_times, grad_updates = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"]

loss_data = []

for bs in batch_sizes:
    event_acc = EventAccumulator('../outputs/ch4cn_batchsize_001/tb_'+str(bs)+'/training')
    event_acc.Reload()
    loss_data.append(event_acc)

fig, ax = plt.subplots(figsize=(8,6))

for i, bs in enumerate(batch_sizes):
    w_times_t, step_nums_t, vals_training = zip(*loss_data[i].Scalars('cost'))
    ax.plot(step_nums_t, vals_training, label="Batchsize %i"%bs)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Cost")
ax.legend()

plt.savefig("batchsize_cost.png", dpi=200)

fug, ux = plt.subplots(figsize=(8, 6))

for i, bs in enumerate(batch_sizes):
    w_times_t, step_nums_t, vals_training = zip(*loss_data[i].Scalars('cost'))
    w_times_t = np.asarray(w_times_t)
    w_times_t -= w_times_t[0]
    ux.plot(w_times_t, vals_training, label="Batchsize %i" % bs)
    ux.set_xlabel("Time (s)")
    ux.set_ylabel("Cost")
    ux.set_xlim((-50,2500))

ux.legend()
plt.savefig("batchsize_times.png", dpi=200)

