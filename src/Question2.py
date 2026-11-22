# we have a ring road with 40 sensors. The slowdown moves one sensor backward per step.

import numpy as np
import matplotlib.pyplot as plt

N = 40
STEPS = 20

# true motion rule equation given: x_i(t+1) = x_{i+1}(t)
def true_step(x):
    return np.roll(x, -1)


def symmetric_layer(x, a, b):
    # layer from part (a): both neighbours share the weight b
    left = np.roll(x, 1)      # left[i]  = x[i-1]
    right = np.roll(x, -1)    # right[i] = x[i+1]
    return a * x + b * (left + right)


# Q2(a)(i): fit a and b on one spike

x_now = np.zeros(N)
x_now[20] = 1.0
x_next = true_step(x_now)     # the spike is now at sensor 19

# model is linear in a and b, so I solve it with least squares.
# column 1 multiplies a. column 2 multiplies b.
col_a = x_now
col_b = np.roll(x_now, 1) + np.roll(x_now, -1)
A = np.stack([col_a, col_b], axis=1)
solution, _, _, _ = np.linalg.lstsq(A, x_next, rcond=None)
a_fit = solution[0]
b_fit = solution[1]

pred = symmetric_layer(x_now, a_fit, b_fit)
loss = np.sum((pred - x_next) ** 2)

print("Q2(a)(i)")
print("  fitted a =", round(a_fit, 4))
print("  fitted b =", round(b_fit, 4))
print("  squared error =", round(loss, 4))
print("  prediction at sensors 19, 20, 21 =", pred[19], pred[20], pred[21])
print("  truth at sensors 19, 20, 21      =", x_next[19], x_next[20], x_next[21])


# Q2(a)(ii): roll out to 20 steps on a smooth peak

i = np.arange(N)
bump = np.exp(-((i - 20) ** 2) / 8)

predicted = np.zeros((STEPS + 1, N))
truth = np.zeros((STEPS + 1, N))
predicted[0] = bump
truth[0] = bump

for t in range(STEPS):
    predicted[t + 1] = symmetric_layer(predicted[t], a_fit, b_fit)
    truth[t + 1] = true_step(truth[t])

print("Q2(a)(ii)")
print("  true bump:      peak sensor at t=20 =", np.argmax(truth[20]),
      ", peak value =", round(truth[20].max(), 3))
print("  predicted bump: peak sensor at t=20 =", np.argmax(predicted[20]),
      ", peak value =", round(predicted[20].max(), 3))

# (vmin and vmax are shared).
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
titles = ["Predicted by the symmetric layer", "True motion"]
data = [predicted, truth]

for ax, title, d in zip(axes, titles, data):
    im = ax.imshow(d, aspect="auto", origin="upper", cmap="Blues",
                   vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Sensor index")
    ax.set_xticks(np.arange(0, N, 5))

axes[0].set_ylabel("Time step t")
axes[0].set_yticks(np.arange(0, STEPS + 1, 5))
fig.colorbar(im, ax=axes, label="Slowdown")
plt.savefig("q2_heatmaps.png", dpi=150, bbox_inches="tight")
print("  saved q2_heatmaps.png")


# Q2(b): try with separate weights for the two directions

a, w_minus, w_plus = 0.0, 0.0, 1.0
rng = np.random.default_rng(0)
x = rng.normal(size=N)        # any input can work, so I tested a random one

pred_b = a * x + w_minus * np.roll(x, 1) + w_plus * np.roll(x, -1)
print("Q2(b)")
print("  exact on a random snapshot:", np.allclose(pred_b, true_step(x)))

# same thing as a 1d convolution with circular padding.
kernel = np.array([0.0, 0.0, 1.0])            # multiplies [x_{i-1}, x_i, x_{i+1}]
padded = np.concatenate([x[-1:], x, x[:1]])   # loop on the ring x_39 on the left, x_0 on the right
conv_out = np.zeros(N)
for k in range(N):
    window = padded[k:k + 3]                  # [x_{k-1}, x_k, x_{k+1}]
    conv_out[k] = np.sum(kernel * window)
print("  convolution matches the true step:", np.allclose(conv_out, true_step(x)))


# Q2(c): equal neighbour weights plus history

a, b, c = 0.0, 1.0, -1.0

# (i) backward motion: x_i(t+1) = x_{i+1}(t)
x_prev = rng.normal(size=N)
x_curr = np.roll(x_prev, -1)
x_true = np.roll(x_curr, -1)
pred_c = a * x_curr + b * (np.roll(x_curr, 1) + np.roll(x_curr, -1)) + c * x_prev
print("Q2(c)")
print("  backward motion, exact:", np.allclose(pred_c, x_true))

# (ii) forward motion: x_i(t+1) = x_{i-1}(t) as same coefficients
x_prev = rng.normal(size=N)
x_curr = np.roll(x_prev, 1)
x_true = np.roll(x_curr, 1)
pred_c = a * x_curr + b * (np.roll(x_curr, 1) + np.roll(x_curr, -1)) + c * x_prev
print("  forward motion, exact: ", np.allclose(pred_c, x_true))