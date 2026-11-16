---
title: How Learning Rate Shapes the Training Curve of a Small CNN
subtitle: One CNN, four learning rates, everything else held fixed
date: 2026-11-16
---

If you only get to tune one hyperparameter, most people will tell you to tune the learning rate. I wanted to see what that advice looks like in practice. So I took one small CNN, trained it four times on Fashion-MNIST, and changed nothing between the runs except the learning rate. **How does the learning rate change the training and validation curves of a small CNN, and where does it go from "too slow" to "broken"?**

Short answer: across a 1000× range, I got everything from a network that was still crawling after 20 epochs, to one that hit 80% validation accuracy in two epochs, to one that gave up completely and guessed at random.

## What the learning rate actually does

Training a network means adjusting its weights $w$ to make a loss $L(w)$ smaller. The loss measures how wrong the predictions are; here it is cross-entropy. The gradient $\nabla L(w)$ points in the direction where the loss grows fastest. So we step the other way:

$$
w_{t+1} = w_t - \eta \, \nabla L(w_t)
$$

That is plain stochastic gradient descent (SGD). "Stochastic" means the gradient is estimated from a small random batch of images instead of the whole dataset. The **learning rate** $\eta$ (eta) is the size of that step. The gradient tells you which way is downhill, and the learning rate decides how far you walk before checking again.

Why does the step size matter so much? Take the simplest possible loss, a parabola $L(w) = \tfrac{a}{2} w^2$, whose minimum is at $w = 0$. Its gradient is $a w$, so one SGD step gives

$$
w_{t+1} = w_t - \eta a w_t = (1 - \eta a)\, w_t .
$$

Each step multiplies the distance from the minimum by $(1 - \eta a)$, which gives three regimes:

- **$\eta$ small** ($\eta a$ close to 0): the factor is just under 1, so you move toward the minimum, but very slowly.
- **$\eta$ in the sweet spot** ($0 < \eta a < 2$): the factor's size is below 1, so you converge. Past $\eta a = 1$ you overshoot and bounce from side to side, but each bounce is smaller.
- **$\eta$ too big** ($\eta a > 2$): the factor's size is above 1, so every step lands *further* from the minimum than the last. Training diverges.

A real network's loss surface has millions of dimensions with different curvature in each, but the same three regimes still show up roughly. My experiment looks for them on a real model.

## Experiment setup

I kept everything fixed except $\eta$:

| Setting | Value |
|---|---|
| Dataset | Fashion-MNIST (28×28 grayscale clothing images, 10 classes) |
| Data used | 10,000 training and 5,000 validation images, both drawn from the official 60k training set. The 10k test set was held out. |
| Model | 2 conv layers (16 and 32 channels, 3×3, ReLU, max-pool) + dense layer of 64 + output layer. 105,866 parameters. |
| Optimizer | Plain SGD: no momentum, no weight decay, no learning-rate schedule |
| Learning rates compared | 0.001, 0.01, 0.1, 1.0 |
| Batch size / epochs / seed | 64 / 20 / 42 |
| Hardware | Laptop CPU, about 25–45 seconds per run |

I used a subset so each run takes under a minute on a CPU. The learning rates are spaced by factors of 10 because their effect works on orders of magnitude.

To make the comparison fair, every run starts from **the same initial weights** and sees **the same batches in the same order**. Only the step size differs:

```python
torch.manual_seed(SEED)                              # identical starting weights
model = SmallCNN()
opt = torch.optim.SGD(model.parameters(), lr=lr)     # the one thing that changes
order = torch.Generator().manual_seed(SEED)          # identical batch order
```

After each epoch I record training loss, validation loss and validation accuracy. I picked the best learning rate using **validation** accuracy only. I then evaluated that single model on the test set once, at the end. Full code: [lr_experiment.py on GitHub](https://github.com/kawlows/learning-rate-cnn/blob/main/lr_experiment.py).

## Results

:::{figure} images/lr-cnn/fig1_train_loss.png
:label: fig-lr-train-loss
:alt: Training loss per epoch for learning rates 0.001, 0.01, 0.1 and 1.0

Training loss per epoch for each learning rate. The dotted line is ln 10 ≈ 2.303, the loss of a model that assigns equal probability to all 10 classes.
:::

Figure 1 shows all three regimes from the parabola. With $\eta = 0.001$ the loss barely moves for five epochs and is still falling steadily at epoch 20 (0.744). With $\eta = 0.01$ and $0.1$ it drops quickly, and 0.1 ends lowest (0.149). With $\eta = 1.0$ the loss jumps to the chance line after the first epoch and stays there.

:::{figure} images/lr-cnn/fig2_val_accuracy.png
:label: fig-lr-val-acc
:alt: Validation accuracy per epoch for learning rates 0.001, 0.01, 0.1 and 1.0

Validation accuracy (%) per epoch for each learning rate.
:::

Figure 2 shows the same story on held-out data, plus something Figure 1 hides: the bigger the step, the jumpier the curve.

| Learning rate | Best validation accuracy | Epoch of best | First epoch with ≥ 80% val. accuracy | Training loss at epoch 20 |
|---|---|---|---|---|
| 0.001 | 72.5% | 20 | never | 0.744 |
| 0.01 | 85.1% | 17 | 10 | 0.387 |
| **0.1** | **88.6%** | 19 | **2** | 0.149 |
| 1.0 | 10.5% | 18 | never | 2.306 |

*Table 1. Summary per learning rate. Chance level on 10 balanced classes is 10%.*

Going from $\eta = 0.01$ to $\eta = 0.1$ raised the best validation accuracy from 85.1% to 88.6%. It also cut the time to reach 80% from 10 epochs to 2. Picked on validation, the $\eta = 0.1$ checkpoint from epoch 19 scored **88.1% on the test set**, close to its 88.6% validation score.

:::{figure} images/lr-cnn/fig3_train_vs_val.png
:label: fig-lr-train-vs-val
:alt: Training and validation loss per epoch, one panel per learning rate

Training (solid) and validation (dashed) loss for each learning rate. Note that each panel has its own y-axis.
:::

Figure 3 puts training and validation loss side by side. At $\eta = 0.001$ the two lines overlap, so the model is still underfitting. At $\eta = 0.1$ a gap opens up: by epoch 19 training loss is 0.162 and validation loss is 0.348, so the model is starting to fit details of the training set that don't generalize.

## Discussion

**Mostly what the theory predicted.** The small rate was stable but slow, the middle rates converged, and the largest broke training. As the parabola suggests, the best value I tried (0.1) sat just below one that failed (1.0).

**Divergence looked different than I expected.** I assumed $\eta = 1.0$ would make the loss blow up to infinity or `NaN`. Instead it went flat at 2.306, right at the chance line. To find out why, I ran one epoch at each rate and counted ReLU units that output zero for *every* validation image. At $\eta = 0.1$: none. At $\eta = 1.0$: 84% of the second conv layer's channels and 97% of the dense layer's units were dead. The output was identical for every image. A ReLU outputs $\max(0, x)$, and its gradient is zero when $x < 0$. The first few huge steps pushed most units into that negative zone, so they get no gradient and never recover. The network ends up predicting the same thing for every input, and its loss settles at ln 10. In a deep ReLU network, "too large" often means *dead* rather than *exploding*.

**Bigger steps, noisier curves.** In Figure 2 the $\eta = 0.1$ curve swings by about 6 points between neighbouring epochs in mid-training. For example, it drops from 85.5% at epoch 5 to 79.5% at epoch 6. Each SGD step uses only 64 images, so every step contains some noise. A larger learning rate amplifies that noise. Both $\eta = 0.01$ and $\eta = 0.1$ also drop sharply at epoch 20: 0.1 falls from 88.6% to 75.5%. The two runs saw exactly the same batches, so I suspect the last few batches of epoch 20 pulled both models the same wrong way. I haven't tested that. The practical lesson is that choosing by *final-epoch* accuracy would have made 0.1 and 0.01 look tied at about 75.5%. Keeping the best validation checkpoint avoids that trap.

**Limitations.**
- **One seed per setting.** Given how noisy the curves are, differences of a point or two between runs are not reliable. Repeating with 3–5 seeds and plotting the mean would fix this.
- **Only 20 epochs on a 10k subset.** The $\eta = 0.001$ run was clearly not finished. With enough epochs it would likely close part of the gap, so "worse" here partly means "slower".
- **A coarse grid.** I only tried powers of 10. The real best value could be anywhere between 0.1 and 1.0.
- **Plain SGD only.** Momentum, Adam or a learning-rate schedule (shrinking $\eta$ over time) would all change where the sweet spot is. A schedule would probably also smooth out the late-epoch swings.

## Takeaways

1. **Search the learning rate on a log scale first.** With nothing else changed, moving from 0.001 to 0.1 took best validation accuracy from 72.5% to 88.6%. Try 0.001, 0.01, 0.1, 1.0, then refine around the best one.
2. **Look at the curve, not just the final number.** A flat loss stuck at ln(number of classes) means the network has stopped learning, often from dead ReLUs. A curve that keeps falling slowly means your rate is probably too small.
3. **With a large learning rate, keep the best checkpoint.** Validation accuracy can swing by several points from one epoch to the next. Selecting on the best validation epoch, not the last one, gives a fairer comparison.

## References

- Goodfellow, Bengio & Courville, *Deep Learning*, Chapter 8, "Optimization for Training Deep Models": <https://www.deeplearningbook.org/contents/optimization.html>
- Y. Bengio (2012), "Practical Recommendations for Gradient-Based Training of Deep Architectures": <https://arxiv.org/abs/1206.5533>
- L. N. Smith (2017), "Cyclical Learning Rates for Training Neural Networks" (learning-rate range test): <https://arxiv.org/abs/1506.01186>
- Stanford CS231n notes, "Neural Networks Part 3: Learning and Evaluation": <https://cs231n.github.io/neural-networks-3/>
- H. Xiao, K. Rasul & R. Vollgraf (2017), "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms": <https://arxiv.org/abs/1708.07747>
- PyTorch documentation, `torch.optim.SGD`: <https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html>
- Code for this post: <https://github.com/kawlows/learning-rate-cnn>
