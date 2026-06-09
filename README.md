# Backpropagation to Reaction–Diffusion Systems
*(Gray–Scott as an example)*

---

## How to use this code repo

💡 Create and activate a virtual environment using Python 3.9, run
```
pip install -r requirements.txt
```
---

💡 Make sure your device (CPU/GPU) uses float64 instead of float32:
```
python check_float64.py
```

---

💡 To start exploring the parameter space of Gray Scott, run
```
python trainer.py
```
There is a target generation stage the first time you run it, but the targets will be stored in a `*.pt` file and reused when you repeat this command. Entering training, logs will show you at each iteration what gradients are generated, what learning rates are tried, which learning rate is finally used, how many Gray Scott discretized steps were taken, when there are overflows or NaN, and what the current parameters are.

---

💡 If you're not satisfied with just log information, copy the logged parameters into `pattern_gen_outside_training.py` and use the following to show stepping animations using such parameters (a list of multiple parameters from a parameter search trajectory can be used for simultaneous animation, so that you can see where the optimizer was going):
```
python pattern_gen_outside_training.py 4
```

---

💡 There are multiple ways to investigate the loss landscape cross-sections, and the following will print instructions:
```
python pattern_gen_outside_training.py 5
```
