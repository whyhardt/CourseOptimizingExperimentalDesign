# AutoRA & Active Learning for Experimental Design

**Course: Optimizing Experimental Design in Cognitive Science**
**Session 5-8: Automated Research & Active Learning**

---

## Slide 1: Title Slide

# AutoRA & Active Learning
## Automated Closed-Loop Experimentation

**Topics:**
- AutoRA Framework
- Closed-Loop Experimentation
- Active Learning Strategies
- Uncertainty & Disagreement Sampling

---

## Slide 2: The Problem with Traditional Experiments

### Traditional Approach
1. **Design** all conditions upfront (factorial, random)
2. **Run** all trials
3. **Analyze** data
4. **(Maybe)** design follow-up study

### Limitations
- **Inefficient**: Many trials provide little information
- **Fixed**: Cannot adapt based on early results
- **Resource-intensive**: Wastes time, participants, money

### Example: 2AFC Experiment
- 100 participants × 100 conditions = **10,000 trials**
- But maybe only **500 informative trials** needed!

---

## Slide 3: Closed-Loop Experimentation

### The Better Way: Adaptive Experimentation

```
┌─────────────────────────────────────────┐
│  1. Start with small set of conditions  │
└──────────────┬──────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  2. Run trials → Fit model               │
│     ↓                                    │
│     Identify INFORMATIVE conditions      │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  3. Repeat until budget exhausted        │
└──────────────────────────────────────────┘
```

### Benefits
✓ **Efficiency**: Learn faster with fewer samples
✓ **Adaptation**: Focus on informative regions
✓ **Automation**: Minimal human intervention

---

## Slide 4: Introducing AutoRA

### AutoRA = Automated Research Assistant

**Mission**: Enable fully automated closed-loop empirical research

**Key Features:**
- 🔄 Closed-loop workflow
- 🧪 Experiment management
- 🤖 Active learning strategies
- 📊 Model fitting & evaluation
- 🔌 Integration with online platforms (Prolific, Firebase)

**Developed by:** Musslick et al. (2024)
**Documentation:** [autoresearch.github.io/autora](https://autoresearch.github.io/autora/)

---

## Slide 5: AutoRA Workflow - The Loop

```
         ┌─────────────────┐
         │  STATE          │
         │  • Variables    │
         │  • Conditions   │
         │  • Data         │
         │  • Models       │
         └────────┬────────┘
                  ↓
         ┌─────────────────┐
         │ EXPERIMENTALIST │ ← Proposes new conditions
         └────────┬────────┘
                  ↓
         ┌─────────────────┐
         │ EXPERIMENT      │ ← Runs experiments,
         │ RUNNER          │   collects data
         └────────┬────────┘
                  ↓
         ┌─────────────────┐
         │ THEORIST        │ ← Builds/updates model
         └────────┬────────┘
                  ↓
         (Loop back to State)
```

**Central Concept**: All components communicate via **State** object

---

## Slide 6: The State Object

### What is a State?

The **State** is the central data structure containing:

```python
StandardState(
    variables = VariableCollection(...),
    conditions = DataFrame([...]),
    experiment_data = DataFrame([...]),
    models = [model1, model2, ...]
)
```

### Components

| Field | Description | Example |
|-------|-------------|---------|
| `variables` | IVs and DVs | participant_id, ratio, scatteredness → response_time |
| `conditions` | Proposed trials | [(0, 0.5, 0.3), (1, 0.8, 0.7), ...] |
| `experiment_data` | Collected observations | conditions + response_time measurements |
| `models` | Current model(s) | [FFNRegressor, GPRegressor, ...] |

---

## Slide 7: Experimentalists - The Brain of the Loop

### What is an Experimentalist?

**Function**: Proposes which conditions to test next

**Input**: Current state (data, model, variables)
**Output**: New conditions to test

### Available Strategies in AutoRA

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Grid** | Evenly-spaced factorial design | Full coverage needed |
| **Random** | Uniform random sampling | Baseline, exploration |
| **Uncertainty** | Sample where model is uncertain | Smooth functions, GP models |
| **Disagreement** | Sample where models disagree | Complex functions, ensembles |
| **Novelty** | Sample far from previous points | Diverse coverage |
| **Leverage** | Sample for maximum model improvement | Fast convergence |

---

## Slide 8: Simple Experimentalists

### Grid Sampling

```python
from autora.experimentalist.grid import grid_pool

conditions = grid_pool(
    variables,
    num_samples=5,  # 5 levels per variable
    sample_all=['participant_id']
)
```

- Creates evenly-spaced grid (factorial design)
- **Pros**: Full coverage, interpretable
- **Cons**: Combinatorial explosion, inefficient

### Random Sampling

```python
from autora.experimentalist.random import random_sample

conditions = random_sample(
    variables,
    num_samples=100,
    random_state=42
)
```

- Uniformly random samples
- **Pros**: Simple, no assumptions
- **Cons**: Inefficient, wastes samples

---

## Slide 9: Active Learning - The Smart Approach

### Core Principle

> **Not all samples are equally informative!**

### Information Theory Connection

**Entropy** (uncertainty):
$$H(Y) = -\\sum_y p(y) \\log p(y)$$

**Mutual Information** (information gain):
$$I(X; Y) = H(Y) - H(Y|X)$$

**Goal of Active Learning**:
Select samples $X$ that **maximize** $I(X; Y)$

### Two Main Approaches

1. **Uncertainty Sampling**: Query where single model is uncertain
2. **Disagreement Sampling**: Query where multiple models disagree

---

## Slide 10: Uncertainty Sampling

### Concept

Sample where the model is **most uncertain** about its prediction

$$x^* = \\arg\\max_x \\sigma(x)$$

where $\\sigma(x)$ = prediction standard deviation

### Implementation

Requires model with uncertainty estimates (e.g., **Gaussian Process**)

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from autora.experimentalist.uncertainty import uncertainty_sample

# GP provides mean μ(x) and std σ(x)
gp = GaussianProcessRegressor(...)
gp.fit(X, y)

# Sample where σ(x) is largest
conditions = uncertainty_sample(
    pool,  # Candidate points
    model=gp,
    num_samples=10
)
```

---

## Slide 11: Gaussian Processes - Uncertainty Estimation

### Why Gaussian Processes?

**Advantages:**
- Provides **natural uncertainty** estimates: $\\mu(x) \\pm \\sigma(x)$
- Uncertainty **low** near data, **high** far from data
- Well-calibrated for smooth functions

### Visualization

```
         High Uncertainty
              ↓
    ┌─────────────────┐
    │   ░░░░░░░░░░░   │  ← Wide confidence band
    │   ─────────────  │  ← Mean prediction
    │   ░░░░░░░░░░░   │
    └─────────────────┘
            ↑
      No training data

         Low Uncertainty
              ↓
    ┌─────────────────┐
    │      ░░░         │  ← Narrow confidence band
    │   ●──────●      │  ← Mean prediction
    │      ░░░         │     ● = training data
    └─────────────────┘
```

---

## Slide 12: Disagreement Sampling (Query-by-Committee)

### Concept

Train **ensemble** of models → Sample where they **disagree** most

$$x^* = \\arg\\max_x \\text{Var}(\\{f_1(x), f_2(x), ..., f_K(x)\\})$$

### Why Disagreement?

- **Epistemic uncertainty**: "We don't know" (resolvable with data)
- **Aleatoric uncertainty**: "Inherent noise" (irreducible)

Disagreement captures epistemic uncertainty!

### Implementation

```python
from autora.experimentalist.inequality import inequality_sample

# Ensemble of neural networks
ensemble = FFNEnsemble(n_models=5, ...)
ensemble.fit(X, y)

# Sample where models disagree most
conditions = inequality_sample(
    pool,
    model=ensemble,
    num_samples=10
)
```

---

## Slide 13: Building an Ensemble

### FFN Ensemble Implementation

```python
class FFNEnsemble:
    def __init__(self, n_models=5, ...):
        self.models = []
        for i in range(n_models):
            model = FFNRegressor(...)
            self.models.append(model)

    def predict(self, X, return_std=False):
        predictions = [m.predict(X) for m in self.models]
        mean = np.mean(predictions, axis=0)

        if return_std:
            std = np.std(predictions, axis=0)  # Disagreement!
            return mean, std
        return mean
```

### Key Points
- Each model trained with **different random initialization**
- Diversity → Different learned functions
- Variance = Disagreement = Uncertainty

---

## Slide 14: Comparison - Random vs. Uncertainty vs. Disagreement

### Typical Results (10 Cycles, 2AFC Experiment)

```
MSE
 │
 │  ●────●────●────●────●  Random
 │   ╲
 │    ■──■──■──■──■       Uncertainty (GP)
 │     ╲
 │      ▲─▲─▲─▲─▲         Disagreement (Ensemble)
 └────────────────────────► Cycle
```

### Performance Gains

| Strategy | Final MSE | Improvement vs. Random |
|----------|-----------|------------------------|
| Random | 0.0450 | — |
| Uncertainty | 0.0180 | **60% better** |
| Disagreement | 0.0160 | **64% better** |

---

## Slide 15: When to Use Each Strategy?

### Decision Guide

| Criterion | Random | Uncertainty (GP) | Disagreement (Ensemble) |
|-----------|--------|------------------|-------------------------|
| **Function smoothness** | Any | Smooth | Non-linear, complex |
| **Sample budget** | Very small | Medium (100-1000) | Large (1000+) |
| **Computational cost** | ✓ Low | ✓ Medium | ✗ High |
| **Model assumptions** | None | Kernel structure | None |
| **Uncertainty calibration** | — | ✓ Good | ✓ Robust |
| **Exploration** | ✓ Good | Medium | Medium |

### Recommendations

1. **Start simple**: Grid or Random for baseline
2. **Add intelligence**: Uncertainty (GP) for efficiency
3. **Go advanced**: Disagreement (Ensemble) for complex problems

---

## Slide 16: Practical Example - 2AFC Experiment

### Setup
- **Participants**: 100
- **IVs**: ratio (0-1), scatteredness (0-1)
- **DV**: response_time
- **Budget**: 500 trials (instead of 10,000!)

### Closed-Loop with AutoRA

```python
from autora.state import StandardState, on_state

# Initialize
state = StandardState(
    variables=experiment.variables,
    conditions=initial_conditions,
    experiment_data=pd.DataFrame(),
    models=[model]
)

# Loop
for cycle in range(10):
    state = experimentalist(state, num_samples=5)
    state = experiment_runner(state)
    state = theorist(state)
```

### Result
- **500 trials** with uncertainty sampling ≈ **5000 trials** random!
- **10× efficiency gain**

---

## Slide 17: Advanced Topics & Extensions

### Hybrid Strategies
Combine multiple approaches:
```python
# Early: Explore with random
if cycle < 3:
    conditions = random_sample(...)
# Mid: Exploit with uncertainty
elif cycle < 7:
    conditions = uncertainty_sample(...)
# Late: Refine with disagreement
else:
    conditions = inequality_sample(...)
```

### Other AutoRA Experimentalists

- **Novelty**: Sample far from previous points
- **Leverage**: Maximize influence on model
- **Falsification**: Test model predictions
- **Mixture**: Weighted combination of strategies

### Integration
- **Online recruitment**: Prolific integration
- **Cloud databases**: Firebase support
- **Real experiments**: Beyond synthetic!

---

## Slide 18: AutoRA Installation & Resources

### Installation

```bash
# Basic AutoRA
pip install autora

# With experimentalists
pip install -U "autora[experimentalist-uncertainty]"
pip install -U "autora[experimentalist-inequality]"

# Or install all at once
pip install -U "autora[all-experimentalists]"
```

### Resources

- **Documentation**: [autoresearch.github.io/autora](https://autoresearch.github.io/autora/)
- **Paper**: Musslick et al. (2024). AutoRA: Automated Research Assistant
- **GitHub**: [github.com/AutoResearch/autora](https://github.com/AutoResearch/autora)
- **Tutorials**: Course repository - `tutorials/autora_*.ipynb`

---

## Slide 19: Summary & Key Takeaways

### What You Learned

1. ✅ **Closed-Loop Experimentation**: Adapt experiments based on results
2. ✅ **AutoRA Framework**: State → Experimentalist → Runner → Theorist
3. ✅ **Active Learning**: Not all samples are equal!
4. ✅ **Uncertainty Sampling**: Query where model is uncertain (GP)
5. ✅ **Disagreement Sampling**: Query where models disagree (Ensemble)
6. ✅ **Efficiency Gains**: 5-10× fewer samples for same accuracy

### The Big Picture

> **Traditional**: Design → Run → Analyze
> **AutoRA**: Design → Run → Analyze → **Adapt** → Repeat

**Result**: Faster science, better use of resources, automated discovery!

---

## Slide 20: Group Project - Apply What You've Learned!

### Your Task

1. **Choose** an active learning strategy (or design your own!)
2. **Implement** using AutoRA framework
3. **Test** on 2AFC experiment with varying noise levels
4. **Compare** to baseline (random sampling)
5. **Present** findings and insights

### Evaluation Criteria

- Implementation correctness
- Comparison thoroughness
- Analysis depth
- Code documentation
- Creativity (bonus!)

### Resources

- **Tutorials**: `autora_intro.ipynb`, `autora_uncertainty.ipynb`, `autora_advanced.ipynb`
- **Literature**: Papers with GitHub repos provided
- **Office Hours**: Ask questions!

**Good luck! 🚀**

---

## Slide 21: Questions & Discussion

### Discussion Points

1. When might active learning **fail**?
2. How to balance **exploration** vs. **exploitation**?
3. Ethical considerations for **automated** experiments?
4. Applications beyond **cognitive science**?

### Further Reading

- Settles, B. (2009). Active Learning Literature Survey
- Musslick et al. (2023). Evaluation of Sampling Strategies for AutoRA
- Rasmussen & Williams (2006). Gaussian Processes for Machine Learning

---

## Backup Slides

### Slide 22: Mathematics of Uncertainty Sampling

**Gaussian Process Posterior:**

Given data $\\mathcal{D} = \\{(x_i, y_i)\\}_{i=1}^n$:

$$
\\begin{aligned}
p(f(x) | \\mathcal{D}) &= \\mathcal{N}(\\mu(x), \\sigma^2(x)) \\\\
\\mu(x) &= k(x)^T (K + \\sigma_n^2 I)^{-1} y \\\\
\\sigma^2(x) &= k(x,x) - k(x)^T (K + \\sigma_n^2 I)^{-1} k(x)
\\end{aligned}
$$

**Uncertainty Sampling:**
$$x^* = \\arg\\max_x \\sigma(x)$$

**Expected Information Gain:**
$$IG(x) \\approx H[y|x] - \\mathbb{E}_{y|x}[H[\\theta | \\mathcal{D} \\cup \\{(x,y)\\}]]$$

---

### Slide 23: Code Example - Complete Workflow

```python
# 1. Setup
from autora.state import StandardState, on_state, estimator_on_state
from autora.experimentalist.uncertainty import uncertainty_sample
from sklearn.gaussian_process import GaussianProcessRegressor

# 2. Initialize
gp = GaussianProcessRegressor(...)
state = StandardState(
    variables=experiment.variables,
    conditions=initial_conditions,
    experiment_data=pd.DataFrame(),
    models=[gp]
)

# 3. Wrap components
experimentalist = on_state(uncertainty_sample, output=['conditions'])
runner = on_state(experiment.run, output=['experiment_data'])
theorist = estimator_on_state(gp)

# 4. Run loop
for cycle in range(10):
    # Generate pool
    pool_state = pool_generator(state, num_samples=50)
    # Select from pool
    pool_state = experimentalist(pool_state, num_samples=5)
    state.conditions = pool_state.conditions
    # Run & fit
    state = runner(state)
    state = theorist(state)
```

---

### Slide 24: Alternative Active Learning Strategies

**Expected Model Change:**
- Select sample that changes model parameters most
- $x^* = \\arg\\max_x |\\theta_{new} - \\theta_{old}|$

**Expected Error Reduction:**
- Select sample that reduces future error most
- $x^* = \\arg\\min_x \\mathbb{E}[\\text{Error}_{future}]$

**Density-Weighted Uncertainty:**
- Balance uncertainty with representativeness
- $x^* = \\arg\\max_x \\sigma(x) \\cdot p(x)$

**Batch Mode:**
- Select multiple diverse samples simultaneously
- Avoid redundancy in batch

**Multi-Objective:**
- Optimize multiple criteria (uncertainty + novelty + ...)
- Pareto optimization

---

**End of Presentation**

---

# Presentation Notes for Instructor

## Timing Suggestions
- Slides 1-6: Introduction & AutoRA overview (15 min)
- Slides 7-9: Experimentalists & Active Learning (10 min)
- Slides 10-11: Uncertainty Sampling (10 min)
- Slides 12-13: Disagreement Sampling (10 min)
- Slides 14-16: Comparison & Guidelines (10 min)
- Slides 17-21: Advanced Topics & Wrap-up (15 min)
- **Total**: ~70 minutes (leaves 10 min for questions in 80-min session)

## Interactive Elements
- **Slide 9**: Poll students on which samples they'd choose
- **Slide 14**: Show live demo if possible
- **Slide 16**: Walk through actual code
- **Slide 21**: Group discussion

## Key Messages
1. Active learning is about **efficiency**
2. AutoRA makes closed-loop experiments **practical**
3. Different strategies for different problems
4. Students will implement this in projects!
