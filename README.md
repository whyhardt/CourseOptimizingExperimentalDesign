# Optimizing Experimental Design in Cognitive Science

A hands-on course introducing modern approaches to experimental design, from traditional factorial methods to automated closed-loop experimentation using active learning and information theory.

## 📋 Course Overview

This course teaches you how to design efficient experiments that maximize information gain while minimizing resources (time, participants, trials). You'll progress from classical experimental design methods to cutting-edge automated experimentation using the AutoRA framework.

### Learning Outcomes

By the end of this course, you will be able to:
- Design and implement synthetic experiments for testing experimental design strategies
- Apply different sampling strategies (random, Latin hypercube, factorial)
- Use information-theoretic principles to optimize experimental designs
- Implement uncertainty-based active learning for intelligent sampling
- Build automated closed-loop experiments using AutoRA
- Evaluate and compare different experimental design strategies

---

## 🗓️ Course Structure

### **Day 1: Foundations**
- **Session 1**: Introduction to Experimental Design
  - Definition and terminology
  - Examples from cognitive science (Stroop, 2AFC, Multi-Armed Bandit)
  - Limitations and resource constraints

- **Session 2**: Simulation-Based Experimentation
  - Implementing synthetic experiments
  - Ground truth functions and noise
  - Model recovery and validation
  - *Tutorial: [syntheticexperiments.ipynb](tutorials/syntheticexperiments.ipynb)*

- **Session 3**: Random Sampling Methods
  - Normal, uniform, and Latin hypercube sampling
  - Coverage analysis and comparison
  - When to use each strategy
  - *Tutorial: [randomsampling.ipynb](tutorials/randomsampling.ipynb)*

- **Session 4**: Factorial Experimental Design
  - Full factorial designs
  - Main effects vs. interactions
  - Factorial explosion and fractional designs
  - *Tutorial: [factorialdesign.ipynb](tutorials/factorialdesign.ipynb)*

### **Day 2: Active Learning & Automation**

- **Session 5**: Introduction to AutoRA & Closed-Loop Experimentation
  - The AutoRA framework (Experimentalist → Experiment Runner → Theorist)
  - State management and workflow
  - Simple experimentalists (grid, random)
  - Building your first closed-loop experiment
  - *Tutorial: [autora_intro.ipynb](tutorials/autora_intro.ipynb)*

- **Session 6**: Information Theory Basics
  - Entropy and information content
  - Conditional entropy and mutual information
  - Application to experimental design
  - *Tutorial: [informationtheory.ipynb](tutorials/informationtheory.ipynb)*

- **Session 7**: Uncertainty-Based Active Learning
  - Uncertainty sampling principles
  - AutoRA uncertainty experimentalist
  - Comparing random vs. uncertainty-based sampling
  - *Tutorial: [autora_uncertainty.ipynb](tutorials/autora_uncertainty.ipynb)*

- **Session 8**: Advanced Active Learning
  - Model disagreement and ensembles
  - AutoRA disagreement experimentalist
  - Comparing uncertainty vs. disagreement strategies
  - *Tutorial: [autora_advanced.ipynb](tutorials/autora_advanced.ipynb)*

### **Day 3: Projects & Presentations**
- Student paper presentations (40% of grade)
- Group project kickoff
- Q&A and consultation

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Basic knowledge of Python programming
- Familiarity with NumPy and Matplotlib
- Basic understanding of statistics and machine learning

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/CourseOptimizingExperimentalDesign.git
cd CourseOptimizingExperimentalDesign
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

4. **Install AutoRA with experimentalists**
```bash
pip install -U "autora[experimentalist-uncertainty]"
pip install -U "autora[experimentalist-inequality]"
pip install -U "autora[experimentalist-novelty]"
# Or install all experimentalists at once:
# pip install -U "autora[all-experimentalists]"
```

5. **Verify installation**
```bash
python -c "import autora; import sklearn; import torch; print('All packages installed successfully!')"
```

See [installation_instructions.md](installation_instructions.md) for detailed troubleshooting.

---

## 📚 Course Materials

### Presentations
Located in `~/Downloads/pdf/`:
1. `01_Introduction_OED.pdf` - Introduction to experimental design
2. `02_RandomSamplingMethods.pdf` - Random sampling strategies
3. `03_FactorialExperimentalDesign.pdf` - Factorial designs
4. `04_InformationTheory.pdf` - Information theory basics
5. `05_AutoRA.pdf` - AutoRA framework and active learning

### Tutorials
All tutorials are Jupyter notebooks in the `tutorials/` directory:

| Tutorial | Topic | Duration |
|----------|-------|----------|
| `syntheticexperiments.ipynb` | Synthetic experiment setup | 60 min |
| `randomsampling.ipynb` | Random sampling comparison | 45 min |
| `factorialdesign.ipynb` | Factorial designs | 30 min |
| `autora_intro.ipynb` | AutoRA basics & closed-loop | 60 min |
| `informationtheory.ipynb` | Information theory concepts | 45 min |
| `autora_uncertainty.ipynb` | Uncertainty-based sampling | 60 min |
| `autora_advanced.ipynb` | Disagreement & ensembles | 60 min |

### Supporting Code
- `resources/synthetic.py` - Synthetic experiment utilities
- `resources/regressors.py` - Neural network models
- `resources/sampler.py` - Sampling strategies

---

## 📝 Assessment

### Group Project (100%)

**Part 1: Paper Presentation (40%)**
- Due: January 25, 2025
- Choose a paper on optimal experimental design or active learning
- Present: content, applicability, benefits, limitations (20 min + 10 min discussion)

**Part 2: Python Implementation (60%)**
- Due: February 25, 2025
- Apply the presented approach to the 2AFC demo experiment
- Compare efficiency/effectiveness vs. baseline random sampling
- Document thoroughly with comments, text sections, and plots

**Bonus: Paper Discussion (up to 10%)**
- Active participation in paper presentation discussions

### Requirements
1. Mandatory course attendance
2. All group project components submitted on time
3. Clear documentation and reproducible code

---

## 🎯 Demo Experiment: 2-Alternative Forced Choice (2AFC)

The course uses a 2-Alternative Forced Choice experiment as a running example:

**Task**: Participants view a grid of colored tiles and identify which color is more prevalent

**Controllable Factors**:
- `ratio`: Proportion of blue vs. orange tiles (0 = all orange, 1 = balanced)
- `scatteredness`: Spatial randomness of tiles (0 = segregated, 1 = random)

**Observations**:
- Response time (continuous)
- Accuracy (binary)

This experiment allows testing various design strategies in a controlled setting.

---

## 📖 Key Concepts Covered

### Experimental Design Terminology
- **Factors**: Types of stimuli (e.g., ratio, scatteredness)
- **Levels**: Stimuli intensity values
- **Treatment**: Specific combination of factor levels
- **Design Space**: All possible treatment combinations
- **Run**: Single execution of an experimental unit
- **Sample Size**: Total number of runs

### Active Learning Concepts
- **Query Strategy**: Method for selecting informative samples
- **Uncertainty Sampling**: Select samples with highest prediction uncertainty
- **Disagreement Sampling**: Select samples where models disagree most
- **Closed-Loop**: Automated cycle of experimentation → modeling → new experiments

### Information Theory
- **Entropy**: Expected information content
- **Conditional Entropy**: Remaining uncertainty given conditions
- **Mutual Information**: Shared information between variables

---

## 🔗 Additional Resources

### Research Papers
- Musslick et al. (2024). AutoRA: Automated Research Assistant for Closed-Loop Empirical Research. *Journal of Open Source Software*.
- Musslick et al. (2023). An Evaluation of Experimental Sampling Strategies for Autonomous Empirical Research in Cognitive Science. *Proceedings of CogSci*.
- Settles, B. (2009). Active Learning Literature Survey. University of Wisconsin-Madison.

### Documentation
- [AutoRA Documentation](https://autoresearch.github.io/autora/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Information Theory Tutorial](https://colah.github.io/posts/2015-09-Visual-Information/)

### Tools & Frameworks
- [AutoRA](https://github.com/AutoResearch/autora) - Automated research assistant
- [SweetPea](https://sites.google.com/view/sweetpea-ai) - Experimental design constraints
- [PsychoPy](https://www.psychopy.org/) - Psychology experiment builder

---

## 🤝 Contributing

Found an issue or have suggestions? Please:
1. Check existing issues
2. Open a new issue with details
3. Or submit a pull request

---

## 📧 Contact

**Instructor**: [Your Name]
**Email**: [your.email@university.edu]
**Office Hours**: [Days/Times]

---

## 📄 License

This course material is licensed under [LICENSE TYPE]. Feel free to use and adapt for educational purposes with attribution.

---

## 🙏 Acknowledgments

This course builds on:
- The AutoRA framework by Musslick et al.
- modAL active learning library
- Examples from cognitive science literature
- Student feedback from previous offerings

---

**Last Updated**: January 2025
**Version**: 2.0 (AutoRA integration)
