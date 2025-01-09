## Installation Instructions

In this file you get a step-by-step instruction on how to install everything you need for the course *Optimizing Experimental Design* from scratch.

If you have something already installed you can skip the respective step.

### Step 1: Install Python

Download and install Python:

- Visit the [official Python website](https://www.python.org/) to download the latest version of Python.
- During installation, make sure to check the option that adds Python to your system's PATH.

### Step 2: Install Git

Download and install Git:

- Visit the [official Git website](https://git-scm.com/) to download the latest version of Git.
- Follow the installation instructions for your operating system.

### Step 3: Clone the Git Repository

- Open a terminal or command prompt.
- Go to the directory where you want to store everything regarding the course:
```bash
cd <directory_name>
```
- Clone the Git repository:
```bash
git clone https://github.com/whyhardt/CourseOptimizingExperimentalDesign.git
```
- Change into the cloned repository:
```bash
cd CourseOptimizingExperimentalDesign
```

### Step 4: Set Up a Virtual Environment

- Create a virtual environment:
```bash 
python -m venv venv
```
- Activate the virtual environment:
--> On Windows:
```bash
.\venv\Scripts\activate
```
--> On Unix or MacOS:
```bash
source venv/bin/activate
```

### Step 5: Install Required Packages
- Install `ipykernel` and other dependencies from the `requirements.txt` file:
```bash
pip install ipykernel
pip install -r requirements.txt
```
We also need Pytorch for a neural network regressor which we will use in the course. 
Since Pytorch is a huge package and we do not need necessarily GPU support due to the simplicity of the neural network model, we will install only its CPU-Version which safes a lot disk space:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Now you can either use your own IDE (needs to support jupyter notebooks) or follow the next steps to run jupyter notebooks in your browser. Alternatively you can use online services like Kaggle or Google Colab.

### Step 6: Create a Jupyter Kernel

- Install the Jupyter kernel for your virtual environment:
```bash
python -m ipykernel install --user --name=oed
```

### Step 7: Start Jupyter Notebook

- Start Jupyter Notebook:
```bash
jupyter notebook
```
- In the *Jupyter* Notebook interface, create a new notebook or open one of the available ones from `tutorials` and select your newly created kernel `oed` from the *Kernel* menu.


## Restarting the environment and Jupyter

### Step 1: Navigate to the Project Directory

- Open a terminal or command prompt.
- Change into the directory where they cloned the Git repository:
```bash
cd <repository_directory>
```
Replace `<repository_directory>` with the name of the directory created when they cloned the repository.

### Step 2: Activate the Virtual Environment

--> On Windows:
```bash
.\venv\Scripts\activate
```
--> On Unix or MacOS:
```bash
source venv/bin/activate
```

### Step 3: Start Jupyter Notebook

- Start Jupyter Notebook:
```bash
jupyter notebook
```
- select your kernel `oed` from the *Kernel* menu.
