"""Check that everything for the course is installed: python check_installation.py (in the course folder)."""

import importlib
import sys

PACKAGES = ["numpy", "scipy", "pandas", "matplotlib", "sklearn", "torch", "autora", "sweetpea", "pygame", "ipykernel"]

problems = []
print(f"Python {sys.version.split()[0]}")
if not (3, 10) <= sys.version_info[:2] <= (3, 12):
    problems.append("Python 3.10, 3.11 or 3.12 is needed (3.11 recommended)")

for name in PACKAGES:
    try:
        module = importlib.import_module(name)
        print(f"  ok  {name} {getattr(module, '__version__', '')}")
    except Exception as error:
        print(f"  --  {name}: {error}")
        problems.append(f"{name} could not be imported")

if "sweetpea" not in " ".join(problems):
    print("Testing SweetPea (the first run may download a few solver programs) ...")
    try:
        from sweetpea import CrossBlock, Factor

        from resources.design_tools import synthesize

        color = Factor("color", ["red", "green"])
        synthesize(CrossBlock([color], [color], []))
        print("  ok  SweetPea can sample sequences")
    except Exception as error:
        print(f"  --  SweetPea: {error}")
        problems.append("SweetPea could not sample a sequence")

if problems:
    print("\nSomething is missing:")
    for problem in problems:
        print(" -", problem)
    sys.exit(1)
print("\nAll good! Last check: python -m experiments.stroop --practice")
