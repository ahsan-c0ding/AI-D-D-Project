# AI Dungeon Master
An Interactive Showcase of Classical AI Algorithms for Procedural Dungeon Generation and Gameplay

### Team Members
* Abdul Rahim (30609)
* Ahsan Haris (30485)
* Syed Junaid Iqbal (30568)
* Muhammad Umer (30622) 

**Course:** Introduction to Artificial Intelligence (Spring 2026)
**Instructor:** Dr. Syed Ali Raza 
**Institution:** Institute of Business Administration, Karachi 

---

## Project Overview
This project presents the AI Dungeon Master, an interactive application combining procedural dungeon generation, NPC decision-making, and Bayesian combat reasoning into a playable tile-based adventure.It serves as both a playable game and an educational showcase of classical AI concepts.

The system implements the following core AI techniques:
* **Procedural Dungeon Generation:** Driven by four algorithms (CSP with backtracking, BFS, DFS, and a Greedy heuristic), visualized step by step.
* **NPC Decision-Making:** NPCs are controlled by five binary decision trees exposed in a live inspector.
* **Combat & Skill Checks:** Governed by a Bayesian probability model that updates based on attacker and defender stats.

---

## Project Deliverables & Links

All required deliverables for the course project have been submitted. Here is where to find them:

* **Demo Video:** The recorded presentation and full code/GUI demonstration can be viewed on Google Drive:
  [https://drive.google.com/drive/folders/11hmR1sF4Fi0xGgBoMMgUQwvtMHFGcTo0?usp=sharing](https://drive.google.com/drive/folders/11hmR1sF4Fi0xGgBoMMgUQwvtMHFGcTo0?usp=sharing)
* **Project Report:** The comprehensive PDF report detailing the problem statement, methodology, evaluation metrics, and results is uploaded to the LMS. 
* **Source Code:** The complete codebase is available in this GitHub repository and has also been submitted as a zipped file on the LMS.

---

## File Structure & AI Modules

The application is built using pure Python and Tkinter, with clean module separation.

* `gui.py`: The main entry point.Contains the 5-tab Tkinter interface (Generation Inspector, NPC Brain, Combat Math, Algorithm Comparison, and Play).
* `game.py`: The game state controller that wires the AI components into a playable D&D ruleset.
* `csp_generator.py`: Implementation of the Constraint Satisfaction Problem (CSP) solver for map generation with backtracking.
* `comparison_generators.py`: Implementations of the BFS, DFS, and Greedy search algorithms used for benchmarking against the CSP.
* `npc_decision_tree.py`: The explicit binary decision trees governing the five NPC archetypes (Enemy, Boss, Merchant, Friendly, Neutral).
* `bayesian_combat.py`: The probability models handling attack resolution, damage calculations, and skill checks.
* `run_evaluation.py` (and the `evaluation/` folder): The automated testing suite used to run 36,000 simulated attacks and evaluate 120 generated dungeons for the project report.
* `models.py`: Core data classes (Player, Room, NPC, Item, etc.) shared across the modules.

---

## Installation & Setup

1. Clone this repository or extract the provided zip file from the LMS.
2. Open a terminal or command prompt and navigate to the project folder.
3. (Optional but recommended) Create and activate a virtual environment.
    ``` bash
   python -m venv venv
   ```
    Then.
    ``` bash
   venv\Scripts\activate 
   ```
5. Install the required dependencies. The core game runs on standard Python, but the mathematical plots in the GUI require Matplotlib:
   ``` bash
   pip install matplotlib
   ```
6. Run the application:
   ``` bash
   python gui.py 
   ```

## Running the Automated Evaluations

To reproduce the data, metrics, and charts found in our project report, you can run the master evaluation script:
``` bash
python run_evaluation.py
```
This will run the full sweep for dungeon generation comparison, NPC decision-tree evaluation, and Bayesian combat calibration. All output charts and CSVs will be saved in the `results/` directory.
