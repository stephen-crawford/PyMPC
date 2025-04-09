

A modular and extensible Python port of the [TUD-AMR MPC Planner](https://github.com/tud-amr/mpc_planner), designed for real-time motion planning under constraints. PyMPC brings together math utilities, solver-agnostic optimization backends (e.g. CasADi, OSQP), and a flexible constraint interface to support rapid prototyping and deployment of Model Predictive Control systems.

---

## 🚀 Features

- **Model Predictive Control (MPC)** framework in Python
- Modular design for swapping solvers or models
- Support for:
  - State/input constraints
  - Soft and hard constraints
  - Obstacle avoidance
- Built-in math utilities for dynamics, linearization, etc.
- Modification of the original [C++ codebase](https://github.com/tud-amr/mpc_planner)
- Unit-test friendly structure for rapid development

---

## 🛠 Installation

```bash
git clone https://github.com/stephen-crawford/PyMPC.git
cd PyMPC
pip install -e .
Requires Python 3.8+, NumPy, and optionally CasADi or OSQP depending on your backend.
```

Related Work
This library is a Python port of the excellent tud-amr/mpc_planner, originally written in C++. Our goal is to preserve its structure and intent while providing a more flexible, Pythonic interface for rapid development and experimentation.

Contributing
Contributions welcome! If you'd like to add models, constraints, or solver support, open a pull request or issue.

MIT License
