# CVXPY Examples

This repo was created to help teach `cvxpy` in AA203: Optimal and Learning-based Control

For the course website, see https://stanfordasl.github.io/aa203

## Installation

```
git clone https://github.com/danielpmorton/cvxpy_examples
cd cvxpy_examples
pip install -e .
```

I recommend working inside a virtual environment. I'm using Python 3.10.8, for reference.

## Examples

### Maximum inscibed ball in a polyhedron

`cvxpy_examples/scripts/demo_inscribed_ball.py`

![inscribed_balls](https://github.com/user-attachments/assets/cdf795d4-3a5a-4d96-aea3-3842036d4dbc)

For more details, check out *Task-Driven Manipulation with Reconfigurable Parallel Robots* -- [website](https://stanfordasl.github.io/reachbot_manipulation/), [code](https://github.com/StanfordASL/reachbot_manipulation), [arXiv](https://arxiv.org/abs/2403.10768)

### Simple manipulator joint limits CBF

`cvxpy_examples/scripts/demo_cbf.py`

![joint_limits](https://github.com/user-attachments/assets/3f8aac7f-2191-4077-a22b-30130e6d8c01)

For more details, check out [CBFpy](https://github.com/danielpmorton/cbfpy) or *Safe, Task-Consistent Manipulation with Operational Space Control Barrier Functions* -- [website](https://stanfordasl.github.io/oscbf/), [code](https://github.com/StanfordASL/oscbf), [arXiv](https://arxiv.org/abs/2503.06736)
