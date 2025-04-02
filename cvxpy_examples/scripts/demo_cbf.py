"""Demo of the CVXPY CBF

This is a recreation of the joint-limit-avoidance demo from CBFpy
https://github.com/danielpmorton/cbfpy/blob/main/cbfpy/examples/joint_limits_demo.py

(the CBFpy version will run faster, but this is a good example of how to use CVXPY
for optimal control)
"""

import pybullet
from pybullet_utils.bullet_client import BulletClient
import numpy as np
from jax import Array
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from cvxpy_examples.problems.cvxpy_cbf import CBF


class JointLimitsEnv:
    """Simulation environment for the 3-DOF arm joint-limit-avoidance demo

    This includes a desired reference trajectory which is unsafe: it will command sinusoidal
    joint motions (with different frequencies per link) that will exceed the joint limits of the robot
    """

    def __init__(self):
        self.client: pybullet = BulletClient(pybullet.GUI)
        urdf = "cvxpy_examples/assets/three_dof_arm.urdf"
        self.robot = pybullet.loadURDF(urdf, useFixedBase=True)
        self.num_joints = self.client.getNumJoints(self.robot)
        self.q_min = np.array(
            [self.client.getJointInfo(self.robot, i)[8] for i in range(self.num_joints)]
        )
        self.q_max = np.array(
            [self.client.getJointInfo(self.robot, i)[9] for i in range(self.num_joints)]
        )
        self.timestep = self.client.getPhysicsEngineParameters()["fixedTimeStep"]
        self.t = 0

        # Sinusoids for the desired joint positions
        # Setting the amplitude to be the full joint range means we will command DOUBLE
        # the joint range, exceeding our limits
        self.omegas = 0.1 * np.array([1.0, 2.0, 3.0])
        self.amps = self.q_max - self.q_min
        self.offsets = np.zeros(3)

    def step(self):
        self.client.stepSimulation()
        self.t += self.timestep

    def get_state(self):
        states = self.client.getJointStates(self.robot, range(self.num_joints))
        return np.array([states[i][0] for i in range(self.num_joints)])

    def get_desired_state(self):
        # Evaluate our unsafe sinusoidal trajectory
        return self.amps * np.sin(self.omegas * self.t) + self.offsets

    def apply_control(self, u):
        self.client.setJointMotorControlArray(
            self.robot,
            list(range(self.num_joints)),
            self.client.VELOCITY_CONTROL,
            targetVelocities=u,
        )


class JointLimitsCBF(CBF):
    """CBF for enforcing joint limits in a 3-DOF velocity-controlled manipulator"""

    def __init__(self):
        self.num_joints = 3

        # True joint limit values from the URDF
        self.q_min = -np.pi / 2 * np.ones(self.num_joints)
        self.q_max = np.pi / 2 * np.ones(self.num_joints)

        # Add some padding so that the true joint limits are not violated
        self.padding = 0.3

        super().__init__(n=self.num_joints, m=self.num_joints)

    def f(self, z):
        return jnp.zeros(self.num_joints)

    def g(self, z):
        return jnp.eye(self.num_joints)

    def h(self, z):
        # Note: here, z is the current joint positions
        return jnp.concatenate(
            [self.q_max - z - self.padding, z - self.q_min - self.padding]
        )

    def alpha(self, h_z):
        return h_z


def nominal_controller(q: Array, q_des: Array) -> Array:
    """Very simple proportional controller: Commands joint velocities to reduce a position error

    Args:
        q (Array): Joint positions, shape (num_joints,)
        q_des (Array): Desired joint positions, shape (num_joints,)

    Returns:
        Array: Joint velocity command, shape (num_joints,)
    """
    k = 1.0
    return k * (q_des - q)


def main():
    cbf = JointLimitsCBF()
    env = JointLimitsEnv()

    q_hist = []
    q_des_hist = []
    u_safe_hist = []
    u_unsafe_hist = []
    sim_time = 100.0
    timestep = env.timestep
    num_timesteps = int(sim_time / timestep)
    for i in range(num_timesteps):
        q = env.get_state()
        q_des = env.get_desired_state()
        u_unsafe = nominal_controller(q, q_des)
        cbf.update_params(q, u_unsafe)
        cbf.solve()
        u = cbf.optimal_control
        env.apply_control(u)
        env.step()
        q_hist.append(q)
        q_des_hist.append(q_des)
        u_unsafe_hist.append(u_unsafe)
        u_safe_hist.append(u)

    ## Plotting ##

    fig, axs = plt.subplots(2, 3)
    ts = timestep * np.arange(num_timesteps)

    # On the top row, plot the q and q des for each joint, along with the joint limits indicated
    for i in range(3):
        (q_line,) = axs[0, i].plot(ts, np.array(q_hist)[:, i], label="q")
        (q_des_line,) = axs[0, i].plot(ts, np.array(q_des_hist)[:, i], label="q_des")
        axs[0, i].plot(
            ts,
            env.q_min[i] * np.ones(num_timesteps),
            ls="--",
            c="red",
            label="q_min (True)",
        )
        axs[0, i].plot(
            ts,
            env.q_max[i] * np.ones(num_timesteps),
            ls="--",
            c="red",
            label="q_max (True)",
        )
        axs[0, i].plot(
            ts,
            (env.q_min[i] + cbf.padding) * np.ones(num_timesteps),
            ls="--",
            c="blue",
            label="q_min (CBF)",
        )
        axs[0, i].plot(
            ts,
            (env.q_max[i] - cbf.padding) * np.ones(num_timesteps),
            ls="--",
            c="blue",
            label="q_max (CBF)",
        )
        legend_elements = [
            q_line,
            q_des_line,
            Line2D([0], [0], color="red", ls="--", label="True limits"),
            Line2D([0], [0], color="blue", ls="--", label="CBF limits"),
        ]
        axs[0, i].legend(handles=legend_elements, loc="lower left")
        axs[0, i].set_title(f"Joint {i} position")

    # On the bottom row, plot the safe and unsafe velocities for each joint
    for i in range(3):
        axs[1, i].plot(ts, np.array(u_safe_hist)[:, i], label="Safe")
        axs[1, i].plot(ts, np.array(u_unsafe_hist)[:, i], label="Unsafe")
        axs[1, i].legend(loc="lower left")
        axs[1, i].set_title(f"Joint {i} velocity")

    plt.show()


if __name__ == "__main__":
    main()
