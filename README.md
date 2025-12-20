# ROB6323 Go2 Project — Isaac Lab

This repository is the starter code for the NYU Reinforcement Learning and Optimal Control project in which students train a Unitree Go2 walking policy in Isaac Lab starting from a minimal baseline and improve it via reward shaping and robustness strategies. Please read this README fully before starting and follow the exact workflow and naming rules below to ensure your runs integrate correctly with the cluster scripts and grading pipeline.

## Repository policy

- Fork this repository and do not change the repository name in your fork.  
- Your fork must be named rob6323_go2_project so cluster scripts and paths work without modification.

### Prerequisites

- **GitHub Account:** You must have a GitHub account to fork this repository and manage your code. If you do not have one, [sign up here](https://github.com/join).

### Links
1.  **Project Webpage:** [https://machines-in-motion.github.io/RL_class_go2_project/](https://machines-in-motion.github.io/RL_class_go2_project/)
2.  **Project Tutorial:** [https://github.com/machines-in-motion/rob6323_go2_project/blob/master/tutorial/tutorial.md](https://github.com/machines-in-motion/rob6323_go2_project/blob/master/tutorial/tutorial.md)

## Connect to Greene

- Connect to the NYU Greene HPC via SSH; if you are off-campus or not on NYU Wi‑Fi, you must connect through the NYU VPN before SSHing to Greene.  
- The official instructions include example SSH config snippets and commands for greene.hpc.nyu.edu and dtn.hpc.nyu.edu as well as VPN and gateway options: https://sites.google.com/nyu.edu/nyu-hpc/accessing-hpc?authuser=0#h.7t97br4zzvip.

## Clone in $HOME

After logging into Greene, `cd` into your home directory (`cd $HOME`). You must clone your fork into `$HOME` only (not scratch or archive). This ensures subsequent scripts and paths resolve correctly on the cluster. Since this is a private repository, you need to authenticate with GitHub. You have two options:

### Option A: Via VS Code (Recommended)
The easiest way to avoid managing keys manually is to configure **VS Code Remote SSH**. If set up correctly, VS Code forwards your local credentials to the cluster.
- Follow the [NYU HPC VS Code guide](https://sites.google.com/nyu.edu/nyu-hpc/training-support/general-hpc-topics/vs-code) to set up the connection.

> **Tip:** Once connected to Greene in VS Code, you can clone directly without using the terminal:
> 1. **Sign in to GitHub:** Click the "Accounts" icon (user profile picture) in the bottom-left sidebar. If you aren't signed in, click **"Sign in with GitHub"** and follow the browser prompts to authorize VS Code.
> 2. **Clone the Repo:** Open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`), type **Git: Clone**, and select it.
> 3. **Select Destination:** When prompted, select your home directory (`/home/<netid>/`) as the clone location.
>
> For more details, see the [VS Code Version Control Documentation](https://code.visualstudio.com/docs/sourcecontrol/intro-to-git#_clone-a-repository-locally).

### Option B: Manual SSH Key Setup
If you prefer using a standard terminal, you must generate a unique SSH key on the Greene cluster and add it to your GitHub account:
1. **Generate a key:** Run the `ssh-keygen` command on Greene (follow the official [GitHub documentation on generating a new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key)).
2. **Add the key to GitHub:** Copy the output of your public key (e.g., `cat ~/.ssh/id_ed25519.pub`) and add it to your account settings (follow the [GitHub documentation on adding a new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)).

### Execute the Clone
Once authenticated, run the following commands. Replace `<your-git-ssh-url>` with the SSH URL of your fork (e.g., `git@github.com:YOUR_USERNAME/rob6323_go2_project.git`).
```
cd $HOME
git clone <your-git-ssh-url> rob6323_go2_project
```
*Note: You must ensure the target directory is named exactly `rob6323_go2_project`. This ensures subsequent scripts and paths resolve correctly on the cluster.*
## Install environment

- Enter the project directory and run the installer to set up required dependencies and cluster-side tooling.  
```
cd $HOME/rob6323_go2_project
./install.sh
```
Do not skip this step, as it configures the environment expected by the training and evaluation scripts. It will launch a job in burst to set up things and clone the IsaacLab repo inside your greene storage. You must wait until the job in burst is complete before launching your first training. To check the progress of the job, you can run `ssh burst "squeue -u $USER"`, and the job should disappear from there once it's completed. It takes around **30 minutes** to complete. 
You should see something similar to the screenshot below (captured from Greene):

![Example burst squeue output](docs/img/burst_squeue_example.png)

In this output, the **ST** (state) column indicates the job status:
- `PD` = pending in the queue (waiting for resources).
- `CF` = instance is being configured.
- `R`  = job is running.

On burst, it is common for an instance to fail to configure; in that case, the provided scripts automatically relaunch the job when this happens, so you usually only need to wait until the job finishes successfully and no longer appears in `squeue`.

## What to edit

- In this project you'll only have to modify the two files below, which define the Isaac Lab task and its configuration (including PPO hyperparameters).  
  - source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env.py  
  - source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env_cfg.py
PPO hyperparameters are defined in source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/agents/rsl_rl_ppo_cfg.py, but you shouldn't need to modify them.

## How to edit

- Option A (recommended): Use VS Code Remote SSH from your laptop to edit files on Greene; follow the NYU HPC VS Code guide and connect to a compute node as instructed (VPN required off‑campus) (https://sites.google.com/nyu.edu/nyu-hpc/training-support/general-hpc-topics/vs-code). If you set it correctly, it makes the login process easier, among other things, e.g., cloning a private repo.
- Option B: Edit directly on Greene using a terminal editor such as nano.  
```
nano source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env.py
```
- Option C: Develop locally on your machine, push to your fork, then pull changes on Greene within your $HOME/rob6323_go2_project clone.

> **Tip:** Don't forget to regularly push your work to github

## Launch training

- From $HOME/rob6323_go2_project on Greene, submit a training job via the provided script.  
```
cd "$HOME/rob6323_go2_project"
./train.sh
```
- Check job status with SLURM using squeue on the burst head node as shown below.  
```
ssh burst "squeue -u $USER"
```
Be aware that jobs can be canceled and requeued by the scheduler or underlying provider policies when higher-priority work preempts your resources, which is normal behavior on shared clusters using preemptible partitions.

## Where to find results

- When a job completes, logs are written under logs in your project clone on Greene in the structure logs/[job_id]/rsl_rl/go2_flat_direct/[date_time]/.  
- Inside each run directory you will find a TensorBoard events file (events.out.tfevents...), neural network checkpoints (model_[epoch].pt), YAML files with the exact PPO and environment parameters, and a rollout video under videos/play/ that showcases the trained policy.  

## Download logs to your computer

Use `rsync` to copy results from the cluster to your local machine. It is faster and can resume interrupted transfers. Run this on your machine (NOT on Greene):

```
rsync -avzP -e 'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' <netid>@dtn.hpc.nyu.edu:/home/<netid>/rob6323_go2_project/logs ./
```

*Explanation of flags:*
- `-a`: Archive mode (preserves permissions, times, and recursive).
- `-v`: Verbose output.
- `-z`: Compresses data during transfer (faster over network).
- `-P`: Shows progress bar and allows resuming partial transfers.

## Visualize with TensorBoard

You can inspect training metrics (reward curves, loss values, episode lengths) using TensorBoard. This requires installing it on your local machine.

1.  **Install TensorBoard:**
    On your local computer (do NOT run this on Greene), install the package:
    ```
    pip install tensorboard
    ```

2.  **Launch the Server:**
    Navigate to the folder where you downloaded your logs and start the server:
    ```
    # Assuming you are in the directory containing the 'logs' folder
    tensorboard --logdir ./logs
    ```

3.  **View Metrics:**
    Open your browser to the URL shown (usually `http://localhost:6006/`).

## Debugging on Burst

Burst storage is accessible only from a job running on burst, not from the burst login node. The provided scripts do not automatically synchronize error logs back to your home directory on Greene. However, you will need access to these logs to debug failed jobs. These error logs differ from the logs in the previous section.

The suggested way to inspect these logs is via the Open OnDemand web interface:

1.  Navigate to [https://ood-burst-001.hpc.nyu.edu](https://ood-burst-001.hpc.nyu.edu).
2.  Select **Files** > **Home Directory** from the top menu.
3.  You will see a list of files, including your `.err` log files.
4.  Click on any `.err` file to view its content directly in the browser.

> **Important:** Do not modify anything inside the `rob6323_go2_project` folder on burst storage. This directory is managed by the job scripts, and manual changes may cause synchronization issues or job failures.

## Project scope reminder

- The assignment expects you to go beyond velocity tracking by adding principled reward terms (posture stabilization, foot clearance, slip minimization, smooth actions, contact and collision penalties), robustness via domain randomization, and clear benchmarking metrics for evaluation as described in the course guidelines.  
- Keep your repository organized, document your changes in the README, and ensure your scripts are reproducible, as these factors are part of grading alongside policy quality and the short demo video deliverable.

## Resources

- [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/ecosystem.html) — Everything you need to know about IsaacLab, and more!
- [Isaac Lab ANYmal C environment](https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c) — This targets ANYmal C (not Unitree Go2), so use it as a reference and adapt robot config, assets, and reward to Go2.
- [DMO (IsaacGym) Go2 walking project page](https://machines-in-motion.github.io/DMO/) • [Go2 walking environment used by the authors](https://github.com/Jogima-cyber/IsaacGymEnvs/blob/e351da69e05e0433e746cef0537b50924fd9fdbf/isaacgymenvs/tasks/go2_terrain.py) • [Config file used by the authors](https://github.com/Jogima-cyber/IsaacGymEnvs/blob/e351da69e05e0433e746cef0537b50924fd9fdbf/isaacgymenvs/cfg/task/Go2Terrain.yaml) — Look at the function `compute_reward_CaT` (beware that some reward terms have a weight of 0 and thus are deactivated, check weights in the config file); this implementation includes strong reward shaping, domain randomization, and training disturbances for robust sim‑to‑real, but it is written for legacy IsaacGym and the challenge is to re-implement it in Isaac Lab.
- **API References**:
    - [ArticulationData (`robot.data`)](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.ArticulationData) — Contains `root_pos_w`, `joint_pos`, `projected_gravity_b`, etc.
    - [ContactSensorData (`_contact_sensor.data`)](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html#isaaclab.sensors.ContactSensorData) — Contains `net_forces_w` (contact forces).

---
Students should only edit README.md below this ligne.

# Group: Christian Hahn, Greta Perez-Haiek, Archit Sharma

The below code explains the changes which we made to the environment and configuration files based on the requirements of the Tutorial and the other Project requirements. Many of the tutorial changes did not require alteration to function and so are not noted here for the sake of brevity, simply being inserted as was instructed in the Tutorial. This accounts for essentially all changes in steps 1-4. What follows are the significant changes or portions of code written to improve the functionality of our simulation.

### Tutorial Additional rewards:
This implements the additional reward terms from tutorial step 5.2, inserted inside of `_get_rewards()`. The relevant reward keywords were then also added to key in `__init__()`
```python
# action rate penalization
# First derivative (Current - Last)
rew_action_rate = torch.sum(torch.square(self._actions - self.last_actions[:, :, 0]), dim=1) * (self.cfg.action_scale ** 2)
# Second derivative (Current - 2*Last + 2ndLast)
rew_action_rate += torch.sum(torch.square(self._actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1) * (self.cfg.action_scale ** 2)

# 1. Penalize non-vertical orientation (projected gravity on XY plane)
rew_orient = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)

# 2. Penalize vertical velocity (z-component of base linear velocity)
rew_lin_vel_z = torch.square(self.robot.data.root_lin_vel_b[:, 2])

# 3. Penalize high joint velocities
rew_dof_vel = torch.sum(torch.square(self.robot.data.joint_vel), dim=1)

# 4. Penalize angular velocity in XY plane (roll/pitch)
rew_ang_vel_xy = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1)

# 5. Action Regularization (L2 norm of actions)
rew_action_mag = torch.sum(torch.square(self.actions), dim = 1)

# Update the prev action hist (roll buffer and insert new action)
self.last_actions = torch.roll(self.last_actions, 1, 2)
self.last_actions[:, :, 0] = self._actions[:]

# Add the implemented rewards dict, akongside other rewards
        rewards = {
            ...
            "rew_action_rate": rew_action_rate * self.cfg.action_rate_reward_scale,
            "rew_action_mag": rew_action_mag * self.cfg.action_mag_reward_scale, 
            "orient": rew_orient * self.cfg.orient_reward_scale,
            "lin_vel_z": rew_lin_vel_z * self.cfg.lin_vel_z_reward_scale,
            "dof_vel": rew_dof_vel * self.cfg.dof_vel_reward_scale,
            "ang_vel_xy": rew_ang_vel_xy * self.cfg.ang_vel_xy_reward_scale,
            }
```

### Feet and Sensor Indicies
This is our implementation of Tutorial Step 6.2, where we find the individual indicies of the Feet within the context of the Sensor.
```python
# add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
self.set_debug_vis(self.cfg.debug_vis)

# Get specific body indices
self._feet_ids = []
self._feet_ids_sensor = []
foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

for name in foot_names:
    foot_id, _ = self.robot.find_bodies(name)
    sensor_id, _ = self._contact_sensor.find_bodies(name)
    self._feet_ids.append(foot_id[0])
    self._feet_ids_sensor.append(sensor_id[0])
```

### Feet Reward Implementations
This function implements a reward for foot clearance.
```python
def _reward_feet_clearance(self) -> torch.Tensor:
        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
        foot_height = self.foot_positions_w[:, :, 2]

        target_height = 0.08 * phases + 0.02

        rew_foot_clearance = torch.square(target_height - foot_height) * (1.0 - self.desired_contact_states)
        rew_feet_clearance = torch.sum(rew_foot_clearance, dim=1)
        return rew_feet_clearance
```

This function implements a reward for low contact forces on each of the feet.
```python
def _reward_tracking_contacts_shaped_force(self) -> torch.Tensor:
        net_forces = self._contact_sensor.data.net_forces_w_history
        latest_forces = net_forces[:, -1]
        force_norm = torch.norm(latest_forces, dim=-1)
        foot_forces = force_norm[:, self._feet_ids_sensor]

        rew_tracking_contacts_shaped_force = 0.0
        for i in range(4):
            rew_tracking_contacts_shaped_force += (1.0 - self.desired_contact_states[:, i]) * (1.0 - torch.exp(-foot_forces[:, i] ** 2 / 100.0))

        rew_tracking_contacts_shaped_force = rew_tracking_contacts_shaped_force / 4.0
        return rew_tracking_contacts_shaped_force
```

The above two functions, as well as `_reward_raibert_heuristic()` are used to extract three additional reward values, which are multiplied against `raibert_heuristic_reward_scale`,`feet_clearance_reward_scale`, and `tracking_contacts_shaped_force_reward_scale` from within the configurations file. This is implemented within `_get_rewards()` directly between `self.last_actions[:, :, 0] = self._actions[:]` and `rewards = ...`.

Note: the code for `_reward_raibert_heuristic()` is not listed here, as its implementation is identical to the one given within the Tutorial document.

```python
self._step_contact_targets() # Update gait state
rew_raibert_heuristic = self._reward_raibert_heuristic()
feet_reward = self._reward_feet_clearance()
contact_reward = self._reward_tracking_contacts_shaped_force()

rewards = {
            ...
            "raibert_heuristic": rew_raibert_heuristic * self.cfg.raibert_heuristic_reward_scale,
            ...
            "feet" : self.cfg.feet_clearance_reward_scale * feet_reward,
            "contact" : self.cfg.tracking_contacts_shaped_force_reward_scale * contact_reward,
        }
```

### PD Control for Joints with the Actuator Friction Model
```python
def _apply_action(self) -> None:
        # Compute PD torques
        torques = (
            self.Kp * (
                    self.desired_joint_pos 
                    - self.robot.data.joint_pos 
                )
                - self.Kd * self.robot.data.joint_vel
        )


        qd = self.robot.data.joint_vel
        tau_stiction = self.Fs * torch.tanh(qd / 0.1)
        tau_viscous = self.mu_v * qd
        tau_friction = tau_stiction + tau_viscous


        torques = torques - tau_friction


        # Apply torques to the robot
        torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
        self.robot.set_joint_effort_target(torques)
```

### Final Rewards Set
These are the rewards which we experimentally derived. In our implementation, they are contained at the end of `rob6323-go2_env_cfg`.
```python
    # reward scales
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    action_rate_reward_scale = -0.1

    # Raibert Heuristic reward scales
    raibert_heuristic_reward_scale = -10.0
    feet_clearance_reward_scale = -60.0 # changed from -30.0
    tracking_contacts_shaped_force_reward_scale = 4.0

    # Additional reward scales
    orient_reward_scale = -5.0
    lin_vel_z_reward_scale = -0.175 # changed from -0.02
    dof_vel_reward_scale = -0.00001 # changed from -0.0001
    ang_vel_xy_reward_scale = -0.005 # changed from -0.001
    action_mag_reward_scale = -1.0
```
Given the above code and our rewards, we were able to get a best-execution score of 23.8718 and 47.629 out of a desired 24 and 48 for `track_ang_vel_z_exp` and `track_lin_vel_xy_exp`, respectively. A video of this simulation is available in this fork, entitled `RL_Final_Resultss_Video.mp4`.

### Addendum: Uneven Terrain Locomotion
To implement training on uneven terrain several changes need to be made. Primarily, the terrain generation needs to be changed from plane to generator and configured as the desired type; eg - stepped pyramid or randomized uniform. 
Rewards such as feet_clearance need to be turned off or modified to use height from ground instead of height in world frame. Weights need to be changed to allow for more freedom in the z direction.

Due to the significant difference between the implementation of uneven terrain generation and the even-terrain, friction implementation, we decided to preserve both versions. The relevant configuration and environment files for uneven terrain are now stored in a sub-folder, named uneven_terrain, within this git fork and should function when swapped for the existing files within the source directory.
```python
terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
```
