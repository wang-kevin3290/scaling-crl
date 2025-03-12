# 1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities
Paper: https://wang-kevin3290.github.io/scaling-crl/static/pdf/scaling_rl.pdf
Code: https://github.com/wang-kevin3290/scaling-crl
Email Kevin Wang (kw6487@princeton.edu) with questions/comments/suggestions

## Set up conda environment
1. Load anaconda module: ```module load anaconda3/2024.2```
2. Clone this repository: ```git clone https://github.com/wang-kevin3290/scaling-crl.git```
3. Create conda environment: ```CONDA_OVERRIDE_CUDA="12.0" conda create --name scaling-crl python=3.10 numpy==1.26.4 jax==0.4.23 "jaxlib==0.4.23=cuda120*" flax==0.7.4 -c conda-forge -c nvidia```
4. Activate conda environment: ```conda activate scaling-crl```
5. Install more dependencies: ```pip install tyro wandb==0.17.9 wandb_osh==1.2.2  brax==0.10.1 mediapy==1.2.2 scipy==1.12.0```

## Run training script
To run the code, you'll need a GPU. Please see the example provided in the slurm script `job.slurm`.

## Troubleshooting Potential Errors

**If you encounter the following error:**
```AttributeError: module 'mujoco.mjx' has no attribute 'ncon'```  

**Fix:**
1. Locate the brax contact.py file in your conda environment: 
   ```
   find ~/.conda/envs/scaling-crl -name contact.py
   ```
2. Open the file and replace it with the following code:

    ```python
    from typing import Optional
    from brax import math
    from brax.base import Contact
    from brax.base import System
    from brax.base import Transform
    import jax
    from jax import numpy as jp
    from mujoco import mjx

    def get(sys: System, x: Transform) -> Optional[Contact]:
        """Calculates contacts.
        Args:
            sys: system defining the kinematic tree and other properties
            x: link transforms in world frame
        Returns:
            Contact pytree
        """
        #NOTE: THIS WAS MODIFIED SINCE AFTER MUJOCO 3.1.5, mjx.ncon IS NOT AVAILABLE
        # ncon = mjx.ncon(sys)
        # if not ncon:
        #   return None
        data = mjx.make_data(sys)
        if data.ncon == 0:
            return None
        @jax.vmap
        def local_to_global(pos1, quat1, pos2, quat2):
            pos = pos1 + math.rotate(pos2, quat1)
            mat = math.quat_to_3x3(math.quat_mul(quat1, quat2))
            return pos, mat
        x = x.concatenate(Transform.zero((1,)))
        xpos = x.pos[sys.geom_bodyid - 1]
        xquat = x.rot[sys.geom_bodyid - 1]
        geom_xpos, geom_xmat = local_to_global(
            xpos, xquat, sys.geom_pos, sys.geom_quat
        )
        # pytype: disable=wrong-arg-types
        d = data.replace(geom_xpos=geom_xpos, geom_xmat=geom_xmat)
        d = mjx.collision(sys, d)
        # pytype: enable=wrong-arg-types
        c = d.contact
        elasticity = (sys.elasticity[c.geom1] + sys.elasticity[c.geom2]) * 0.5
        body1 = jp.array(sys.geom_bodyid)[c.geom1] - 1
        body2 = jp.array(sys.geom_bodyid)[c.geom2] - 1
        link_idx = (body1, body2)
        return Contact(elasticity=elasticity, link_idx=link_idx, **c.__dict__)
    ```
3. Save the file and rerun the training script.


**If you encounter the following error:** ```Error rendering final policy: unsupported operand type(s) for ==: 'ArrayImpl' and 'list'```  

**Fix:**
1. Locate the brax json.py file in your conda environment:
   ```
   find ~/.conda/envs/expl-env -name json.py | grep "/brax/io/json.py"
   ```
2. Open the file and change the if statement in line 159 to:
    ```python
    if (rgba == jp.array([0.5, 0.5, 0.5, 1.0])).all():
    ```
3. Save the file and rerun the training script.