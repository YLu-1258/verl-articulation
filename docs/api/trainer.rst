Trainer Interface
================================

Last updated: 06/08/2025 (API docstrings are auto-generated).

Trainers drive the training loop. Introducing new trainer classes in case of new training paradiam is encouraged.

.. autosummary::
   :nosignatures:

   verl_articulation.trainer.ppo.ray_trainer.RayPPOTrainer


Core APIs
~~~~~~~~~~~~~~~~~

.. autoclass:: verl_articulation.trainer.ppo.ray_trainer.RayPPOTrainer
   :members: __init__, init_workers, fit

.. automodule:: verl_articulation.utils.tokenizer
   :members: hf_tokenizer

.. automodule:: verl_articulation.trainer.ppo.core_algos
   :members: agg_loss, kl_penalty, compute_policy_loss, kl_penalty

.. automodule:: verl_articulation.trainer.ppo.reward
   :members: load_reward_manager, compute_reward, compute_reward_async

.. autoclass:: verl_articulation.workers.reward_manager.NaiveRewardManager

.. autoclass:: verl_articulation.workers.reward_manager.DAPORewardManager
