Utilities
============

Last updated: 05/19/2025 (API docstrings are auto-generated).

This section documents the utility functions and classes in the VERL library.

Python Functional Utilities
------------------------------

.. automodule:: verl_articulation.utils.py_functional
   :members: append_to_dict

File System Utilities
------------------------

.. automodule:: verl_articulation.utils.fs
   :members: copy_to_local

Tracking Utilities
---------------------

.. automodule:: verl_articulation.utils.tracking
   :members: Tracking

Metrics Utilities
---------------------

.. automodule::  verl_articulation.utils.metric
   :members: reduce_metrics

Checkpoint Management
------------------------

.. automodule:: verl_articulation.utils.checkpoint.checkpoint_manager
   :members: find_latest_ckpt_path

.. automodule:: verl_articulation.utils.checkpoint.fsdp_checkpoint_manager
   :members: FSDPCheckpointManager

Dataset Utilities
---------------------

.. automodule:: verl_articulation.utils.dataset.rl_dataset
   :members: RLHFDataset, collate_fn

Torch Functional Utilities
-----------------------------

.. automodule:: verl_articulation.utils.torch_functional
   :members: get_constant_schedule_with_warmup, masked_whiten, masked_mean, logprobs_from_logits

Sequence Length Balancing
----------------------------

.. automodule:: verl_articulation.utils.seqlen_balancing
   :members: get_reverse_idx, rearrange_micro_batches

Ulysses Utilities
--------------------

.. automodule:: verl_articulation.utils.ulysses
   :members: gather_outputs_and_unpad, ulysses_pad_and_slice_inputs

FSDP Utilities
------------------

.. automodule:: verl_articulation.utils.fsdp_utils
   :members: get_fsdp_wrap_policy, get_init_weight_context_manager, init_fn, load_fsdp_model_to_gpu, load_fsdp_optimizer, offload_fsdp_model_to_cpu, offload_fsdp_optimizer,

Debug Utilities
-------------------

.. automodule:: verl_articulation.utils.profiler
   :members: log_gpu_memory_usage, GPUMemoryLogger

