"""susgrad -- suspiciousness / gradient correlation experiments.

Public sub-packages:
    susgrad.models       -- shared model architectures (import once, reuse everywhere)
    susgrad.sbfl         -- suspiciousness computation (a_s/a_f/n_s/n_f; ochiai/tarantula/d*)
    susgrad.grads        -- per-neuron gradient capture
    susgrad.training     -- dataset prep, training (incl. per-epoch), evaluation
    susgrad.persistence  -- store/load models and grad/susp tensors
    susgrad.viz          -- bounded transforms + HTML rendering
    susgrad.registry     -- the single source of truth for model/dataset combinations
"""

__version__ = "1.0.0"
