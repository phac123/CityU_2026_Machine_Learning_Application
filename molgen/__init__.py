"""molgen - the three generative models of the course project.

Each method is a self-contained package with its own data loader, model,
metrics, distributed helpers, training entry point and evaluation entry point:

    molgen.geobfn   Bayesian flow networks (GeoBFN)
    molgen.edm      E(3)-equivariant diffusion
    molgen.llm      SMILES language model + ETKDG bridge

The packages share no code.  The top-level ``train.py`` / ``evaluate.py`` only
dispatch to them based on the ``method`` key of the Hydra config.
"""
