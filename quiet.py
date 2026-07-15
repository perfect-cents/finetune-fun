"""Import this FIRST (before transformers / unsloth) to quiet the noisy
deprecation logs and warnings those libraries emit on import.

    import quiet          # sets env vars + warning filters at import time
    ...                   # then import unsloth / transformers
    quiet.hush()          # call once after they're imported to lower log levels
"""

import os
import warnings

# Must be set before transformers is imported to take effect at import time.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.filterwarnings("ignore")


def hush():
    """Lower transformers/datasets logger verbosity (e.g. generation warnings)."""
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass
    try:
        import datasets
        datasets.utils.logging.set_verbosity_error()
    except Exception:
        pass
