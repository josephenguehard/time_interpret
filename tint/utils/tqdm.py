try:
    from tqdm import tqdm
    from tqdm.notebook import tqdm as tqdm_notebook
except ImportError:
    tqdm = None
    tqdm_notebook = None


def is_notebook_lab():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            # Jupyter notebook/lab or qtconsole
            return True
        elif shell == "TerminalInteractiveShell":
            # Terminal running IPython
            return False
        else:
            # Other type (?)
            return False
    except NameError:
        # Probably standard Python interpreter
        return False


def get_progress_bars():
    """
    Get which progress bar to use depending on usage.
    Returns:
        A tqdm pbar.
    """
    assert tqdm is not None, "tqdm is not installed"
    return tqdm_notebook if is_notebook_lab() else tqdm
