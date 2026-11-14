"""Guards command-line pipeline utilities from being used as Streamlit apps."""
from __future__ import annotations


def is_streamlit_context() -> bool:
    """Return whether this process is executing a Streamlit script run."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:
        return False
    return get_script_run_ctx(suppress_warning=True) is not None


def reject_streamlit_pipeline_entrypoint(script_name: str) -> bool:
    """Show a useful setup error instead of running an offline pipeline in Cloud."""
    if not is_streamlit_context():
        return False
    import streamlit as st

    st.error(
        f"{script_name} is an offline utility, not the dashboard entry point. "
        "Set the Streamlit app file to "
        "src/os_performance_prediction/dashboard/app.py."
    )
    return True
