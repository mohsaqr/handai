"""
Progress Display Component
UI for showing processing progress, metrics, and logs
"""

import streamlit as st
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class ProgressState:
    """State for progress display"""
    completed: int = 0
    total: int = 0
    success_count: int = 0
    error_count: int = 0
    retry_count: int = 0
    log_lines: List[str] = None
    error_lines: List[str] = None

    def __post_init__(self):
        if self.log_lines is None:
            self.log_lines = []
        if self.error_lines is None:
            self.error_lines = []


class ProgressDisplay:
    """Progress display widget with metrics, progress bar, and logs"""

    def __init__(self, title: str = "Processing Status", max_log_lines: int = 10):
        """
        Initialize progress display.

        Args:
            title: Section title
            max_log_lines: Maximum log lines to display
        """
        self.title = title
        self.max_log_lines = max_log_lines
        self.state = ProgressState()

        # Create placeholders
        st.subheader(self.title)
        self.status_metrics = st.empty()
        self.progress_bar = st.progress(0)
        self.log_placeholder = st.empty()
        self.error_expander = st.expander("Errors", expanded=False)

    def update(
        self,
        completed: int,
        total: int,
        success_count: int,
        error_count: int,
        retry_count: int,
        log_entry: str,
        is_error: bool = False
    ):
        """
        Update progress display.

        Args:
            completed: Number of completed items
            total: Total items
            success_count: Successful items
            error_count: Failed items
            retry_count: Total retries
            log_entry: New log entry to add
            is_error: Whether this entry is an error
        """
        self.state.completed = completed
        self.state.total = total
        self.state.success_count = success_count
        self.state.error_count = error_count
        self.state.retry_count = retry_count

        # Add log entry
        status = "✓" if not is_error else "✗"
        log_line = f"{status} {log_entry}"
        self.state.log_lines.insert(0, log_line)
        if len(self.state.log_lines) > self.max_log_lines:
            self.state.log_lines = self.state.log_lines[:self.max_log_lines]

        # Track errors separately
        if is_error:
            self.state.error_lines.insert(0, log_entry)
            if len(self.state.error_lines) > 50:
                self.state.error_lines = self.state.error_lines[:50]

        # Update UI
        self._render()

    def _render(self):
        """Render current state"""
        # Progress bar
        progress = self.state.completed / self.state.total if self.state.total > 0 else 0
        self.progress_bar.progress(progress)

        # Metrics
        self.status_metrics.markdown(
            f"**Progress:** {self.state.completed}/{self.state.total} | "
            f"✓ {self.state.success_count} | "
            f"✗ {self.state.error_count} | "
            f"↻ {self.state.retry_count} retries"
        )

        # Log
        log_text = "\n".join(self.state.log_lines)
        self.log_placeholder.code(log_text)

        # Errors
        if self.state.error_lines:
            with self.error_expander:
                error_text = "\n".join(self.state.error_lines[:20])
                st.text(error_text)

    def complete(self, message: str = None):
        """Mark progress as complete"""
        self.progress_bar.progress(1.0)
        if message:
            st.success(message)


def render_simple_progress(completed: int, total: int, status_text: str = ""):
    """
    Render a simple progress indicator.

    Args:
        completed: Completed items
        total: Total items
        status_text: Optional status text
    """
    progress = completed / total if total > 0 else 0
    st.progress(progress)
    if status_text:
        st.caption(status_text)


def render_metrics_row(
    success_count: int,
    error_count: int,
    retry_count: int,
    avg_latency: float = None,
    total_duration: float = None
):
    """
    Render a row of metrics.

    Args:
        success_count: Successful items
        error_count: Failed items
        retry_count: Total retries
        avg_latency: Average latency in seconds
        total_duration: Total duration in seconds
    """
    cols = st.columns(5 if avg_latency and total_duration else 3)

    with cols[0]:
        st.metric("Success", success_count)
    with cols[1]:
        st.metric("Errors", error_count)
    with cols[2]:
        st.metric("Retries", retry_count)

    if avg_latency is not None and len(cols) > 3:
        with cols[3]:
            st.metric("Avg Latency", f"{avg_latency:.2f}s")
    if total_duration is not None and len(cols) > 4:
        with cols[4]:
            st.metric("Duration", f"{total_duration:.1f}s")


def render_completion_summary(
    success_count: int,
    error_count: int,
    retry_count: int,
    avg_latency: float,
    total_duration: float
):
    """
    Render completion summary.

    Args:
        success_count: Successful items
        error_count: Failed items
        retry_count: Total retries
        avg_latency: Average latency
        total_duration: Total duration
    """
    total = success_count + error_count
    success_rate = (success_count / total * 100) if total > 0 else 0

    st.success(
        f"Complete! "
        f"✓ {success_count} | "
        f"✗ {error_count} | "
        f"↻ {retry_count} retries | "
        f"Avg: {avg_latency:.2f}s | "
        f"Total: {total_duration:.1f}s"
    )

    # Show warning if there were errors
    if error_count > 0:
        st.warning(f"Completed with {error_count} errors ({100 - success_rate:.1f}% failure rate)")
