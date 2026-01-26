"""
Result Inspector Component
UI for inspecting individual results
"""

import streamlit as st
import pandas as pd
from typing import Optional


def render_result_inspector(
    df: pd.DataFrame,
    output_column: str = "ai_output",
    key_prefix: str = "inspector"
):
    """
    Render result inspector widget.

    Args:
        df: DataFrame with results
        output_column: Name of the output column
        key_prefix: Prefix for widget keys
    """
    if df is None or df.empty:
        st.info("No results to inspect")
        return

    st.subheader("Result Inspector")

    col1, col2 = st.columns([1, 3])

    with col1:
        max_row = max(0, len(df) - 1)
        row_to_inspect = st.number_input(
            "Row",
            min_value=0,
            max_value=max_row,
            value=0,
            key=f"{key_prefix}_row"
        )

    with col2:
        inspect_mode = st.radio(
            "View",
            ["AI Output", "Full Row"],
            horizontal=True,
            key=f"{key_prefix}_mode"
        )

    if row_to_inspect < len(df):
        sel_row = df.iloc[row_to_inspect]

        if inspect_mode == "AI Output":
            content = sel_row.get(output_column, "")
            if content:
                # Try to detect if it's JSON
                try:
                    import json
                    parsed = json.loads(content)
                    st.json(parsed)
                except:
                    st.code(content, language="markdown")
            else:
                st.info("No output for this row")
        else:
            st.code(sel_row.to_json(indent=2), language="json")


def render_result_inspector_with_navigation(
    df: pd.DataFrame,
    output_column: str = "ai_output",
    key_prefix: str = "nav_inspector"
):
    """
    Render result inspector with prev/next navigation.

    Args:
        df: DataFrame with results
        output_column: Name of the output column
        key_prefix: Prefix for widget keys
    """
    if df is None or df.empty:
        st.info("No results to inspect")
        return

    st.subheader("Result Inspector")

    # Initialize current row in session state
    if f"{key_prefix}_current_row" not in st.session_state:
        st.session_state[f"{key_prefix}_current_row"] = 0

    current_row = st.session_state[f"{key_prefix}_current_row"]
    max_row = len(df) - 1

    # Navigation
    col1, col2, col3, col4 = st.columns([1, 1, 2, 2])

    with col1:
        if st.button("Prev", key=f"{key_prefix}_prev", disabled=current_row <= 0):
            st.session_state[f"{key_prefix}_current_row"] = max(0, current_row - 1)
            st.rerun()

    with col2:
        if st.button("Next", key=f"{key_prefix}_next", disabled=current_row >= max_row):
            st.session_state[f"{key_prefix}_current_row"] = min(max_row, current_row + 1)
            st.rerun()

    with col3:
        new_row = st.number_input(
            "Go to row",
            min_value=0,
            max_value=max_row,
            value=current_row,
            key=f"{key_prefix}_goto",
            label_visibility="collapsed"
        )
        if new_row != current_row:
            st.session_state[f"{key_prefix}_current_row"] = new_row
            st.rerun()

    with col4:
        st.caption(f"Row {current_row + 1} of {len(df)}")

    # Display row
    current_row = st.session_state[f"{key_prefix}_current_row"]
    if current_row < len(df):
        sel_row = df.iloc[current_row]

        # Show status if available
        if "status" in sel_row:
            status = sel_row.get("status", "")
            if status == "success":
                st.success("Success")
            elif status == "error":
                st.error("Error")
                if "error_message" in sel_row and sel_row.get("error_message"):
                    st.caption(f"Error: {sel_row.get('error_message')}")

        # Show output
        content = sel_row.get(output_column, "")
        if content:
            # Tabs for different views
            tab1, tab2, tab3 = st.tabs(["Output", "Input", "Full Row"])

            with tab1:
                try:
                    import json
                    parsed = json.loads(content)
                    st.json(parsed)
                except:
                    st.code(content, language="markdown")

            with tab2:
                if "input_json" in sel_row:
                    try:
                        import json
                        parsed = json.loads(sel_row.get("input_json", "{}"))
                        st.json(parsed)
                    except:
                        st.code(sel_row.get("input_json", ""), language="json")
                else:
                    # Show input columns
                    input_cols = [c for c in df.columns if c not in [output_column, "latency_s", "_latency_s", "_error"]]
                    st.json(sel_row[input_cols].to_dict())

            with tab3:
                st.code(sel_row.to_json(indent=2), language="json")
        else:
            st.info("No output for this row")


def render_error_summary(df: pd.DataFrame, error_column: str = "_error"):
    """
    Render summary of errors in results.

    Args:
        df: DataFrame with results
        error_column: Name of the error column
    """
    if error_column not in df.columns:
        return

    errors = df[df[error_column].notna()]
    if errors.empty:
        return

    with st.expander(f"Errors ({len(errors)} rows)", expanded=False):
        error_counts = errors[error_column].value_counts()

        st.write("**Error Summary:**")
        for error, count in error_counts.items():
            st.write(f"- {error}: {count} occurrences")

        st.divider()
        st.write("**Affected Rows:**")
        st.dataframe(
            errors[[error_column]].reset_index(),
            use_container_width=True,
            hide_index=True
        )
