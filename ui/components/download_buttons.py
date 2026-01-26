"""
Download Buttons Component
UI for exporting results in various formats
"""

import streamlit as st
import pandas as pd
import io
from typing import Optional


def render_download_buttons(
    df: pd.DataFrame,
    filename_prefix: str = "handai_results",
    key_prefix: str = "download"
):
    """
    Render download buttons for CSV, Excel, and JSON.

    Args:
        df: DataFrame to export
        filename_prefix: Prefix for downloaded filenames
        key_prefix: Prefix for widget keys
    """
    if df is None or df.empty:
        st.info("No data to download")
        return

    st.subheader("Download Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "CSV",
            csv,
            f"{filename_prefix}.csv",
            "text/csv",
            key=f"{key_prefix}_csv",
            use_container_width=True
        )

    with col2:
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        st.download_button(
            "Excel",
            excel_buffer,
            f"{filename_prefix}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{key_prefix}_excel",
            use_container_width=True
        )

    with col3:
        json_data = df.to_json(orient="records", indent=2)
        st.download_button(
            "JSON",
            json_data,
            f"{filename_prefix}.json",
            "application/json",
            key=f"{key_prefix}_json",
            use_container_width=True
        )


def render_download_buttons_compact(
    df: pd.DataFrame,
    filename_prefix: str = "handai_results",
    key_prefix: str = "dl_compact"
):
    """
    Render compact download buttons in a single row.

    Args:
        df: DataFrame to export
        filename_prefix: Prefix for downloaded filenames
        key_prefix: Prefix for widget keys
    """
    if df is None or df.empty:
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "CSV",
            csv,
            f"{filename_prefix}.csv",
            "text/csv",
            key=f"{key_prefix}_csv"
        )

    with col2:
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        st.download_button(
            "Excel",
            excel_buffer,
            f"{filename_prefix}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{key_prefix}_excel"
        )

    with col3:
        json_data = df.to_json(orient="records", indent=2)
        st.download_button(
            "JSON",
            json_data,
            f"{filename_prefix}.json",
            "application/json",
            key=f"{key_prefix}_json"
        )


def render_single_download(
    data: str,
    filename: str,
    label: str = "Download",
    mime_type: str = "text/plain",
    key: str = "single_download"
):
    """
    Render a single download button.

    Args:
        data: Data to download
        filename: Filename for download
        label: Button label
        mime_type: MIME type of the data
        key: Widget key
    """
    st.download_button(
        label,
        data,
        filename,
        mime_type,
        key=key
    )


def export_session_data(session_id: str, db) -> Optional[str]:
    """
    Export full session data as JSON.

    Args:
        session_id: Session ID to export
        db: Database instance

    Returns:
        JSON string of session data or None
    """
    import json

    session = db.get_session(session_id)
    if not session:
        return None

    runs = db.get_session_runs(session_id)
    logs = db.get_session_logs(session_id)

    export = {
        "session": {
            "id": session.session_id,
            "name": session.name,
            "mode": session.mode,
            "created_at": session.created_at,
            "settings": session.get_settings()
        },
        "runs": [],
        "logs": []
    }

    for run in runs:
        run_export = {
            "id": run.run_id,
            "type": run.run_type,
            "provider": run.provider,
            "model": run.model,
            "temperature": run.temperature,
            "max_tokens": run.max_tokens,
            "system_prompt": run.system_prompt,
            "schema": run.get_schema(),
            "variables": run.get_variables(),
            "status": run.status,
            "input_file": run.input_file,
            "settings": {
                "json_mode": bool(run.json_mode),
                "max_concurrency": run.max_concurrency,
                "auto_retry": bool(run.auto_retry),
                "max_retry_attempts": run.max_retry_attempts,
                "full_settings": run.get_run_settings()
            },
            "stats": {
                "input_rows": run.input_rows,
                "success_count": run.success_count,
                "error_count": run.error_count,
                "retry_count": run.retry_count,
                "avg_latency": run.avg_latency,
                "total_duration": run.total_duration
            },
            "started_at": run.started_at,
            "completed_at": run.completed_at,
            "results": []
        }

        results = db.get_run_results(run.run_id)
        for r in results:
            run_export["results"].append({
                "row_index": r.row_index,
                "input": r.input_json,
                "output": r.output,
                "status": r.status,
                "error_type": r.error_type,
                "error_message": r.error_message,
                "latency": r.latency
            })

        export["runs"].append(run_export)

    for log in logs:
        export["logs"].append({
            "level": log.level,
            "message": log.message,
            "details": log.get_details(),
            "timestamp": log.timestamp,
            "run_id": log.run_id
        })

    return json.dumps(export, indent=2)
