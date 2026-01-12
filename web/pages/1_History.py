"""
Handai History Page
Browse sessions, runs, and results
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
import io
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handai_db import get_db, RunStatus, ResultStatus, LogLevel

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Handai History",
    layout="wide",
    page_icon="📜"
)

# Initialize database
db = get_db()

# ==========================================
# HELPER FUNCTIONS (defined first)
# ==========================================

def show_run_details(run):
    """Display detailed information about a run"""
    st.subheader(f"Run Details: {run.run_id}")

    # Status badge
    status_colors = {
        "completed": "🟢",
        "running": "🔵",
        "failed": "🔴",
        "cancelled": "🟡"
    }
    status_icon = status_colors.get(run.status, "⚪")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Model Configuration**")
        st.write(f"- **Status:** {status_icon} {run.status}")
        st.write(f"- **Type:** {run.run_type}")
        st.write(f"- **Provider:** {run.provider}")
        st.write(f"- **Model:** {run.model}")
        st.write(f"- **Temperature:** {run.temperature}")
        st.write(f"- **Max Tokens:** {run.max_tokens}")

    with col2:
        st.write("**Run Settings**")
        json_mode = getattr(run, 'json_mode', False)
        max_concurrency = getattr(run, 'max_concurrency', 5)
        auto_retry = getattr(run, 'auto_retry', True)
        max_retry_attempts = getattr(run, 'max_retry_attempts', 3)
        st.write(f"- **JSON Mode:** {'Yes' if json_mode else 'No'}")
        st.write(f"- **Concurrency:** {max_concurrency}")
        st.write(f"- **Auto-Retry:** {'Yes' if auto_retry else 'No'}")
        st.write(f"- **Max Retries:** {max_retry_attempts}")
        st.write(f"- **Input File:** {run.input_file}")

    with col3:
        st.write("**Results**")
        st.write(f"- **Input Rows:** {run.input_rows}")
        st.write(f"- **Success:** {run.success_count}")
        st.write(f"- **Errors:** {run.error_count}")
        st.write(f"- **Retries:** {run.retry_count}")
        st.write(f"- **Avg Latency:** {run.avg_latency:.2f}s")
        st.write(f"- **Duration:** {run.total_duration:.1f}s")

    # System prompt
    with st.expander("System Prompt"):
        st.code(run.system_prompt or "N/A")

    # Full settings snapshot
    run_settings_json = getattr(run, 'run_settings_json', '{}')
    if run_settings_json and run_settings_json != '{}':
        with st.expander("Full Settings Snapshot"):
            try:
                settings = json.loads(run_settings_json)
                st.json(settings)
            except:
                st.code(run_settings_json)

    # Schema (if generation)
    if run.schema_json and run.schema_json != "{}":
        with st.expander("Schema"):
            st.json(json.loads(run.schema_json))

    # Results
    results = db.get_run_results(run.run_id)
    if results:
        with st.expander(f"Results ({len(results)} rows)"):
            result_data = []
            for r in results:
                result_data.append({
                    "Row": r.row_index,
                    "Status": r.status,
                    "Output": r.output[:50] + "..." if r.output and len(r.output) > 50 else r.output,
                    "Error Type": r.error_type or "-",
                    "Latency": f"{r.latency:.2f}s",
                    "Retry": r.retry_attempt
                })
            st.dataframe(pd.DataFrame(result_data), use_container_width=True)

            # Export results
            if st.button("📥 Export Results", key=f"export_results_{run.run_id}"):
                export_df = pd.DataFrame([{
                    "row_index": r.row_index,
                    "input": r.input_json,
                    "output": r.output,
                    "status": r.status,
                    "error_type": r.error_type,
                    "error_message": r.error_message,
                    "latency": r.latency,
                    "retry_attempt": r.retry_attempt
                } for r in results])

                csv = export_df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    f"results_{run.run_id}.csv",
                    "text/csv",
                    key=f"dl_results_{run.run_id}"
                )

    # Logs for this run
    logs = db.get_run_logs(run.run_id)
    if logs:
        with st.expander(f"Logs ({len(logs)} entries)"):
            for log in logs[:20]:  # Show first 20
                level_icon = {"ERROR": "❌", "WARNING": "⚠️", "INFO": "ℹ️", "DEBUG": "🔍"}.get(log.level, "📝")
                st.write(f"{level_icon} [{log.timestamp[:19]}] {log.message}")


def export_session(session_id: str) -> dict:
    """Export full session data as dictionary"""
    session = db.get_session(session_id)
    runs = db.get_session_runs(session_id)
    logs = db.get_session_logs(session_id)

    export = {
        "session": {
            "id": session.session_id,
            "name": session.name,
            "mode": session.mode,
            "created_at": session.created_at,
            "settings": json.loads(session.settings_json) if session.settings_json else {}
        },
        "runs": [],
        "logs": []
    }

    for run in runs:
        # Get extended settings safely (for backward compatibility)
        run_settings_json = getattr(run, 'run_settings_json', '{}')
        run_settings = {}
        try:
            run_settings = json.loads(run_settings_json) if run_settings_json else {}
        except:
            pass

        run_export = {
            "id": run.run_id,
            "type": run.run_type,
            "provider": run.provider,
            "model": run.model,
            "temperature": run.temperature,
            "max_tokens": run.max_tokens,
            "system_prompt": run.system_prompt,
            "schema": json.loads(run.schema_json) if run.schema_json else {},
            "variables": json.loads(run.variables_json) if run.variables_json else {},
            "status": run.status,
            "input_file": run.input_file,
            "settings": {
                "json_mode": bool(getattr(run, 'json_mode', False)),
                "max_concurrency": getattr(run, 'max_concurrency', 5),
                "auto_retry": bool(getattr(run, 'auto_retry', True)),
                "max_retry_attempts": getattr(run, 'max_retry_attempts', 3),
                "full_settings": run_settings
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
            "details": json.loads(log.details_json) if log.details_json else {},
            "timestamp": log.timestamp,
            "run_id": log.run_id
        })

    return export


# ==========================================
# MAIN PAGE
# ==========================================

st.title("📜 History & Logs")

# ==========================================
# SIDEBAR - FILTERS
# ==========================================
with st.sidebar:
    st.header("Filters")

    view_mode = st.radio(
        "View",
        ["Sessions", "All Runs", "Logs", "Statistics"],
        key="history_view_mode"
    )

    st.divider()

    if view_mode == "All Runs":
        status_filter = st.selectbox(
            "Status Filter",
            ["All", "completed", "failed", "running"],
            key="status_filter"
        )

    if view_mode == "Logs":
        level_filter = st.selectbox(
            "Level Filter",
            ["All", "ERROR", "WARNING", "INFO", "DEBUG"],
            key="level_filter"
        )

    limit = st.slider("Max Results", 10, 500, 100)

# ==========================================
# SESSIONS VIEW
# ==========================================
if view_mode == "Sessions":
    st.header("Sessions")

    sessions = db.get_all_sessions(limit=limit)

    if not sessions:
        st.info("No sessions yet. Start processing data to create your first session.")
    else:
        for session in sessions:
            stats = db.get_session_stats(session.session_id)
            runs = db.get_session_runs(session.session_id)

            with st.expander(
                f"📁 {session.name} | {session.mode} | {len(runs)} runs | "
                f"Created: {session.created_at[:16]}",
                expanded=False
            ):
                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    st.write("**Session Info**")
                    st.write(f"- **ID:** `{session.session_id}`")
                    st.write(f"- **Mode:** {session.mode}")
                    st.write(f"- **Created:** {session.created_at[:19]}")
                    st.write(f"- **Updated:** {session.updated_at[:19]}")

                with col2:
                    st.write("**Statistics**")
                    st.write(f"- **Total Runs:** {stats.get('total_runs', 0)}")
                    st.write(f"- **Completed:** {stats.get('completed_runs', 0)}")
                    st.write(f"- **Success:** {stats.get('total_success', 0)}")
                    st.write(f"- **Errors:** {stats.get('total_errors', 0)}")

                with col3:
                    # Rename session
                    new_name = st.text_input(
                        "Rename",
                        value=session.name,
                        key=f"rename_{session.session_id}"
                    )
                    if new_name != session.name:
                        if st.button("Save", key=f"save_rename_{session.session_id}"):
                            db.update_session_name(session.session_id, new_name)
                            st.rerun()

                    if st.button("🗑️ Delete", key=f"del_{session.session_id}", type="secondary"):
                        db.delete_session(session.session_id)
                        st.success("Session deleted")
                        st.rerun()

                # Show runs for this session
                if runs:
                    st.divider()
                    st.write("**Runs in this session:**")

                    run_data = []
                    for run in runs:
                        run_data.append({
                            "Run ID": run.run_id,
                            "Type": run.run_type,
                            "Provider": run.provider,
                            "Model": run.model,
                            "Status": run.status,
                            "Success": run.success_count,
                            "Errors": run.error_count,
                            "Avg Latency": f"{run.avg_latency:.2f}s",
                            "Started": run.started_at[:16]
                        })

                    st.dataframe(pd.DataFrame(run_data), use_container_width=True)

                    # Select run to view details
                    selected_run_id = st.selectbox(
                        "View Run Details",
                        [r.run_id for r in runs],
                        key=f"select_run_{session.session_id}"
                    )

                    if selected_run_id:
                        selected_run = next((r for r in runs if r.run_id == selected_run_id), None)
                        if selected_run:
                            st.divider()
                            show_run_details(selected_run)

                # Export session
                st.divider()
                if st.button("📥 Export Session", key=f"export_{session.session_id}"):
                    export_data = export_session(session.session_id)
                    st.download_button(
                        "Download JSON",
                        json.dumps(export_data, indent=2),
                        f"session_{session.session_id}.json",
                        "application/json",
                        key=f"dl_{session.session_id}"
                    )

# ==========================================
# ALL RUNS VIEW
# ==========================================
elif view_mode == "All Runs":
    st.header("All Runs")

    filter_status = None if status_filter == "All" else status_filter
    runs = db.get_all_runs(limit=limit, status_filter=filter_status)

    if not runs:
        st.info("No runs found.")
    else:
        run_data = []
        for run in runs:
            run_data.append({
                "Run ID": run.run_id,
                "Session": run.session_id,
                "Type": run.run_type,
                "Provider": run.provider,
                "Model": run.model,
                "Status": run.status,
                "Rows": run.input_rows,
                "Success": run.success_count,
                "Errors": run.error_count,
                "Retries": run.retry_count,
                "Avg Latency": f"{run.avg_latency:.2f}s",
                "Duration": f"{run.total_duration:.1f}s",
                "Started": run.started_at[:16]
            })

        df = pd.DataFrame(run_data)
        st.dataframe(df, use_container_width=True)

        # Select run for details
        st.divider()
        selected_run_id = st.selectbox("Select Run for Details", [r.run_id for r in runs])

        if selected_run_id:
            selected_run = next((r for r in runs if r.run_id == selected_run_id), None)
            if selected_run:
                show_run_details(selected_run)

# ==========================================
# LOGS VIEW
# ==========================================
elif view_mode == "Logs":
    st.header("System Logs")

    filter_level = None if level_filter == "All" else level_filter
    logs = db.get_recent_logs(limit=limit, level_filter=filter_level)

    if not logs:
        st.info("No logs found.")
    else:
        log_data = []
        for log in logs:
            log_data.append({
                "Timestamp": log.timestamp[:19],
                "Level": log.level,
                "Message": log.message[:100] + ("..." if len(log.message) > 100 else ""),
                "Run ID": log.run_id or "-",
                "Session": log.session_id or "-"
            })

        df = pd.DataFrame(log_data)
        st.dataframe(df, use_container_width=True)

        # Log detail viewer
        st.divider()
        max_log_idx = max(0, len(logs) - 1)
        log_index = st.number_input("View Log Details (index)", 0, max_log_idx, 0)

        if log_index < len(logs):
            selected_log = logs[log_index]
            st.write(f"**Level:** {selected_log.level}")
            st.write(f"**Timestamp:** {selected_log.timestamp}")
            st.write(f"**Message:** {selected_log.message}")

            if selected_log.details_json and selected_log.details_json != "{}":
                st.write("**Details:**")
                st.json(json.loads(selected_log.details_json))

# ==========================================
# STATISTICS VIEW
# ==========================================
elif view_mode == "Statistics":
    st.header("Statistics")

    # Global stats
    global_stats = db.get_global_stats()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sessions", global_stats.get('total_sessions', 0))
    with col2:
        st.metric("Total Runs", global_stats.get('total_runs', 0))
    with col3:
        st.metric("Rows Processed", global_stats.get('total_rows_processed', 0))
    with col4:
        total = (global_stats.get('total_success', 0) or 0) + (global_stats.get('total_errors', 0) or 0)
        success_rate = (global_stats.get('total_success', 0) or 0) / total * 100 if total > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")

    st.divider()

    # Provider stats
    st.subheader("By Provider")
    provider_stats = db.get_provider_stats()

    if provider_stats:
        provider_df = pd.DataFrame(provider_stats)
        st.dataframe(provider_df, use_container_width=True)

        # Simple bar chart
        if len(provider_stats) > 0:
            chart_data = pd.DataFrame({
                'Provider': [p['provider'] for p in provider_stats],
                'Runs': [p['runs'] for p in provider_stats]
            })
            st.bar_chart(chart_data.set_index('Provider'))
    else:
        st.info("No provider data available yet.")
