"""
Manual Coder Tool
Code qualitative data manually with clickable codes
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List

from .base import BaseTool, ToolConfig, ToolResult
from core.sample_data import get_sample_data, get_dataset_info

# Sessions directory
SESSIONS_DIR = Path(".manual_coder_sessions")


class ManualCoderTool(BaseTool):
    """Tool for manual coding of qualitative data with clickable codes"""

    id = "manual_coder"
    name = "Manual Coder"
    description = "Code qualitative data manually with clickable codes"
    icon = ":material/touch_app:"
    category = "Processing"

    # Sample codes for each dataset
    SAMPLE_CODES = {
        "product_reviews": ["Positive", "Negative", "Neutral", "Mixed", "Quality Issue", "Shipping Issue"],
        "healthcare_interviews": ["Burnout", "Resilience", "Team Support", "Resource Issue", "Leadership", "Work-Life Balance"],
        "support_tickets": ["Bug", "Feature Request", "Billing", "Account Issue", "Shipping", "Refund"],
        "learning_experience": ["Positive Experience", "Negative Experience", "Technical Issue", "Engagement", "Isolation", "Flexibility"],
        "exit_interviews": ["Compensation", "Career Growth", "Management", "Work-Life Balance", "Culture", "Relocation"],
        "mixed_feedback": ["Positive", "Negative", "Neutral", "Detailed", "Brief"],
    }

    # Sample highlight words for each code
    SAMPLE_HIGHLIGHTS = {
        "product_reviews": {
            "Positive": "love, amazing, great, excellent, fantastic, happy, best, perfect, recommend",
            "Negative": "terrible, worst, broke, waste, disappointed, bad, poor, hate, awful",
            "Neutral": "okay, fine, decent, average, nothing special",
        },
        "healthcare_interviews": {
            "Burnout": "exhausted, burnout, tired, overwhelmed, stress",
            "Resilience": "keeps me going, purpose, proud, survive",
            "Team Support": "team, colleagues, support, together, camaraderie",
        },
        "support_tickets": {
            "Bug": "crash, error, broken, not working, fails",
            "Billing": "charge, payment, invoice, refund, money",
            "Account Issue": "login, password, access, account, locked",
        },
        "learning_experience": {
            "Positive Experience": "loved, helpful, great, appreciated, engaging",
            "Negative Experience": "frustrating, difficult, struggled, overwhelmed",
            "Technical Issue": "technical, video, connection, system, crashed",
        },
        "exit_interviews": {
            "Compensation": "salary, pay, money, compensation, offer, financial",
            "Career Growth": "growth, progression, career, promotion, opportunity",
            "Management": "leadership, management, manager, boss",
        },
        "mixed_feedback": {
            "Positive": "love, great, excellent, happy, exceeded, recommend, amazing",
            "Negative": "terrible, frustrating, wrong, mistake, never, worst, painful",
            "Neutral": "okay, average, nothing special, meh, decent",
        },
    }

    # Color palette for code highlights (subtle pastels)
    CODE_COLORS = [
        "#FFF3BF",  # Soft yellow
        "#C3FAE8",  # Soft teal
        "#D0EBFF",  # Soft blue
        "#F3D9FA",  # Soft purple
        "#FFE8CC",  # Soft orange
        "#D3F9D8",  # Soft green
        "#FFE3E3",  # Soft red
        "#E5DBFF",  # Soft violet
    ]

    def _generate_session_name(self) -> str:
        """Generate auto session name based on timestamp"""
        now = datetime.now()
        return f"session_{now.strftime('%Y%m%d_%H%M%S')}"

    def _get_session_file(self, name: str) -> Path:
        """Get path for a session file"""
        return SESSIONS_DIR / f"{name}.json"

    def _save_session(self, name: str = None) -> str:
        """Save coding session to file. Returns session name."""
        try:
            SESSIONS_DIR.mkdir(exist_ok=True)
            if not name:
                name = st.session_state.get("mc_current_session", self._generate_session_name())

            save_data = {
                "name": name,
                "saved_at": datetime.now().isoformat(),
                "coding_data": st.session_state.get("mc_coding_data", {}),
                "current_row": st.session_state.get("mc_current_row", 0),
                "total_rows": len(st.session_state.get("mc_df", [])) if st.session_state.get("mc_df") is not None else 0,
                "codes": st.session_state.get("mc_codes", []),
                "text_col": st.session_state.get("mc_text_col"),
                "highlights": st.session_state.get("mc_highlights", {}),
                "coded_count": self._count_coded_rows(),
            }
            # Convert int keys to strings for JSON
            save_data["coding_data"] = {str(k): v for k, v in save_data["coding_data"].items()}

            with open(self._get_session_file(name), "w") as f:
                json.dump(save_data, f, indent=2)

            st.session_state["mc_current_session"] = name
            return name
        except Exception as e:
            return None

    def _save_progress(self):
        """Auto-save current session"""
        if st.session_state.get("mc_current_session"):
            self._save_session(st.session_state["mc_current_session"])

    def _load_session(self, name: str) -> bool:
        """Load a saved session. Returns True if loaded."""
        try:
            session_file = self._get_session_file(name)
            if session_file.exists():
                with open(session_file, "r") as f:
                    save_data = json.load(f)
                # Convert string keys back to int
                coding_data = {int(k): v for k, v in save_data.get("coding_data", {}).items()}
                st.session_state["mc_coding_data"] = coding_data
                st.session_state["mc_current_row"] = save_data.get("current_row", 0)
                st.session_state["mc_codes"] = save_data.get("codes", [])
                st.session_state["mc_text_col"] = save_data.get("text_col")
                st.session_state["mc_highlights"] = save_data.get("highlights", {})
                st.session_state["mc_current_session"] = name
                return True
        except Exception as e:
            pass
        return False

    def _list_sessions(self) -> List[Dict[str, Any]]:
        """List all saved sessions with metadata"""
        sessions = []
        try:
            if SESSIONS_DIR.exists():
                for f in sorted(SESSIONS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
                    try:
                        with open(f, "r") as file:
                            data = json.load(file)
                            sessions.append({
                                "name": f.stem,
                                "saved_at": data.get("saved_at", ""),
                                "coded_count": data.get("coded_count", 0),
                                "total_rows": data.get("total_rows", 0),
                                "current_row": data.get("current_row", 0),
                            })
                    except:
                        pass
        except:
            pass
        return sessions

    def _delete_session(self, name: str):
        """Delete a saved session"""
        try:
            session_file = self._get_session_file(name)
            if session_file.exists():
                session_file.unlink()
        except:
            pass

    def _init_session_state(self):
        """Initialize session state keys for the tool"""
        first_init = "mc_initialized" not in st.session_state

        if "mc_df" not in st.session_state:
            st.session_state["mc_df"] = None
        if "mc_codes" not in st.session_state:
            st.session_state["mc_codes"] = []
        if "mc_coding_data" not in st.session_state:
            st.session_state["mc_coding_data"] = {}
        if "mc_current_row" not in st.session_state:
            st.session_state["mc_current_row"] = 0
        if "mc_text_col" not in st.session_state:
            st.session_state["mc_text_col"] = None
        if "mc_auto_advance" not in st.session_state:
            st.session_state["mc_auto_advance"] = False
        if "mc_highlights" not in st.session_state:
            st.session_state["mc_highlights"] = {}  # {code: "word1, word2, ..."}
        if "mc_context_rows" not in st.session_state:
            st.session_state["mc_context_rows"] = 2  # rows before/after current
        if "mc_codebook" not in st.session_state:
            st.session_state["mc_codebook"] = None  # uploaded codebook dataframe
        if "mc_light_mode" not in st.session_state:
            st.session_state["mc_light_mode"] = True  # light mode for highlight visibility
        if "mc_horizontal_codes" not in st.session_state:
            st.session_state["mc_horizontal_codes"] = True  # horizontal code buttons layout (default)
        if "mc_autosave_enabled" not in st.session_state:
            st.session_state["mc_autosave_enabled"] = True  # autosave enabled by default
        if "mc_buttons_above" not in st.session_state:
            st.session_state["mc_buttons_above"] = False  # buttons below text by default
        if "mc_immersive_mode" not in st.session_state:
            st.session_state["mc_immersive_mode"] = False  # immersive coding mode
        if "mc_current_session" not in st.session_state:
            st.session_state["mc_current_session"] = None  # current session name

        # Auto-load most recent session on first init (page refresh)
        if first_init:
            st.session_state["mc_initialized"] = True
            sessions = self._list_sessions()
            if sessions:
                # Load the most recent session
                most_recent = sessions[0]["name"]
                if self._load_session(most_recent):
                    st.toast(f"Resumed: {most_recent}")

    def _add_code(self, row_idx: int, code: str):
        """Add a code to a specific row (allows duplicates)"""
        if row_idx not in st.session_state["mc_coding_data"]:
            st.session_state["mc_coding_data"][row_idx] = []

        st.session_state["mc_coding_data"][row_idx].append(code)
        self._save_progress()  # Auto-save

        # Auto-advance if enabled
        if st.session_state["mc_auto_advance"]:
            total_rows = len(st.session_state["mc_df"])
            if st.session_state["mc_current_row"] < total_rows - 1:
                st.session_state["mc_current_row"] += 1

    def _remove_code_at(self, row_idx: int, position: int):
        """Remove a code at a specific position"""
        if row_idx in st.session_state["mc_coding_data"]:
            codes = st.session_state["mc_coding_data"][row_idx]
            if 0 <= position < len(codes):
                codes.pop(position)
                st.session_state["mc_coding_data"][row_idx] = codes
                self._save_progress()  # Auto-save

    def _move_code_up(self, row_idx: int, position: int):
        """Move a code up in the list"""
        if row_idx in st.session_state["mc_coding_data"]:
            codes = st.session_state["mc_coding_data"][row_idx]
            if 0 < position < len(codes):
                codes[position], codes[position - 1] = codes[position - 1], codes[position]
                st.session_state["mc_coding_data"][row_idx] = codes
                self._save_progress()  # Auto-save

    def _move_code_down(self, row_idx: int, position: int):
        """Move a code down in the list"""
        if row_idx in st.session_state["mc_coding_data"]:
            codes = st.session_state["mc_coding_data"][row_idx]
            if 0 <= position < len(codes) - 1:
                codes[position], codes[position + 1] = codes[position + 1], codes[position]
                st.session_state["mc_coding_data"][row_idx] = codes
                self._save_progress()  # Auto-save

    def _get_code_color(self, code: str) -> str:
        """Get consistent color for a code"""
        codes = st.session_state.get("mc_codes", [])
        if code in codes:
            idx = codes.index(code) % len(self.CODE_COLORS)
            return self.CODE_COLORS[idx]
        return self.CODE_COLORS[0]

    def _highlight_text(self, text: str) -> str:
        """Apply word highlights to text based on defined patterns"""
        import re
        highlighted = str(text)
        highlights = st.session_state.get("mc_highlights", {})

        for code, words_str in highlights.items():
            if not words_str.strip():
                continue
            color = self._get_code_color(code)
            words = [w.strip() for w in words_str.split(",") if w.strip()]
            for word in words:
                # Case-insensitive word matching
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                highlighted = pattern.sub(
                    f'<mark style="background-color: {color}; padding: 2px 4px; border-radius: 3px;">{word}</mark>',
                    highlighted
                )
        return highlighted

    def _get_applied_codes(self, row_idx: int) -> List[str]:
        """Get list of applied codes for a row"""
        return st.session_state["mc_coding_data"].get(row_idx, [])

    def _count_coded_rows(self) -> int:
        """Count how many rows have at least one code"""
        return sum(1 for codes in st.session_state["mc_coding_data"].values() if codes)

    def render_config(self) -> ToolConfig:
        """Render manual coder configuration UI"""
        self._init_session_state()

        # Step 1: Load Data
        st.header("1. Load Data")

        col_upload, col_sample = st.columns([3, 1])
        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload Dataset",
                type=["csv", "xlsx", "xls"],
                key="mc_upload"
            )
        with col_sample:
            st.write("")
            use_sample = st.button(
                "Use Sample Data",
                help="Load sample data for testing",
                key="mc_sample_btn"
            )

        if use_sample:
            st.session_state["mc_use_sample"] = True
            # Reset coding data and codes when loading new data
            st.session_state["mc_coding_data"] = {}
            st.session_state["mc_current_row"] = 0
            st.session_state["mc_codes"] = []  # Reset to load sample codes
            st.session_state["mc_highlights"] = {}  # Reset highlights

        # Sample data selector
        if st.session_state.get("mc_use_sample"):
            sample_options = {
                "mixed_feedback": "Mixed Feedback (15 items, varying lengths)",
                "product_reviews": "Product Reviews (20 reviews with sentiment)",
                "healthcare_interviews": "Healthcare Interviews (15 worker experiences)",
                "support_tickets": "Support Tickets (20 customer issues)",
                "learning_experience": "Learning Experience (20 student responses)",
                "exit_interviews": "Exit Interviews (15 employee departures)",
            }
            selected_sample = st.selectbox(
                "Choose sample dataset",
                options=list(sample_options.keys()),
                format_func=lambda x: sample_options[x],
                key="mc_sample_choice"
            )

        # Load data
        df = None

        if uploaded_file:
            file_ext = uploaded_file.name.split(".")[-1].lower()
            try:
                if file_ext == "csv":
                    df = pd.read_csv(uploaded_file)
                elif file_ext in ["xlsx", "xls"]:
                    df = pd.read_excel(uploaded_file)
                # Reset coding data when loading new data
                if df is not None and not df.equals(st.session_state.get("mc_df")):
                    st.session_state["mc_coding_data"] = {}
                    st.session_state["mc_current_row"] = 0
            except Exception as e:
                return ToolConfig(
                    is_valid=False,
                    error_message=f"Error loading file: {str(e)}"
                )

        elif st.session_state.get("mc_use_sample"):
            selected = st.session_state.get("mc_sample_choice", "mixed_feedback")
            df = pd.DataFrame(get_sample_data(selected))
            info = get_dataset_info()[selected]

            # Only show toast once when sample changes
            if st.session_state.get("mc_last_sample") != selected:
                st.session_state["mc_last_sample"] = selected
                st.toast(f"Loaded: {info['name']} ({info['rows']} rows)")

                # Auto-select text column from dataset info
                if "text_column" in info:
                    st.session_state["mc_text_col"] = info["text_column"]

                # Load sample codes
                if selected in self.SAMPLE_CODES:
                    st.session_state["mc_codes"] = self.SAMPLE_CODES[selected]
                # Load sample highlights
                if selected in self.SAMPLE_HIGHLIGHTS:
                    st.session_state["mc_highlights"] = self.SAMPLE_HIGHLIGHTS[selected]

        if df is None:
            return ToolConfig(
                is_valid=False,
                error_message="Please upload a file or use sample data"
            )

        if df.empty:
            return ToolConfig(
                is_valid=False,
                error_message="The uploaded file is empty"
            )

        st.session_state["mc_df"] = df

        # Step 2: Define Codes
        st.header("2. Define Codes")

        code_method = st.radio(
            "How to define codes?",
            ["Enter manually", "Upload codebook"],
            horizontal=True,
            key="mc_code_method"
        )

        codes = []
        codebook_df = None

        if code_method == "Upload codebook":
            codebook_file = st.file_uploader(
                "Upload codebook (CSV/Excel) - first column = codes",
                type=["csv", "xlsx", "xls"],
                key="mc_codebook_upload"
            )

            if codebook_file:
                try:
                    ext = codebook_file.name.split(".")[-1].lower()
                    if ext == "csv":
                        codebook_df = pd.read_csv(codebook_file)
                    else:
                        codebook_df = pd.read_excel(codebook_file)

                    if not codebook_df.empty:
                        # First column is always codes
                        code_col = codebook_df.columns[0]
                        codes = codebook_df[code_col].dropna().astype(str).tolist()
                        st.session_state["mc_codes"] = codes
                        st.session_state["mc_codebook"] = codebook_df

                        st.toast(f"Loaded {len(codes)} codes from codebook")

                        # Display codebook reference
                        with st.expander("📖 Codebook Reference", expanded=False):
                            other_cols = [c for c in codebook_df.columns if c != code_col]

                            for idx, row in codebook_df.iterrows():
                                code = str(row[code_col])
                                color = self.CODE_COLORS[idx % len(self.CODE_COLORS)]

                                # Code header
                                st.markdown(
                                    f'<div style="background-color: {color}; padding: 8px 12px; '
                                    f'border-radius: 5px; margin: 5px 0; font-weight: bold;">'
                                    f'{code}</div>',
                                    unsafe_allow_html=True
                                )

                                # Other columns as details
                                if other_cols:
                                    details = []
                                    for col in other_cols:
                                        val = row[col]
                                        if pd.notna(val) and str(val).strip():
                                            details.append(f"**{col}:** {val}")
                                    if details:
                                        st.markdown("  \n".join(details))
                                st.markdown("")

                except Exception as e:
                    st.error(f"Error loading codebook: {str(e)}")
            else:
                # Check if codebook was previously loaded
                if st.session_state.get("mc_codebook") is not None:
                    codebook_df = st.session_state["mc_codebook"]
                    code_col = codebook_df.columns[0]
                    codes = codebook_df[code_col].dropna().astype(str).tolist()

                    with st.expander("📖 Codebook Reference", expanded=False):
                        other_cols = [c for c in codebook_df.columns if c != code_col]

                        for idx, row in codebook_df.iterrows():
                            code = str(row[code_col])
                            color = self.CODE_COLORS[idx % len(self.CODE_COLORS)]

                            st.markdown(
                                f'<div style="background-color: {color}; padding: 8px 12px; '
                                f'border-radius: 5px; margin: 5px 0; font-weight: bold;">'
                                f'{code}</div>',
                                unsafe_allow_html=True
                            )

                            if other_cols:
                                details = []
                                for col in other_cols:
                                    val = row[col]
                                    if pd.notna(val) and str(val).strip():
                                        details.append(f"**{col}:** {val}")
                                if details:
                                    st.markdown("  \n".join(details))
                            st.markdown("")
                else:
                    st.info("Upload a codebook file with codes in the first column. Additional columns (meaning, example, etc.) will be shown as reference.")

        else:  # Enter manually
            codes_input = st.text_area(
                "Enter your codes (one per line)",
                value="\n".join(st.session_state.get("mc_codes", [])),
                height=150,
                help="Enter the codes you want to apply to the data, one code per line",
                key="mc_codes_input"
            )
            codes = [c.strip() for c in codes_input.strip().split("\n") if c.strip()]
            st.session_state["mc_codes"] = codes

        if not codes:
            return ToolConfig(
                is_valid=False,
                error_message="Please enter at least one code"
            )

        st.caption(f"{len(codes)} code(s) defined")

        # Step 3: Select Text Column (collapsible once set)
        text_columns = df.columns.tolist()
        default_col = st.session_state.get("mc_text_col")
        if default_col not in text_columns:
            default_col = text_columns[0] if text_columns else None

        # Auto-set if only one text-like column or already set
        if default_col and default_col in text_columns:
            text_col = default_col
            with st.expander(f"Text Column: **{text_col}** (click to change)"):
                text_col = st.selectbox(
                    "Column to display for coding",
                    options=text_columns,
                    index=text_columns.index(default_col),
                    key="mc_text_col_select"
                )
        else:
            st.header("3. Select Text Column")
            text_col = st.selectbox(
                "Column to display for coding",
                options=text_columns,
                index=0,
                key="mc_text_col_select"
            )
        st.session_state["mc_text_col"] = text_col

        # Preview data (collapsed if column already selected)
        if default_col and default_col in text_columns:
            with st.expander("Preview data"):
                st.dataframe(df.head(), height=150)
        else:
            st.dataframe(df.head(), height=150)

        return ToolConfig(
            is_valid=True,
            config_data={
                "df": df,
                "codes": codes,
                "text_col": text_col
            }
        )

    def _render_immersive_mode(self, df, codes, text_col, total_rows):
        """Render full-page immersive coding interface"""
        current_row = st.session_state["mc_current_row"]
        light_mode = st.session_state.get("mc_light_mode", True)
        context_rows = st.session_state.get("mc_context_rows", 2)
        auto_advance = st.session_state.get("mc_auto_advance", False)

        # Header row
        head_col1, head_col2, head_col3, head_col4 = st.columns([2, 1, 1, 1])
        with head_col1:
            coded_count = self._count_coded_rows()
            progress_pct = coded_count / total_rows if total_rows > 0 else 0
            st.markdown(f"**Row {current_row + 1}/{total_rows}** | {coded_count} coded ({progress_pct:.0%})")
        with head_col2:
            if st.button("Save", key="mc_imm_save", use_container_width=True):
                if not st.session_state.get("mc_current_session"):
                    st.session_state["mc_current_session"] = self._generate_session_name()
                self._save_session()
                st.toast("Saved!")
        with head_col3:
            if st.button("Options", key="mc_imm_options", use_container_width=True):
                st.session_state["mc_imm_show_options"] = not st.session_state.get("mc_imm_show_options", False)
        with head_col4:
            if st.button("Exit", key="mc_imm_exit", use_container_width=True):
                st.session_state["mc_immersive_mode"] = False
                st.rerun()

        # Options panel (collapsible)
        if st.session_state.get("mc_imm_show_options"):
            opt1, opt2, opt3 = st.columns(3)
            with opt1:
                new_light = st.toggle("Light mode", value=light_mode, key="mc_imm_light")
                if new_light != light_mode:
                    st.session_state["mc_light_mode"] = new_light
            with opt2:
                new_ctx = st.number_input("Context rows", min_value=0, max_value=5, value=context_rows, key="mc_imm_ctx")
                if new_ctx != context_rows:
                    st.session_state["mc_context_rows"] = new_ctx
            with opt3:
                new_auto = st.toggle("Auto-advance", value=auto_advance, key="mc_imm_auto")
                if new_auto != auto_advance:
                    st.session_state["mc_auto_advance"] = new_auto

        # Colors
        if light_mode:
            current_bg = "#FFFEF5"
            current_text = "#1a1a1a"
            context_bg = "#F8F9FA"
            context_text = "#555"
        else:
            current_bg = "#1a1a2e"
            current_text = "#eee"
            context_bg = "#2a2a3e"
            context_text = "#bbb"

        # Build text display with context
        start_row = max(0, current_row - context_rows)
        end_row = min(total_rows, current_row + context_rows + 1)

        text_html = []
        for row_idx in range(start_row, end_row):
            is_current = row_idx == current_row
            text_content = str(df.iloc[row_idx][text_col])
            highlighted_text = self._highlight_text(text_content)
            row_codes = self._get_applied_codes(row_idx)

            # Codes badges for this row
            badges = ""
            if row_codes:
                badges = " ".join([
                    f'<span style="background-color: {self._get_code_color(c)}; '
                    f'padding: 1px 6px; border-radius: 3px; font-size: 0.85em;">{c}</span>'
                    for c in row_codes
                ])

            if is_current:
                text_html.append(
                    f'<div style="background-color: {current_bg}; color: {current_text}; '
                    f'padding: 12px 15px; border-radius: 8px; border-left: 4px solid #4CAF50; '
                    f'margin: 4px 0; font-size: 1.05em;">'
                    f'<strong>► Row {row_idx + 1}</strong> {badges}<br/>'
                    f'{highlighted_text}</div>'
                )
            else:
                text_html.append(
                    f'<div style="background-color: {context_bg}; color: {context_text}; '
                    f'padding: 8px 12px; border-radius: 5px; margin: 2px 0; '
                    f'font-size: 0.9em; opacity: 0.85;">'
                    f'<span style="color: #888;">Row {row_idx + 1}</span> {badges}<br/>'
                    f'{highlighted_text}</div>'
                )

        # Fixed height text container
        st.markdown(
            f'<div style="min-height: 200px; max-height: 350px; overflow-y: auto; '
            f'padding: 5px; margin: 5px 0;">{"".join(text_html)}</div>',
            unsafe_allow_html=True
        )

        # Code buttons
        num_codes = len(codes)
        if num_codes > 0:
            code_cols = st.columns(num_codes)
            for i, code in enumerate(codes):
                color = self._get_code_color(code)
                with code_cols[i]:
                    st.markdown(f'<div style="background:{color}; height:4px; border-radius:2px; margin-bottom:2px;"></div>', unsafe_allow_html=True)
                    if st.button(code, key=f"mc_imm_code_{current_row}_{code}", use_container_width=True):
                        self._add_code(current_row, code)
                        st.rerun()

        # Navigation
        is_disabled_prev = current_row <= 0
        is_disabled_next = current_row >= total_rows - 1

        nav1, nav2, nav3, nav4 = st.columns([1, 1, 1, 1])
        with nav1:
            if st.button("◀◀", disabled=is_disabled_prev, key="mc_imm_prev5", use_container_width=True):
                st.session_state["mc_current_row"] = max(0, current_row - 5)
                st.rerun()
        with nav2:
            if st.button("◀ Prev", disabled=is_disabled_prev, key="mc_imm_prev", use_container_width=True):
                st.session_state["mc_current_row"] = max(0, current_row - 1)
                st.rerun()
        with nav3:
            if st.button("Next ▶", disabled=is_disabled_next, key="mc_imm_next", type="primary", use_container_width=True):
                st.session_state["mc_current_row"] = min(total_rows - 1, current_row + 1)
                self._save_progress()
                st.rerun()
        with nav4:
            if st.button("▶▶", disabled=is_disabled_next, key="mc_imm_next5", use_container_width=True):
                st.session_state["mc_current_row"] = min(total_rows - 1, current_row + 5)
                st.rerun()

        # Applied codes section - fixed height container to prevent jumping
        applied_codes = self._get_applied_codes(current_row)
        st.markdown('<div style="min-height: 50px;">', unsafe_allow_html=True)
        if applied_codes:
            app_cols = st.columns(min(len(applied_codes), 6))  # Max 6 columns
            for i, code in enumerate(applied_codes[:6]):  # Show max 6
                color = self._get_code_color(code)
                with app_cols[i]:
                    st.markdown(f'<span style="background-color: {color}; padding: 2px 6px; border-radius: 3px; font-size: 0.85em;">{code}</span>', unsafe_allow_html=True)
                    if st.button("×", key=f"mc_imm_rm_{current_row}_{i}"):
                        self._remove_code_at(current_row, i)
                        st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Progress bar
        coded_count = self._count_coded_rows()
        progress = coded_count / total_rows if total_rows > 0 else 0
        st.progress(progress)

    async def execute(
        self,
        config: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> ToolResult:
        """
        Execute is not used for manual coding - all work happens in render_results.
        This just passes through the config data.
        """
        return ToolResult(
            success=True,
            data=config["df"],
            stats={
                "total_rows": len(config["df"]),
                "codes_defined": len(config["codes"])
            }
        )

    def render_results(self, result: ToolResult):
        """Render the manual coding interface"""
        from ui.components.download_buttons import render_download_buttons

        if not result.success:
            st.error(f"Error: {result.error_message}")
            return

        df = st.session_state["mc_df"]
        codes = st.session_state["mc_codes"]
        text_col = st.session_state["mc_text_col"]
        total_rows = len(df)
        immersive_mode = st.session_state.get("mc_immersive_mode", False)

        # Session management bar (always visible)
        session_col1, session_col2, session_col3, session_col4 = st.columns([2, 1, 1, 1])
        with session_col1:
            current_session = st.session_state.get("mc_current_session")
            coded_count = self._count_coded_rows()
            if current_session:
                st.markdown(f"**Session:** `{current_session}` ({coded_count}/{total_rows} coded)")
            else:
                st.markdown(f"**New Session** ({coded_count}/{total_rows} coded)")
        with session_col2:
            if st.button("Save Session", key="mc_save_session", use_container_width=True):
                if not st.session_state.get("mc_current_session"):
                    st.session_state["mc_current_session"] = self._generate_session_name()
                name = self._save_session()
                if name:
                    st.toast(f"Saved: {name}")
        with session_col3:
            if st.button("Load Session", key="mc_show_load", use_container_width=True):
                st.session_state["mc_show_sessions"] = not st.session_state.get("mc_show_sessions", False)
        with session_col4:
            if st.button("Immersive Mode", key="mc_open_immersive", type="primary", use_container_width=True):
                st.session_state["mc_immersive_mode"] = True

        # Session browser popup
        if st.session_state.get("mc_show_sessions"):
            sessions = self._list_sessions()
            if sessions:
                st.markdown("---")
                st.markdown("##### Saved Sessions")
                for sess in sessions[:10]:  # Show last 10
                    s_col1, s_col2, s_col3 = st.columns([3, 1, 1])
                    with s_col1:
                        saved_time = sess.get("saved_at", "")[:16].replace("T", " ")
                        st.markdown(f"`{sess['name']}` - {sess['coded_count']}/{sess['total_rows']} coded ({saved_time})")
                    with s_col2:
                        if st.button("Load", key=f"mc_load_{sess['name']}", use_container_width=True):
                            if self._load_session(sess['name']):
                                st.session_state["mc_show_sessions"] = False
                                st.toast(f"Loaded: {sess['name']}")
                                st.rerun()
                    with s_col3:
                        if st.button("Delete", key=f"mc_del_{sess['name']}", use_container_width=True):
                            self._delete_session(sess['name'])
                            st.rerun()
                st.markdown("---")
            else:
                st.info("No saved sessions found")
                st.session_state["mc_show_sessions"] = False

        # Immersive mode - full page takeover
        if immersive_mode:
            self._render_immersive_mode(df, codes, text_col, total_rows)
            return  # Don't show regular interface

        # Word highlighter configuration (outside fragment)
        with st.expander("Word Highlighter - Define words to highlight for each code"):
            for code in codes:
                color = self._get_code_color(code)
                col_label, col_input = st.columns([1, 4])
                with col_label:
                    st.markdown(
                        f'<div style="background-color: {color}; padding: 6px 10px; '
                        f'border-radius: 5px; font-weight: bold; margin-top: 5px;">{code}</div>',
                        unsafe_allow_html=True
                    )
                with col_input:
                    words = st.text_input(
                        f"Words for {code}",
                        value=st.session_state.get("mc_highlights", {}).get(code, ""),
                        key=f"mc_hl_{code}",
                        placeholder="yes, agree, ok",
                        label_visibility="collapsed"
                    )
                    if "mc_highlights" not in st.session_state:
                        st.session_state["mc_highlights"] = {}
                    st.session_state["mc_highlights"][code] = words

        # Define the coding interface as a fragment for fast updates
        @st.fragment
        def coding_interface():
            current_row = st.session_state["mc_current_row"]
            context_rows = st.session_state.get("mc_context_rows", 2)
            light_mode = st.session_state.get("mc_light_mode", True)

            # All options in one compact row
            opt_col1, opt_col2, opt_col3, opt_col4, opt_col5 = st.columns([1, 1, 1, 1, 1])
            with opt_col1:
                new_light_mode = st.toggle("Light mode", value=light_mode, key="mc_theme_toggle")
                if new_light_mode != light_mode:
                    st.session_state["mc_light_mode"] = new_light_mode
                    st.rerun()
            with opt_col2:
                old_horiz = st.session_state.get("mc_horizontal_codes", False)
                new_horiz = st.toggle("Horizontal codes", value=old_horiz, key="mc_horiz_top")
                if new_horiz != old_horiz:
                    st.session_state["mc_horizontal_codes"] = new_horiz
                    st.rerun()
            with opt_col3:
                old_above = st.session_state.get("mc_buttons_above", False)
                new_above = st.toggle("Buttons above text", value=old_above, key="mc_above_toggle")
                if new_above != old_above:
                    st.session_state["mc_buttons_above"] = new_above
                    st.rerun()
            with opt_col4:
                auto_adv = st.toggle("Auto-advance", value=st.session_state.get("mc_auto_advance", False), key="mc_auto")
                st.session_state["mc_auto_advance"] = auto_adv
            with opt_col5:
                ctx = st.number_input("Context", min_value=0, max_value=5,
                                     value=st.session_state.get("mc_context_rows", 2), key="mc_ctx")
                st.session_state["mc_context_rows"] = ctx

            # Define colors based on mode
            if light_mode:
                current_bg = "#FFFEF5"
                current_text = "#1a1a1a"
                context_bg = "#F8F9FA"
                context_text = "#333"
                row_label_color = "#666"
            else:
                current_bg = "#1a1a2e"
                current_text = "#eee"
                context_bg = "rgba(128,128,128,0.2)"
                context_text = "#ddd"
                row_label_color = "#888"

            # Check layout mode
            horizontal_mode = st.session_state.get("mc_horizontal_codes", False)
            buttons_above = st.session_state.get("mc_buttons_above", False)

            # Helper function to render text display
            def render_text_display():
                # Calculate visible range
                start_row = max(0, current_row - context_rows)
                end_row = min(total_rows, current_row + context_rows + 1)

                text_container_html = []
                for row_idx in range(start_row, end_row):
                    is_current = row_idx == current_row
                    text_content = str(df.iloc[row_idx][text_col])
                    highlighted_text = self._highlight_text(text_content)

                    # Show applied codes for this row
                    row_codes = self._get_applied_codes(row_idx)
                    codes_badge = ""
                    if row_codes:
                        codes_badge = " ".join([
                            f'<span style="background-color: {self._get_code_color(c)}; '
                            f'padding: 1px 4px; border-radius: 3px; font-size: 0.8em; margin-left: 4px;">{c}</span>'
                            for c in row_codes
                        ])

                    if is_current:
                        text_container_html.append(
                            f'<div style="background-color: {current_bg}; color: {current_text}; '
                            f'padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50; '
                            f'margin: 5px 0; font-size: 1.1em;">'
                            f'<strong>► Row {row_idx + 1} of {total_rows}</strong>{codes_badge}<br/>'
                            f'{highlighted_text}</div>'
                        )
                    else:
                        text_container_html.append(
                            f'<div style="background-color: {context_bg}; color: {context_text}; '
                            f'padding: 10px; border-radius: 5px; margin: 3px 0; '
                            f'opacity: 0.85; font-size: 0.95em;">'
                            f'<span style="color: {row_label_color};">Row {row_idx + 1}</span>{codes_badge}<br/>'
                            f'{highlighted_text}</div>'
                        )

                # Render in container with min-height (expands for long text)
                st.markdown(
                    f'<div style="min-height: 300px; padding-right: 10px;">{"".join(text_container_html)}</div>',
                    unsafe_allow_html=True
                )

            # Helper function to render horizontal controls (code buttons, Next, navigation)
            def render_horizontal_controls():
                is_disabled = current_row >= total_rows - 1

                # Code buttons in columns with color indicators
                num_codes = len(codes)
                if num_codes > 0:
                    code_cols = st.columns(num_codes)
                    for i, code in enumerate(codes):
                        color = self._get_code_color(code)
                        with code_cols[i]:
                            # Color bar above button
                            st.markdown(f'<div style="background:{color}; height:4px; border-radius:2px; margin-bottom:2px;"></div>', unsafe_allow_html=True)
                            st.button(
                                code,
                                key=f"mc_code_{current_row}_{code}",
                                type="secondary",
                                use_container_width=True,
                                on_click=self._add_code,
                                args=(current_row, code)
                            )

                # Next button below codes
                if st.button("Next ▶", key="mc_next_horiz", type="primary", use_container_width=True, disabled=is_disabled):
                    st.session_state["mc_current_row"] = min(total_rows - 1, current_row + 1)
                    # Autosave on Next
                    if st.session_state.get("mc_autosave_enabled", True):
                        self._save_progress()
                    st.rerun()

                # Navigation buttons after Next
                def go_prev():
                    st.session_state["mc_current_row"] = max(0, st.session_state["mc_current_row"] - 1)

                def go_next():
                    st.session_state["mc_current_row"] = min(total_rows - 1, st.session_state["mc_current_row"] + 1)

                nav1, nav2, nav3, nav4 = st.columns([1, 1, 1, 1])
                with nav1:
                    st.button("◀◀", disabled=current_row <= 0, key="mc_prev5_h",
                              on_click=lambda: st.session_state.update({"mc_current_row": max(0, current_row - 5)}))
                with nav2:
                    st.button("◀ Prev", disabled=current_row <= 0, key="mc_prev_h", on_click=go_prev)
                with nav3:
                    st.button("Next ▶", disabled=current_row >= total_rows - 1, key="mc_next_h", on_click=go_next)
                with nav4:
                    st.button("▶▶", disabled=current_row >= total_rows - 1, key="mc_next5_h",
                              on_click=lambda: st.session_state.update({"mc_current_row": min(total_rows - 1, current_row + 5)}))

            # Helper function to render Applied codes and progress (always below text)
            def render_applied_and_progress():
                applied_codes = self._get_applied_codes(current_row)

                # Applied codes with reordering
                if applied_codes:
                    st.markdown("##### Applied")
                    for i, code in enumerate(applied_codes):
                        color = self._get_code_color(code)
                        col_up, col_down, col_code, col_remove = st.columns([1, 1, 10, 1])
                        with col_up:
                            st.button("▲", key=f"mc_up_{current_row}_{i}", disabled=i == 0,
                                      on_click=self._move_code_up, args=(current_row, i))
                        with col_down:
                            st.button("▼", key=f"mc_down_{current_row}_{i}", disabled=i == len(applied_codes) - 1,
                                      on_click=self._move_code_down, args=(current_row, i))
                        with col_code:
                            st.markdown(
                                f'<span style="background-color: {color}; padding: 2px 6px; '
                                f'border-radius: 3px;">{i + 1}. {code}</span>',
                                unsafe_allow_html=True
                            )
                        with col_remove:
                            st.button("×", key=f"mc_remove_{current_row}_{i}",
                                      on_click=self._remove_code_at, args=(current_row, i))

                # Progress
                coded_count = self._count_coded_rows()
                progress = coded_count / total_rows if total_rows > 0 else 0
                st.caption(f"**{coded_count}/{total_rows}** coded")
                st.progress(progress)

            # Main layout depends on mode
            if horizontal_mode:
                # Horizontal mode: buttons above or below text, Applied always at bottom
                if buttons_above:
                    render_horizontal_controls()
                    render_text_display()
                else:
                    render_text_display()
                    render_horizontal_controls()
                # Applied codes always below text
                render_applied_and_progress()

            else:
                # Vertical layout: text on left, codes on right
                col_text, col_codes = st.columns([5, 1])
                with col_text:
                    render_text_display()
                with col_codes:
                    st.markdown("##### Codes")

                    # Show codebook reference if available
                    codebook_df = st.session_state.get("mc_codebook")
                    if codebook_df is not None:
                        with st.expander("📖 Reference", expanded=False):
                            code_col = codebook_df.columns[0]
                            other_cols = [c for c in codebook_df.columns if c != code_col]
                            for idx, row in codebook_df.iterrows():
                                code = str(row[code_col])
                                color = self._get_code_color(code)
                                st.markdown(f'<span style="background-color: {color}; padding: 2px 6px; border-radius: 3px; font-weight: bold; font-size: 0.9em;">{code}</span>', unsafe_allow_html=True)
                                if other_cols:
                                    for col in other_cols:
                                        val = row[col]
                                        if pd.notna(val) and str(val).strip():
                                            st.caption(f"{col}: {val}")

                    applied_codes = self._get_applied_codes(current_row)

                    # Big Next button at TOP
                    is_disabled = current_row >= total_rows - 1
                    if st.button("Next ▶", key="mc_next_top", type="primary", use_container_width=True, disabled=is_disabled):
                        st.session_state["mc_current_row"] = min(total_rows - 1, current_row + 1)
                        if st.session_state.get("mc_autosave_enabled", True):
                            self._save_progress()
                        st.rerun()

                    # Code buttons with color indicators
                    for i, code in enumerate(codes):
                        color = self._get_code_color(code)
                        st.markdown(f'<div style="background:{color}; height:4px; border-radius:2px; margin-bottom:2px;"></div>', unsafe_allow_html=True)
                        st.button(
                            code,
                            key=f"mc_code_{current_row}_{code}",
                            type="secondary",
                            use_container_width=True,
                            on_click=self._add_code,
                            args=(current_row, code)
                        )

                    # Big Next button at BOTTOM
                    if st.button("Next ▶", key="mc_next_bottom", type="primary", use_container_width=True, disabled=is_disabled):
                        st.session_state["mc_current_row"] = min(total_rows - 1, current_row + 1)
                        if st.session_state.get("mc_autosave_enabled", True):
                            self._save_progress()
                        st.rerun()

                    # Navigation buttons
                    def go_prev_v():
                        st.session_state["mc_current_row"] = max(0, st.session_state["mc_current_row"] - 1)

                    def go_next_v():
                        st.session_state["mc_current_row"] = min(total_rows - 1, st.session_state["mc_current_row"] + 1)

                    nav1, nav2, nav3, nav4 = st.columns([1, 1, 1, 1])
                    with nav1:
                        st.button("◀◀", disabled=current_row <= 0, key="mc_prev5_v",
                                  on_click=lambda: st.session_state.update({"mc_current_row": max(0, current_row - 5)}))
                    with nav2:
                        st.button("◀ Prev", disabled=current_row <= 0, key="mc_prev_v", on_click=go_prev_v)
                    with nav3:
                        st.button("Next ▶", disabled=current_row >= total_rows - 1, key="mc_next_v", on_click=go_next_v)
                    with nav4:
                        st.button("▶▶", disabled=current_row >= total_rows - 1, key="mc_next5_v",
                                  on_click=lambda: st.session_state.update({"mc_current_row": min(total_rows - 1, current_row + 5)}))

                    # Show applied codes in order with reordering
                    if applied_codes:
                        st.markdown("---")
                        st.markdown("##### Applied")
                        for i, code in enumerate(applied_codes):
                            color = self._get_code_color(code)
                            col_up, col_down, col_code, col_remove = st.columns([1, 1, 6, 1])
                            with col_up:
                                st.button("▲", key=f"mc_up_{current_row}_{i}", disabled=i == 0,
                                          on_click=self._move_code_up, args=(current_row, i))
                            with col_down:
                                st.button("▼", key=f"mc_down_{current_row}_{i}", disabled=i == len(applied_codes) - 1,
                                          on_click=self._move_code_down, args=(current_row, i))
                            with col_code:
                                st.markdown(
                                    f'<span style="background-color: {color}; padding: 2px 6px; '
                                    f'border-radius: 3px;">{i + 1}. {code}</span>',
                                    unsafe_allow_html=True
                                )
                            with col_remove:
                                st.button("×", key=f"mc_remove_{current_row}_{i}",
                                          on_click=self._remove_code_at, args=(current_row, i))

                    st.markdown("---")
                    # Progress
                    coded_count = self._count_coded_rows()
                    progress = coded_count / total_rows if total_rows > 0 else 0
                    st.caption(f"**{coded_count}/{total_rows}** coded")
                    st.progress(progress)

        # Render the fragment
        coding_interface()

        # Export section (outside fragment)
        st.divider()
        st.subheader("Export Results")

        coded_count = self._count_coded_rows()
        if coded_count == 0:
            st.info("Code some rows before exporting")
        else:
            # Export format selection
            export_format = st.radio(
                "Export format",
                ["Standard (codes as text)", "One-hot encoding (codes as columns)"],
                horizontal=True,
                key="mc_export_format"
            )

            export_df = df.copy()

            if export_format == "One-hot encoding (codes as columns)":
                # Create one-hot encoded columns for each code
                for code in codes:
                    export_df[f"code_{code}"] = export_df.index.map(
                        lambda idx, c=code: 1 if c in st.session_state["mc_coding_data"].get(idx, []) else 0
                    )
                export_df["code_count"] = export_df.index.map(
                    lambda idx: len(st.session_state["mc_coding_data"].get(idx, []))
                )
            else:
                # Standard format: codes as semicolon-separated text
                export_df["applied_codes"] = export_df.index.map(
                    lambda idx: "; ".join(st.session_state["mc_coding_data"].get(idx, []))
                )
                export_df["code_count"] = export_df.index.map(
                    lambda idx: len(st.session_state["mc_coding_data"].get(idx, []))
                )

            with st.expander("Preview export data"):
                st.dataframe(export_df)

            render_download_buttons(export_df, filename_prefix="manual_coded_results")
