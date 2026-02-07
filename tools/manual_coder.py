"""
Manual Coder Tool
Code qualitative data manually with clickable codes
"""

import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List

from .base import BaseTool, ToolConfig, ToolResult
from core.sample_data import get_sample_data, get_dataset_info

# Auto-save directory
AUTOSAVE_DIR = Path(".manual_coder_saves")
AUTOSAVE_FILE = AUTOSAVE_DIR / "autosave.json"


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

    def _save_progress(self):
        """Auto-save coding progress to file"""
        try:
            AUTOSAVE_DIR.mkdir(exist_ok=True)
            save_data = {
                "coding_data": st.session_state.get("mc_coding_data", {}),
                "current_row": st.session_state.get("mc_current_row", 0),
                "codes": st.session_state.get("mc_codes", []),
                "text_col": st.session_state.get("mc_text_col"),
                "highlights": st.session_state.get("mc_highlights", {}),
            }
            # Convert int keys to strings for JSON
            save_data["coding_data"] = {str(k): v for k, v in save_data["coding_data"].items()}
            with open(AUTOSAVE_FILE, "w") as f:
                json.dump(save_data, f)
        except Exception as e:
            pass  # Silent fail for auto-save

    def _load_progress(self) -> bool:
        """Load saved progress if available. Returns True if loaded."""
        try:
            if AUTOSAVE_FILE.exists():
                with open(AUTOSAVE_FILE, "r") as f:
                    save_data = json.load(f)
                # Convert string keys back to int
                coding_data = {int(k): v for k, v in save_data.get("coding_data", {}).items()}
                st.session_state["mc_coding_data"] = coding_data
                st.session_state["mc_current_row"] = save_data.get("current_row", 0)
                st.session_state["mc_codes"] = save_data.get("codes", [])
                st.session_state["mc_text_col"] = save_data.get("text_col")
                st.session_state["mc_highlights"] = save_data.get("highlights", {})
                return True
        except Exception as e:
            pass
        return False

    def _clear_autosave(self):
        """Clear the autosave file"""
        try:
            if AUTOSAVE_FILE.exists():
                AUTOSAVE_FILE.unlink()
        except Exception:
            pass

    def _init_session_state(self):
        """Initialize session state keys for the tool"""
        if "mc_df" not in st.session_state:
            st.session_state["mc_df"] = None
        if "mc_codes" not in st.session_state:
            st.session_state["mc_codes"] = []
        if "mc_coding_data" not in st.session_state:
            st.session_state["mc_coding_data"] = {}
            # Try to load saved progress on first init
            if "mc_loaded_autosave" not in st.session_state:
                st.session_state["mc_loaded_autosave"] = self._load_progress()
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
            selected = st.session_state.get("mc_sample_choice", "product_reviews")
            df = pd.DataFrame(get_sample_data(selected))
            info = get_dataset_info()[selected]
            st.success(f"Using sample data: {info['name']} ({info['rows']} rows)")

            # Load sample codes if not already set
            if selected in self.SAMPLE_CODES and not st.session_state.get("mc_codes"):
                st.session_state["mc_codes"] = self.SAMPLE_CODES[selected]
            # Load sample highlights
            if selected in self.SAMPLE_HIGHLIGHTS:
                if not st.session_state.get("mc_highlights"):
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

                        st.success(f"Loaded {len(codes)} codes from '{code_col}' column")

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

        # Step 3: Select Text Column
        st.header("3. Select Text Column")

        text_columns = df.columns.tolist()
        default_col = st.session_state.get("mc_text_col")
        if default_col not in text_columns:
            default_col = text_columns[0] if text_columns else None

        text_col = st.selectbox(
            "Column to display for coding",
            options=text_columns,
            index=text_columns.index(default_col) if default_col in text_columns else 0,
            key="mc_text_col_select"
        )
        st.session_state["mc_text_col"] = text_col

        # Preview data
        st.dataframe(df.head(), height=150)

        # Show autosave restore message (stable location - before Start Coding button)
        if st.session_state.get("mc_loaded_autosave") and not st.session_state.get("mc_autosave_dismissed"):
            coded_count = self._count_coded_rows()
            if coded_count > 0:
                col_msg, col_btn1, col_btn2 = st.columns([3, 1, 1])
                with col_msg:
                    st.success(f"✅ Restored {coded_count} coded rows from autosave")
                with col_btn1:
                    if st.button("Clear", key="mc_clear_autosave", type="secondary"):
                        self._clear_autosave()
                        st.session_state["mc_coding_data"] = {}
                        st.session_state["mc_current_row"] = 0
                        st.session_state["mc_autosave_dismissed"] = True
                        st.rerun()
                with col_btn2:
                    if st.button("OK", key="mc_dismiss_autosave", type="primary"):
                        st.session_state["mc_autosave_dismissed"] = True
                        st.rerun()

        return ToolConfig(
            is_valid=True,
            config_data={
                "df": df,
                "codes": codes,
                "text_col": text_col
            }
        )

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
            opt_col1, opt_col2, opt_col3, opt_col4 = st.columns([1, 1, 1, 1])
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
                auto_adv = st.toggle("Auto-advance", value=st.session_state.get("mc_auto_advance", False), key="mc_auto")
                st.session_state["mc_auto_advance"] = auto_adv
            with opt_col4:
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

            # Main layout depends on mode
            if horizontal_mode:
                # Full width for text in horizontal mode
                col_text = st.container()
            else:
                # Text on left, codes on right
                col_text, col_codes = st.columns([5, 1])

            with col_text:
                # Calculate visible range
                start_row = max(0, current_row - context_rows)
                end_row = min(total_rows, current_row + context_rows + 1)

                # Fixed height container to prevent button jumping
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

                # Render in fixed height container with scroll
                st.markdown(
                    f'<div style="height: 350px; overflow-y: auto; padding-right: 10px;">{"".join(text_container_html)}</div>',
                    unsafe_allow_html=True
                )

            if horizontal_mode:
                # Horizontal layout: codes in a row below text, then Next button
                applied_codes = self._get_applied_codes(current_row)
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

                nav1, nav2, nav_spacer, nav3, nav4 = st.columns([1, 1, 3, 1, 1])
                with nav1:
                    st.button("◀◀", disabled=current_row <= 0, key="mc_prev5_h",
                              on_click=lambda: st.session_state.update({"mc_current_row": max(0, current_row - 5)}))
                with nav2:
                    st.button("◀ Prev", disabled=current_row <= 0, key="mc_prev_h", on_click=go_prev)
                with nav_spacer:
                    # Autosave indicator
                    autosave_enabled = st.session_state.get("mc_autosave_enabled", True)
                    if autosave_enabled:
                        st.markdown('<div style="text-align:center; color:#888; font-size:0.85em;">💾 Autosave enabled</div>', unsafe_allow_html=True)
                with nav3:
                    st.button("Next ▶", disabled=current_row >= total_rows - 1, key="mc_next_h", on_click=go_next)
                with nav4:
                    st.button("▶▶", disabled=current_row >= total_rows - 1, key="mc_next5_h",
                              on_click=lambda: st.session_state.update({"mc_current_row": min(total_rows - 1, current_row + 5)}))

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

            else:
                # Vertical layout in sidebar column (original)
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

                    nav1, nav2, nav_spacer, nav3, nav4 = st.columns([1, 1, 2, 1, 1])
                    with nav1:
                        st.button("◀◀", disabled=current_row <= 0, key="mc_prev5_v",
                                  on_click=lambda: st.session_state.update({"mc_current_row": max(0, current_row - 5)}))
                    with nav2:
                        st.button("◀ Prev", disabled=current_row <= 0, key="mc_prev_v", on_click=go_prev_v)
                    with nav_spacer:
                        autosave_enabled = st.session_state.get("mc_autosave_enabled", True)
                        if autosave_enabled:
                            st.markdown('<div style="text-align:center; color:#888; font-size:0.85em;">💾 Autosave</div>', unsafe_allow_html=True)
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
