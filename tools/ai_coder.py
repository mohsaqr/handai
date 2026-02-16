"""
AI Coder Tool
AI-Assisted Manual Coding with inter-rater reliability analytics
"""

import streamlit as st
import pandas as pd
import json
import os
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List

from .base import BaseTool, ToolConfig, ToolResult
from core.sample_data import get_sample_data, get_dataset_info
from core.llm_client import get_client, call_llm_simple
from core.providers import LLMProvider, PROVIDER_CONFIGS, supports_json_mode
from core.ai_coder_analytics import calculate_all_metrics, get_disagreement_analysis, interpret_kappa
from database import get_db

# Sessions directory - use user data directory for persistence
def get_sessions_dir() -> Path:
    """Get sessions directory in user's data folder"""
    if os.name == 'nt':  # Windows
        base = Path(os.environ.get('APPDATA', Path.home()))
    else:  # macOS/Linux
        base = Path.home() / '.local' / 'share'
    sessions_dir = base / 'handai' / 'ai_coder_sessions'
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir

SESSIONS_DIR = get_sessions_dir()

# AI Coder System Prompt
AI_CODER_SYSTEM_PROMPT = """Analyze the text and suggest codes.

AVAILABLE CODES:
{codes_list}

OUTPUT FORMAT (JSON only):
{{
  "codes": ["Code1", "Code2"],
  "confidence": {{"Code1": 0.95, "Code2": 0.72}},
  "reasoning": "Brief explanation"
}}

RULES:
- Only suggest from available codes
- Order by confidence (highest first)
- Include codes with confidence >= 0.3
- Return valid JSON only
"""


def _get_enabled_providers():
    """Return list of enabled configured providers."""
    db = get_db()
    providers = db.get_enabled_configured_providers()
    if providers:
        return [
            {
                "display_name": p.display_name,
                "provider_type": p.provider_type,
                "api_key": p.api_key,
                "base_url": p.base_url,
                "default_model": p.default_model,
            }
            for p in providers
        ]
    return []


class AICoderTool(BaseTool):
    """Tool for AI-assisted manual coding of qualitative data"""

    id = "ai_coder"
    name = "AI Coder"
    description = "AI-assisted manual coding with inter-rater reliability"
    icon = ":material/smart_toy:"
    category = "Processing"

    # Sample codes for each dataset (same as manual_coder)
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

    # Color palette for code highlights
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
        return f"ai_session_{now.strftime('%Y%m%d_%H%M%S')}"

    def _get_session_file(self, name: str) -> Path:
        """Get path for a session file"""
        return SESSIONS_DIR / f"{name}.json"

    def _save_session(self, name: str = None) -> str:
        """Save coding session to file. Returns session name."""
        try:
            SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
            if not name:
                name = st.session_state.get("aic_current_session", self._generate_session_name())

            save_data = {
                "name": name,
                "tool": "ai_coder",
                "version": "2.0",
                "saved_at": datetime.now().isoformat(),
                "data": {
                    "coding_data": {str(k): v for k, v in st.session_state.get("aic_coding_data", {}).items()},
                    "current_row": st.session_state.get("aic_current_row", 0),
                    "total_rows": len(st.session_state.get("aic_df", [])) if st.session_state.get("aic_df") is not None else 0,
                    "codes": st.session_state.get("aic_codes", []),
                    "text_cols": st.session_state.get("aic_text_cols", []),
                    "highlights": st.session_state.get("aic_highlights", {}),
                    "coded_count": self._count_coded_rows(),
                    "sample_dataset": st.session_state.get("aic_sample_choice") if st.session_state.get("aic_use_sample") else None,
                },
                "ai_config": {
                    "mode": st.session_state.get("aic_ai_mode", "per_row"),
                    "display": st.session_state.get("aic_ai_display", "ai_first"),
                    "provider": st.session_state.get("aic_ai_provider", ""),
                    "model": st.session_state.get("aic_ai_model", ""),
                    # Advanced AI Settings
                    "custom_prompt_enabled": st.session_state.get("aic_custom_prompt_enabled", False),
                    "custom_prompt": st.session_state.get("aic_custom_prompt", ""),
                    "code_definitions": st.session_state.get("aic_code_definitions", {}),
                    "thresholds_enabled": st.session_state.get("aic_thresholds_enabled", False),
                    "threshold_auto_accept": st.session_state.get("aic_threshold_auto_accept", 0.90),
                    "threshold_flag": st.session_state.get("aic_threshold_flag", 0.50),
                    "threshold_skip": st.session_state.get("aic_threshold_skip", 0.20),
                    "ai_context_rows": st.session_state.get("aic_ai_context_rows", 0),
                    "training_enabled": st.session_state.get("aic_training_enabled", False),
                    "training_examples_count": st.session_state.get("aic_training_examples_count", 5),
                },
                "ai_suggestions": {str(k): v for k, v in st.session_state.get("aic_ai_suggestions", {}).items()},
                "automation_data": {
                    "flagged_rows": list(st.session_state.get("aic_flagged_rows", set())),
                    "auto_accepted_rows": list(st.session_state.get("aic_auto_accepted_rows", set())),
                },
            }

            with open(self._get_session_file(name), "w") as f:
                json.dump(save_data, f, indent=2)

            st.session_state["aic_current_session"] = name
            return name
        except Exception as e:
            return None

    def _save_progress(self):
        """Auto-save current session"""
        if st.session_state.get("aic_current_session"):
            self._save_session(st.session_state["aic_current_session"])

    def _load_session(self, name: str) -> bool:
        """Load a saved session. Returns True if loaded."""
        try:
            session_file = self._get_session_file(name)
            if session_file.exists():
                with open(session_file, "r") as f:
                    save_data = json.load(f)

                data = save_data.get("data", save_data)  # Handle both old and new format

                # Core data
                coding_data = {int(k): v for k, v in data.get("coding_data", {}).items()}
                st.session_state["aic_coding_data"] = coding_data
                st.session_state["aic_current_row"] = data.get("current_row", 0)
                st.session_state["aic_codes"] = data.get("codes", [])
                # Backward compat: handle old text_col (string) or new text_cols (list)
                text_cols = data.get("text_cols")
                if text_cols is None:
                    old_col = data.get("text_col")
                    text_cols = [old_col] if old_col else []
                st.session_state["aic_text_cols"] = text_cols
                st.session_state["aic_highlights"] = data.get("highlights", {})
                st.session_state["aic_current_session"] = name

                # AI config
                ai_config = save_data.get("ai_config", {})
                st.session_state["aic_ai_mode"] = ai_config.get("mode", "per_row")
                st.session_state["aic_ai_display"] = ai_config.get("display", "ai_first")
                st.session_state["aic_ai_provider"] = ai_config.get("provider", "")
                st.session_state["aic_ai_model"] = ai_config.get("model", "")

                # Advanced AI Settings
                st.session_state["aic_custom_prompt_enabled"] = ai_config.get("custom_prompt_enabled", False)
                st.session_state["aic_custom_prompt"] = ai_config.get("custom_prompt", "")
                st.session_state["aic_code_definitions"] = ai_config.get("code_definitions", {})
                st.session_state["aic_thresholds_enabled"] = ai_config.get("thresholds_enabled", False)
                st.session_state["aic_threshold_auto_accept"] = ai_config.get("threshold_auto_accept", 0.90)
                st.session_state["aic_threshold_flag"] = ai_config.get("threshold_flag", 0.50)
                st.session_state["aic_threshold_skip"] = ai_config.get("threshold_skip", 0.20)
                st.session_state["aic_ai_context_rows"] = ai_config.get("ai_context_rows", 0)
                st.session_state["aic_training_enabled"] = ai_config.get("training_enabled", False)
                st.session_state["aic_training_examples_count"] = ai_config.get("training_examples_count", 5)

                # AI suggestions
                ai_suggestions = save_data.get("ai_suggestions", {})
                st.session_state["aic_ai_suggestions"] = {int(k): v for k, v in ai_suggestions.items()}

                # Automation data
                automation_data = save_data.get("automation_data", {})
                st.session_state["aic_flagged_rows"] = set(automation_data.get("flagged_rows", []))
                st.session_state["aic_auto_accepted_rows"] = set(automation_data.get("auto_accepted_rows", []))

                # Restore sample dataset if it was used
                sample_dataset = data.get("sample_dataset")
                if sample_dataset:
                    st.session_state["aic_use_sample"] = True
                    st.session_state["aic_sample_choice"] = sample_dataset
                    st.session_state["aic_last_sample"] = sample_dataset
                    df = pd.DataFrame(get_sample_data(sample_dataset))
                    st.session_state["aic_df"] = df

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
                            session_data = data.get("data", data)
                            sessions.append({
                                "name": f.stem,
                                "saved_at": data.get("saved_at", ""),
                                "coded_count": session_data.get("coded_count", 0),
                                "total_rows": session_data.get("total_rows", 0),
                                "current_row": session_data.get("current_row", 0),
                                "has_ai": bool(data.get("ai_suggestions")),
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

    def _rename_session(self, old_name: str, new_name: str) -> bool:
        """Rename a session file. Returns True if successful."""
        try:
            old_file = self._get_session_file(old_name)
            new_file = self._get_session_file(new_name)
            if old_file.exists() and not new_file.exists():
                with open(old_file, "r") as f:
                    data = json.load(f)
                data["name"] = new_name
                with open(new_file, "w") as f:
                    json.dump(data, f, indent=2)
                old_file.unlink()
                if st.session_state.get("aic_current_session") == old_name:
                    st.session_state["aic_current_session"] = new_name
                return True
        except:
            pass
        return False

    def _init_session_state(self):
        """Initialize session state keys for the tool"""
        first_init = "aic_initialized" not in st.session_state

        # Core state (replicated from manual_coder with aic_ prefix)
        if "aic_df" not in st.session_state:
            st.session_state["aic_df"] = None
        if "aic_codes" not in st.session_state:
            st.session_state["aic_codes"] = []
        if "aic_coding_data" not in st.session_state:
            st.session_state["aic_coding_data"] = {}
        if "aic_current_row" not in st.session_state:
            st.session_state["aic_current_row"] = 0
        if "aic_text_cols" not in st.session_state:
            st.session_state["aic_text_cols"] = []
        # Use global coding settings as defaults
        if "aic_auto_advance" not in st.session_state:
            st.session_state["aic_auto_advance"] = st.session_state.get("coding_auto_advance", False)
        if "aic_highlights" not in st.session_state:
            st.session_state["aic_highlights"] = {}
        if "aic_context_rows" not in st.session_state:
            st.session_state["aic_context_rows"] = st.session_state.get("coding_context_rows", 2)
        if "aic_codebook" not in st.session_state:
            st.session_state["aic_codebook"] = None
        if "aic_light_mode" not in st.session_state:
            st.session_state["aic_light_mode"] = st.session_state.get("coding_light_mode", True)
        if "aic_horizontal_codes" not in st.session_state:
            st.session_state["aic_horizontal_codes"] = st.session_state.get("coding_horizontal_codes", False)
        if "aic_autosave_enabled" not in st.session_state:
            st.session_state["aic_autosave_enabled"] = st.session_state.get("aic_autosave_enabled", True)
        if "aic_buttons_above" not in st.session_state:
            st.session_state["aic_buttons_above"] = st.session_state.get("coding_buttons_above", False)
        if "aic_immersive_trigger" not in st.session_state:
            st.session_state["aic_immersive_trigger"] = False
        if "aic_immersive_active" not in st.session_state:
            st.session_state["aic_immersive_active"] = False
        if "aic_current_session" not in st.session_state:
            st.session_state["aic_current_session"] = None
        if "aic_use_sample" not in st.session_state:
            st.session_state["aic_use_sample"] = False
        if "aic_session_restored" not in st.session_state:
            st.session_state["aic_session_restored"] = False

        # AI-specific state - use global defaults
        if "aic_ai_mode" not in st.session_state:
            st.session_state["aic_ai_mode"] = st.session_state.get("aic_default_mode", "per_row")
        if "aic_ai_display" not in st.session_state:
            st.session_state["aic_ai_display"] = st.session_state.get("aic_default_display", "ai_first")
        if "aic_ai_suggestions" not in st.session_state:
            st.session_state["aic_ai_suggestions"] = {}  # {row_idx: {codes, confidence, reasoning}}
        if "aic_ai_batch_status" not in st.session_state:
            st.session_state["aic_ai_batch_status"] = "idle"  # "idle" | "running" | "complete"
        if "aic_ai_provider" not in st.session_state:
            st.session_state["aic_ai_provider"] = ""
        if "aic_ai_model" not in st.session_state:
            st.session_state["aic_ai_model"] = ""
        if "aic_batch_progress" not in st.session_state:
            st.session_state["aic_batch_progress"] = 0

        # Advanced AI Settings - Custom Prompt
        if "aic_custom_prompt_enabled" not in st.session_state:
            st.session_state["aic_custom_prompt_enabled"] = False
        if "aic_custom_prompt" not in st.session_state:
            st.session_state["aic_custom_prompt"] = ""

        # Advanced AI Settings - Code Definitions
        if "aic_code_definitions" not in st.session_state:
            st.session_state["aic_code_definitions"] = {}  # {code: {definition, examples, keywords}}

        # Advanced AI Settings - Confidence Thresholds
        if "aic_thresholds_enabled" not in st.session_state:
            st.session_state["aic_thresholds_enabled"] = False
        if "aic_threshold_auto_accept" not in st.session_state:
            st.session_state["aic_threshold_auto_accept"] = 0.90
        if "aic_threshold_flag" not in st.session_state:
            st.session_state["aic_threshold_flag"] = 0.50
        if "aic_threshold_skip" not in st.session_state:
            st.session_state["aic_threshold_skip"] = 0.20
        if "aic_flagged_rows" not in st.session_state:
            st.session_state["aic_flagged_rows"] = set()
        if "aic_auto_accepted_rows" not in st.session_state:
            st.session_state["aic_auto_accepted_rows"] = set()

        # Advanced AI Settings - AI Context Window
        if "aic_ai_context_rows" not in st.session_state:
            st.session_state["aic_ai_context_rows"] = 0  # 0-3 rows before/after for AI prompt

        # Advanced AI Settings - Training Mode
        if "aic_training_enabled" not in st.session_state:
            st.session_state["aic_training_enabled"] = False
        if "aic_training_examples_count" not in st.session_state:
            st.session_state["aic_training_examples_count"] = 5

        # Auto-load most recent session on first init
        if first_init:
            st.session_state["aic_initialized"] = True
            st.session_state["aic_immersive_trigger"] = False
            st.session_state["aic_immersive_active"] = False
            try:
                sessions = self._list_sessions()
                if sessions:
                    most_recent = sessions[0]["name"]
                    if self._load_session(most_recent):
                        st.session_state["aic_session_restored"] = True
                        st.toast(f"Resumed: {most_recent}")
            except Exception as e:
                st.toast(f"Could not restore session: {e}")

    def _add_code(self, row_idx: int, code: str):
        """Add a code to a specific row"""
        if row_idx not in st.session_state["aic_coding_data"]:
            st.session_state["aic_coding_data"][row_idx] = []
        st.session_state["aic_coding_data"][row_idx].append(code)
        self._save_progress()

        if st.session_state["aic_auto_advance"]:
            total_rows = len(st.session_state["aic_df"])
            if st.session_state["aic_current_row"] < total_rows - 1:
                st.session_state["aic_current_row"] += 1

    def _remove_code_at(self, row_idx: int, position: int):
        """Remove a code at a specific position"""
        if row_idx in st.session_state["aic_coding_data"]:
            codes = st.session_state["aic_coding_data"][row_idx]
            if 0 <= position < len(codes):
                codes.pop(position)
                st.session_state["aic_coding_data"][row_idx] = codes
                self._save_progress()

    def _move_code_up(self, row_idx: int, position: int):
        """Move a code up in the list"""
        if row_idx in st.session_state["aic_coding_data"]:
            codes = st.session_state["aic_coding_data"][row_idx]
            if 0 < position < len(codes):
                codes[position], codes[position - 1] = codes[position - 1], codes[position]
                st.session_state["aic_coding_data"][row_idx] = codes
                self._save_progress()

    def _move_code_down(self, row_idx: int, position: int):
        """Move a code down in the list"""
        if row_idx in st.session_state["aic_coding_data"]:
            codes = st.session_state["aic_coding_data"][row_idx]
            if 0 <= position < len(codes) - 1:
                codes[position], codes[position + 1] = codes[position + 1], codes[position]
                st.session_state["aic_coding_data"][row_idx] = codes
                self._save_progress()

    def _get_code_color(self, code: str) -> str:
        """Get consistent color for a code"""
        codes = st.session_state.get("aic_codes", [])
        if code in codes:
            idx = codes.index(code) % len(self.CODE_COLORS)
            return self.CODE_COLORS[idx]
        return self.CODE_COLORS[0]

    def _highlight_text(self, text: str) -> str:
        """Apply word highlights to text based on defined patterns"""
        import re
        highlighted = str(text)
        highlights = st.session_state.get("aic_highlights", {})

        for code, words_str in highlights.items():
            if not words_str.strip():
                continue
            color = self._get_code_color(code)
            words = [w.strip() for w in words_str.split(",") if w.strip()]
            for word in words:
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                highlighted = pattern.sub(
                    f'<mark style="background-color: {color}; padding: 2px 4px; border-radius: 3px;">{word}</mark>',
                    highlighted
                )
        return highlighted

    def _get_applied_codes(self, row_idx: int) -> List[str]:
        """Get list of applied codes for a row"""
        return st.session_state["aic_coding_data"].get(row_idx, [])

    def _count_coded_rows(self) -> int:
        """Count how many rows have at least one code"""
        return sum(1 for codes in st.session_state["aic_coding_data"].values() if codes)

    async def _get_ai_suggestion(self, row_idx: int, text: str) -> Optional[Dict]:
        """Get AI suggestion for a single row"""
        # Check cache first
        if row_idx in st.session_state.get("aic_ai_suggestions", {}):
            return st.session_state["aic_ai_suggestions"][row_idx]

        provider_name = st.session_state.get("aic_ai_provider", "")
        model = st.session_state.get("aic_ai_model", "")
        codes = st.session_state.get("aic_codes", [])

        if not provider_name:
            return {"codes": [], "confidence": {}, "reasoning": "No AI provider selected", "error": True}
        if not model:
            return {"codes": [], "confidence": {}, "reasoning": "No AI model selected", "error": True}
        if not codes:
            return {"codes": [], "confidence": {}, "reasoning": "No codes defined", "error": True}

        # Get provider config
        providers = _get_enabled_providers()
        provider_entry = None
        for p in providers:
            if p["display_name"] == provider_name:
                provider_entry = p
                break

        if not provider_entry:
            return {"codes": [], "confidence": {}, "reasoning": f"Provider '{provider_name}' not found in configured providers", "error": True}

        try:
            provider_enum = LLMProvider(provider_entry["provider_type"])
        except ValueError:
            provider_enum = LLMProvider.CUSTOM

        client = get_client(
            provider_enum,
            provider_entry["api_key"] or "dummy",
            provider_entry["base_url"]
        )

        # Build prompt using advanced settings
        system_prompt = self._build_ai_prompt()

        # Build context text (includes surrounding rows if configured)
        df = st.session_state.get("aic_df")
        text_cols = st.session_state.get("aic_text_cols", [])
        if df is not None and text_cols:
            context_text = self._build_context_text(df, text_cols, row_idx)
        else:
            context_text = text

        # Check if JSON mode is supported
        use_json_mode = supports_json_mode(provider_enum, model)

        try:
            output, error = await call_llm_simple(
                client=client,
                system_prompt=system_prompt,
                user_content=f"TEXT TO ANALYZE:\n{context_text}",
                model=model,
                temperature=0.3,
                max_tokens=500,
                json_mode=use_json_mode,
                provider=provider_enum
            )

            if error:
                return {"codes": [], "confidence": {}, "reasoning": f"Error: {error}", "error": True}

            if not output:
                return {"codes": [], "confidence": {}, "reasoning": "Empty response from AI", "error": True}

            # Parse JSON response - try multiple extraction methods
            import re
            result = None

            # Method 1: Try direct JSON parse
            try:
                result = json.loads(output)
            except json.JSONDecodeError:
                pass

            # Method 2: Try to find JSON in markdown code blocks
            if not result:
                code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', output, re.DOTALL)
                if code_block_match:
                    try:
                        result = json.loads(code_block_match.group(1))
                    except json.JSONDecodeError:
                        pass

            # Method 3: Try to find any JSON object with "codes" key
            if not result:
                # Match JSON object that contains "codes" - handle nested braces
                json_match = re.search(r'\{[^{}]*"codes"\s*:\s*\[[^\]]*\][^{}]*\}', output, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass

            # Method 4: Try to find JSON starting with { and ending with }
            if not result:
                # Find the first { and last }
                start = output.find('{')
                end = output.rfind('}')
                if start != -1 and end != -1 and end > start:
                    try:
                        result = json.loads(output[start:end+1])
                    except json.JSONDecodeError:
                        pass

            # Method 5: Try to extract codes from text if all else fails
            if not result:
                # Look for codes mentioned in the text
                found_codes = []
                for code in codes:
                    if code.lower() in output.lower():
                        found_codes.append(code)
                if found_codes:
                    result = {
                        "codes": found_codes,
                        "confidence": {c: 0.7 for c in found_codes},
                        "reasoning": "Extracted from AI text response"
                    }

            if not result:
                return {"codes": [], "confidence": {}, "reasoning": f"Could not parse AI response: {output[:300]}", "error": True}

            # Validate and filter codes
            valid_codes = [c for c in result.get("codes", []) if c in codes]
            confidence = {k: v for k, v in result.get("confidence", {}).items() if k in codes}
            suggestion = {
                "codes": valid_codes,
                "confidence": confidence,
                "reasoning": result.get("reasoning", "")
            }

            # Apply confidence thresholds
            suggestion = self._apply_confidence_thresholds(row_idx, suggestion)

            # Cache the result
            if "aic_ai_suggestions" not in st.session_state:
                st.session_state["aic_ai_suggestions"] = {}
            st.session_state["aic_ai_suggestions"][row_idx] = suggestion
            self._save_progress()
            return suggestion

        except Exception as e:
            return {"codes": [], "confidence": {}, "reasoning": f"Error: {str(e)}", "error": True}

    async def _process_batch(self, df: pd.DataFrame, text_cols: list, progress_callback=None):
        """Process all rows in batch mode"""
        import asyncio

        total_rows = len(df)
        st.session_state["aic_ai_batch_status"] = "running"
        st.session_state["aic_batch_progress"] = 0
        st.session_state["aic_batch_debug"] = f"Starting batch for {total_rows} rows"

        # Clear ALL previous suggestions to force fresh batch
        st.session_state["aic_ai_suggestions"] = {}
        # Also clear threshold tracking
        st.session_state["aic_flagged_rows"] = set()
        st.session_state["aic_auto_accepted_rows"] = set()

        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests

        async def process_row(row_idx: int, text: str):
            async with semaphore:
                result = await self._get_ai_suggestion(row_idx, text)
                return row_idx, result

        tasks = []
        for row_idx in range(total_rows):
            text = self._get_row_text(df, text_cols, row_idx)
            tasks.append(process_row(row_idx, text))

        st.session_state["aic_batch_debug"] = f"Created {len(tasks)} tasks"

        if not tasks:
            st.session_state["aic_ai_batch_status"] = "complete"
            st.session_state["aic_batch_progress"] = 1.0
            st.session_state["aic_batch_debug"] = "No tasks to process"
            return

        completed = 0
        errors = 0
        first_error = None
        for coro in asyncio.as_completed(tasks):
            row_idx, result = await coro
            completed += 1
            if result and result.get("error"):
                errors += 1
                if not first_error:
                    first_error = result.get("reasoning", "Unknown")
            st.session_state["aic_batch_progress"] = completed / len(tasks)
            if progress_callback:
                progress_callback(completed, len(tasks))

        st.session_state["aic_ai_batch_status"] = "complete"
        st.session_state["aic_batch_errors"] = errors
        st.session_state["aic_batch_debug"] = f"Completed: {completed}, Errors: {errors}"
        if first_error:
            st.session_state["aic_batch_first_error"] = first_error
        self._save_progress()

    def _accept_ai_suggestions(self, row_idx: int):
        """Accept all AI suggestions for current row"""
        suggestion = st.session_state.get("aic_ai_suggestions", {}).get(row_idx, {})
        if suggestion and suggestion.get("codes"):
            for code in suggestion["codes"]:
                if code not in self._get_applied_codes(row_idx):
                    self._add_code(row_idx, code)

    def _build_ai_prompt(self) -> str:
        """Build complete AI prompt with code definitions, custom prompt, and training examples"""
        codes = st.session_state.get("aic_codes", [])
        code_definitions = st.session_state.get("aic_code_definitions", {})
        custom_prompt_enabled = st.session_state.get("aic_custom_prompt_enabled", False)
        custom_prompt = st.session_state.get("aic_custom_prompt", "")
        training_enabled = st.session_state.get("aic_training_enabled", False)

        # Build codes list with definitions
        codes_section = "AVAILABLE CODES:\n"
        for code in codes:
            codes_section += f"- {code}"
            if code in code_definitions:
                defn = code_definitions[code]
                if defn.get("definition"):
                    codes_section += f"\n  Definition: {defn['definition']}"
                if defn.get("examples"):
                    codes_section += f"\n  Examples: {defn['examples']}"
                if defn.get("keywords"):
                    codes_section += f"\n  Keywords: {defn['keywords']}"
            codes_section += "\n"

        # Build training examples section
        training_section = ""
        if training_enabled:
            training_section = self._build_training_examples()

        # Use custom prompt or default
        if custom_prompt_enabled and custom_prompt.strip():
            # Replace placeholders in custom prompt
            prompt = custom_prompt.replace("{codes_list}", codes_section)
            if training_section:
                prompt = prompt.replace("{training_examples}", training_section)
            else:
                prompt = prompt.replace("{training_examples}", "")
        else:
            # Default prompt
            prompt = f"""Analyze the text and suggest codes.

{codes_section}
{training_section}
OUTPUT FORMAT (JSON only):
{{
  "codes": ["Code1", "Code2"],
  "confidence": {{"Code1": 0.95, "Code2": 0.72}},
  "reasoning": "Brief explanation"
}}

RULES:
- Only suggest from available codes
- Order by confidence (highest first)
- Include codes with confidence >= 0.3
- Return valid JSON only
"""

        return prompt

    def _get_row_text(self, df: pd.DataFrame, text_cols: list, row_idx: int) -> str:
        """Get combined text from selected columns for a row"""
        if not text_cols:
            return ""
        if len(text_cols) == 1:
            return str(df.iloc[row_idx][text_cols[0]])
        parts = []
        for col in text_cols:
            parts.append(f"[{col}]: {df.iloc[row_idx][col]}")
        return "\n".join(parts)

    def _get_row_text_html(self, df: pd.DataFrame, text_cols: list, row_idx: int) -> str:
        """Get combined text from selected columns for a row, formatted as HTML"""
        if not text_cols:
            return ""
        if len(text_cols) == 1:
            return str(df.iloc[row_idx][text_cols[0]])
        parts = []
        for col in text_cols:
            parts.append(f"<b>{col}:</b> {df.iloc[row_idx][col]}")
        return "<br/>".join(parts)

    def _build_context_text(self, df: pd.DataFrame, text_cols: list, row_idx: int) -> str:
        """Build text with N surrounding rows for context"""
        ai_context_rows = st.session_state.get("aic_ai_context_rows", 0)

        if ai_context_rows == 0:
            return self._get_row_text(df, text_cols, row_idx)

        total_rows = len(df)
        start_idx = max(0, row_idx - ai_context_rows)
        end_idx = min(total_rows, row_idx + ai_context_rows + 1)

        context_parts = []
        for idx in range(start_idx, end_idx):
            text = self._get_row_text(df, text_cols, idx)
            if idx == row_idx:
                context_parts.append(f"[CURRENT ROW {idx + 1}]: {text}")
            else:
                context_parts.append(f"[Row {idx + 1}]: {text}")

        return "\n\n".join(context_parts)

    def _build_training_examples(self) -> str:
        """Format few-shot examples from manually coded rows"""
        training_count = st.session_state.get("aic_training_examples_count", 5)
        coding_data = st.session_state.get("aic_coding_data", {})
        df = st.session_state.get("aic_df")
        text_cols = st.session_state.get("aic_text_cols", [])

        if df is None or not text_cols or not coding_data:
            return ""

        # Get rows that have been coded
        coded_rows = [(idx, codes) for idx, codes in coding_data.items() if codes]
        if not coded_rows:
            return ""

        # Take up to training_count examples
        examples = coded_rows[:training_count]

        training_section = "TRAINING EXAMPLES (learn from these human-coded samples):\n"
        for row_idx, codes in examples:
            if row_idx < len(df):
                text = self._get_row_text(df, text_cols, row_idx)
                # Truncate long texts
                if len(text) > 200:
                    text = text[:200] + "..."
                training_section += f"\nText: \"{text}\"\nCodes: {codes}\n"

        training_section += "\nUse these examples to understand the coding patterns.\n\n"
        return training_section

    def _apply_confidence_thresholds(self, row_idx: int, suggestion: dict) -> dict:
        """Apply confidence thresholds to filter, auto-accept, or flag suggestions"""
        if not st.session_state.get("aic_thresholds_enabled", False):
            return suggestion

        if not suggestion or suggestion.get("error") or not suggestion.get("codes"):
            return suggestion

        threshold_auto_accept = st.session_state.get("aic_threshold_auto_accept", 0.90)
        threshold_flag = st.session_state.get("aic_threshold_flag", 0.50)
        threshold_skip = st.session_state.get("aic_threshold_skip", 0.20)

        confidence_dict = suggestion.get("confidence", {})
        codes = suggestion.get("codes", [])

        # Calculate average confidence
        if codes and confidence_dict:
            avg_confidence = sum(confidence_dict.get(c, 0) for c in codes) / len(codes)
        else:
            avg_confidence = 0

        # Skip if below skip threshold
        if avg_confidence < threshold_skip:
            suggestion["skipped"] = True
            suggestion["threshold_action"] = "skip"
            return suggestion

        # Auto-accept if above auto-accept threshold
        if avg_confidence >= threshold_auto_accept:
            # Apply codes automatically
            for code in codes:
                if code not in self._get_applied_codes(row_idx):
                    if row_idx not in st.session_state["aic_coding_data"]:
                        st.session_state["aic_coding_data"][row_idx] = []
                    st.session_state["aic_coding_data"][row_idx].append(code)
            st.session_state["aic_auto_accepted_rows"].add(row_idx)
            suggestion["auto_accepted"] = True
            suggestion["threshold_action"] = "auto_accept"
            return suggestion

        # Flag if below flag threshold
        if avg_confidence < threshold_flag:
            st.session_state["aic_flagged_rows"].add(row_idx)
            suggestion["flagged"] = True
            suggestion["threshold_action"] = "flag"

        return suggestion

    def render_config(self) -> ToolConfig:
        """Render AI coder configuration UI"""
        self._init_session_state()

        # Step 1: Load Data
        st.header("1. Load Data")

        col_upload, col_sample = st.columns([3, 1])
        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload Dataset",
                type=["csv", "xlsx", "xls"],
                key="aic_upload"
            )
        with col_sample:
            st.write("")
            use_sample = st.button(
                "Use Sample Data",
                help="Load sample data for testing",
                key="aic_sample_btn"
            )

        session_restored = st.session_state.get("aic_session_restored", False)

        if use_sample:
            st.session_state["aic_use_sample"] = True
            st.session_state["aic_session_restored"] = False
            st.session_state["aic_coding_data"] = {}
            st.session_state["aic_current_row"] = 0
            st.session_state["aic_codes"] = []
            st.session_state["aic_highlights"] = {}
            st.session_state["aic_ai_suggestions"] = {}

        if st.session_state.get("aic_use_sample"):
            sample_options = {
                "mixed_feedback": "Mixed Feedback (15 items, varying lengths)",
                "product_reviews": "Product Reviews (20 reviews with sentiment)",
                "healthcare_interviews": "Healthcare Interviews (15 worker experiences)",
                "support_tickets": "Support Tickets (20 customer issues)",
                "learning_experience": "Learning Experience (20 student responses)",
                "exit_interviews": "Exit Interviews (15 employee departures)",
            }
            current_sample = st.session_state.get("aic_sample_choice", "mixed_feedback")
            default_idx = list(sample_options.keys()).index(current_sample) if current_sample in sample_options else 0
            selected_sample = st.selectbox(
                "Choose sample dataset",
                options=list(sample_options.keys()),
                format_func=lambda x: sample_options[x],
                index=default_idx,
                key="aic_sample_choice_select"
            )
            st.session_state["aic_sample_choice"] = selected_sample

        df = st.session_state.get("aic_df")

        if uploaded_file:
            file_ext = uploaded_file.name.split(".")[-1].lower()
            try:
                if file_ext == "csv":
                    df = pd.read_csv(uploaded_file)
                elif file_ext in ["xlsx", "xls"]:
                    df = pd.read_excel(uploaded_file)
                if df is not None and not df.equals(st.session_state.get("aic_df")):
                    st.session_state["aic_coding_data"] = {}
                    st.session_state["aic_current_row"] = 0
                    st.session_state["aic_ai_suggestions"] = {}
            except Exception as e:
                return ToolConfig(
                    is_valid=False,
                    error_message=f"Error loading file: {str(e)}"
                )

        elif st.session_state.get("aic_use_sample"):
            selected = st.session_state.get("aic_sample_choice", "mixed_feedback")
            df = pd.DataFrame(get_sample_data(selected))
            info = get_dataset_info()[selected]

            if st.session_state.get("aic_last_sample") != selected:
                st.session_state["aic_last_sample"] = selected

                if session_restored:
                    st.toast(f"Resumed session with {info['name']}")
                    st.session_state["aic_session_restored"] = False
                else:
                    st.toast(f"Loaded: {info['name']} ({info['rows']} rows)")

                    if "text_column" in info:
                        st.session_state["aic_text_cols"] = [info["text_column"]]

                    if selected in self.SAMPLE_CODES:
                        st.session_state["aic_codes"] = self.SAMPLE_CODES[selected]
                    if selected in self.SAMPLE_HIGHLIGHTS:
                        st.session_state["aic_highlights"] = self.SAMPLE_HIGHLIGHTS[selected]

        if df is None or (hasattr(df, 'empty') and df.empty):
            return ToolConfig(
                is_valid=False,
                error_message="Please upload a file or use sample data"
            )

        if df.empty:
            return ToolConfig(
                is_valid=False,
                error_message="The uploaded file is empty"
            )

        st.session_state["aic_df"] = df

        # Step 2: Define Codes
        st.header("2. Define Codes")

        code_method = st.radio(
            "How to define codes?",
            ["Enter manually", "Upload codebook"],
            horizontal=True,
            key="aic_code_method"
        )

        codes = []
        codebook_df = None

        if code_method == "Upload codebook":
            codebook_file = st.file_uploader(
                "Upload codebook (CSV/Excel) - first column = codes",
                type=["csv", "xlsx", "xls"],
                key="aic_codebook_upload"
            )

            if codebook_file:
                try:
                    ext = codebook_file.name.split(".")[-1].lower()
                    if ext == "csv":
                        codebook_df = pd.read_csv(codebook_file)
                    else:
                        codebook_df = pd.read_excel(codebook_file)

                    if not codebook_df.empty:
                        code_col = codebook_df.columns[0]
                        codes = codebook_df[code_col].dropna().astype(str).tolist()
                        st.session_state["aic_codes"] = codes
                        st.session_state["aic_codebook"] = codebook_df
                        st.toast(f"Loaded {len(codes)} codes from codebook")

                        with st.expander("Codebook Reference", expanded=False):
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

                except Exception as e:
                    st.error(f"Error loading codebook: {str(e)}")
            else:
                if st.session_state.get("aic_codebook") is not None:
                    codebook_df = st.session_state["aic_codebook"]
                    code_col = codebook_df.columns[0]
                    codes = codebook_df[code_col].dropna().astype(str).tolist()
                else:
                    st.info("Upload a codebook file with codes in the first column.")

        else:  # Enter manually
            codes_input = st.text_area(
                "Enter your codes (one per line)",
                value="\n".join(st.session_state.get("aic_codes", [])),
                height=150,
                help="Enter the codes you want to apply to the data, one code per line",
                key="aic_codes_input"
            )
            codes = [c.strip() for c in codes_input.strip().split("\n") if c.strip()]
            st.session_state["aic_codes"] = codes

        if not codes:
            return ToolConfig(
                is_valid=False,
                error_message="Please enter at least one code"
            )

        st.caption(f"{len(codes)} code(s) defined")

        # Step 3: Select Text Columns
        all_columns = df.columns.tolist()
        default_cols = st.session_state.get("aic_text_cols", [])
        # Filter out any columns that no longer exist in the data
        default_cols = [c for c in default_cols if c in all_columns]
        if not default_cols:
            default_cols = [all_columns[0]] if all_columns else []

        if default_cols:
            label = ", ".join(default_cols)
            with st.expander(f"Text Columns: **{label}** (click to change)"):
                text_cols = st.multiselect(
                    "Columns to display for coding",
                    options=all_columns,
                    default=default_cols,
                    key="aic_text_cols_select"
                )
        else:
            st.header("3. Select Text Columns")
            text_cols = st.multiselect(
                "Columns to display for coding",
                options=all_columns,
                default=default_cols,
                key="aic_text_cols_select"
            )
        if not text_cols:
            text_cols = default_cols
        st.session_state["aic_text_cols"] = text_cols

        if default_cols:
            with st.expander("Preview data"):
                st.dataframe(df.head(), height=150)
        else:
            st.dataframe(df.head(), height=150)

        # Step 4: AI Configuration (NEW)
        st.header("4. AI Configuration")

        providers = _get_enabled_providers()

        if not providers:
            st.warning("No AI providers configured. Go to **LLM Providers** to set up an API key.")
            st.info("You can still use manual coding without AI assistance.")
        else:
            provider_names = [p["display_name"] for p in providers]
            current_provider = st.session_state.get("aic_ai_provider", "")
            default_provider_idx = 0
            if current_provider in provider_names:
                default_provider_idx = provider_names.index(current_provider)

            col_prov, col_model = st.columns(2)
            with col_prov:
                selected_provider = st.selectbox(
                    "Provider",
                    options=provider_names,
                    index=default_provider_idx,
                    key="aic_provider_select"
                )
                st.session_state["aic_ai_provider"] = selected_provider

            with col_model:
                provider_entry = providers[provider_names.index(selected_provider)]
                default_model = provider_entry.get("default_model") or ""
                provider_type = provider_entry.get("provider_type", "")
                base_url = provider_entry.get("base_url", "")

                # For local providers (LM Studio/Ollama), fetch the loaded model
                is_local = provider_type in ["LM Studio (Local)", "Ollama (Local)"]
                local_models = []
                if is_local and base_url:
                    from core.llm_client import fetch_local_models
                    local_models = fetch_local_models(base_url)
                    if local_models and not default_model:
                        default_model = local_models[0]  # Use the first/loaded model

                # Use default model if current is empty
                current_model = st.session_state.get("aic_ai_model") or default_model

                # If provider changed, reset to provider's default model
                if st.session_state.get("aic_last_provider") != selected_provider:
                    current_model = default_model
                    st.session_state["aic_last_provider"] = selected_provider

                if is_local and local_models:
                    # Show dropdown with available models
                    selected_idx = 0
                    if current_model in local_models:
                        selected_idx = local_models.index(current_model)
                    model = st.selectbox(
                        "Model (from LM Studio)",
                        options=local_models,
                        index=selected_idx,
                        key="aic_model_select"
                    )
                elif is_local:
                    st.warning("Could not connect to LM Studio. Make sure it's running.")
                    model = st.text_input(
                        "Model",
                        value=current_model,
                        key="aic_model_input",
                        placeholder="Enter model name manually"
                    )
                else:
                    model = st.text_input(
                        "Model",
                        value=current_model,
                        key="aic_model_input",
                        placeholder="Enter model name"
                    )

                # Save the model
                st.session_state["aic_ai_model"] = model if model else default_model

            st.markdown("**Processing Mode**")
            ai_mode = st.radio(
                "Processing Mode",
                options=["per_row", "batch"],
                format_func=lambda x: "Per-row - AI suggests on navigation" if x == "per_row" else "Batch - Pre-process all rows",
                index=0 if st.session_state.get("aic_ai_mode", "per_row") == "per_row" else 1,
                horizontal=True,
                key="aic_mode_radio",
                label_visibility="collapsed"
            )
            st.session_state["aic_ai_mode"] = ai_mode

            st.markdown("**Display Mode**")
            display_options = {
                "ai_first": "AI First - Show suggestions above codes",
                "inline_badges": "Inline badges - Confidence on buttons"
            }
            current_display = st.session_state.get("aic_ai_display", "ai_first")
            ai_display = st.radio(
                "Display Mode",
                options=list(display_options.keys()),
                format_func=lambda x: display_options[x],
                index=list(display_options.keys()).index(current_display),
                horizontal=True,
                key="aic_display_radio",
                label_visibility="collapsed"
            )
            st.session_state["aic_ai_display"] = ai_display

            # AI Behavior Settings
            with st.expander("AI Behavior", expanded=False):
                # AI Context Window
                ai_context = st.slider(
                    "AI context window (surrounding rows)",
                    min_value=0, max_value=3,
                    value=st.session_state.get("aic_ai_context_rows", 0),
                    key="aic_ai_context_slider",
                    help="Number of rows before/after to include in AI prompt for context"
                )
                st.session_state["aic_ai_context_rows"] = ai_context
                if ai_context > 0:
                    st.caption(f"AI will see {ai_context} row(s) before and after each text.")

                st.divider()

                # Training Mode
                st.markdown("**Training Mode (Few-Shot Learning)**")
                training_enabled = st.checkbox(
                    "Enable training examples",
                    value=st.session_state.get("aic_training_enabled", False),
                    key="aic_training_checkbox",
                    help="Include your manually coded examples in the AI prompt"
                )
                st.session_state["aic_training_enabled"] = training_enabled

                if training_enabled:
                    training_count = st.slider(
                        "Number of examples",
                        min_value=1, max_value=10,
                        value=st.session_state.get("aic_training_examples_count", 5),
                        key="aic_training_count_slider",
                        help="How many coded examples to include in the prompt"
                    )
                    st.session_state["aic_training_examples_count"] = training_count

                    coded_count = self._count_coded_rows()
                    if coded_count > 0:
                        st.caption(f"✓ {coded_count} coded rows available as training examples")
                    else:
                        st.caption("⚠️ Code some rows first to use as training examples")

            # Confidence Thresholds
            with st.expander("Confidence Thresholds", expanded=False):
                thresholds_enabled = st.checkbox(
                    "Enable automatic threshold actions",
                    value=st.session_state.get("aic_thresholds_enabled", False),
                    key="aic_thresholds_checkbox",
                    help="Automatically accept high-confidence, flag low-confidence, skip very low."
                )
                st.session_state["aic_thresholds_enabled"] = thresholds_enabled

                if thresholds_enabled:
                    thresh_col1, thresh_col2, thresh_col3 = st.columns(3)
                    with thresh_col1:
                        auto_accept = st.slider(
                            "Auto-accept ≥",
                            min_value=0.0, max_value=1.0,
                            value=st.session_state.get("aic_threshold_auto_accept", 0.90),
                            step=0.05,
                            key="aic_thresh_auto",
                            help="Auto-apply codes with confidence ≥ this value"
                        )
                        st.session_state["aic_threshold_auto_accept"] = auto_accept
                    with thresh_col2:
                        flag_thresh = st.slider(
                            "Flag <",
                            min_value=0.0, max_value=1.0,
                            value=st.session_state.get("aic_threshold_flag", 0.50),
                            step=0.05,
                            key="aic_thresh_flag",
                            help="Flag rows with confidence < this value for review"
                        )
                        st.session_state["aic_threshold_flag"] = flag_thresh
                    with thresh_col3:
                        skip_thresh = st.slider(
                            "Skip <",
                            min_value=0.0, max_value=1.0,
                            value=st.session_state.get("aic_threshold_skip", 0.20),
                            step=0.05,
                            key="aic_thresh_skip",
                            help="Skip suggestions with confidence < this value"
                        )
                        st.session_state["aic_threshold_skip"] = skip_thresh

                    # Show threshold stats
                    flagged = len(st.session_state.get("aic_flagged_rows", set()))
                    auto_accepted = len(st.session_state.get("aic_auto_accepted_rows", set()))
                    if flagged > 0 or auto_accepted > 0:
                        st.caption(f"📊 {auto_accepted} auto-accepted, {flagged} flagged for review")

            # Custom Prompt & Code Definitions
            with st.expander("Prompt Customization", expanded=False):
                # Custom System Prompt
                st.markdown("**Custom System Prompt**")
                custom_prompt_enabled = st.checkbox(
                    "Enable custom prompt",
                    value=st.session_state.get("aic_custom_prompt_enabled", False),
                    key="aic_custom_prompt_checkbox",
                    help="Use a custom prompt instead of the default. Use {codes_list} and {training_examples} as placeholders."
                )
                st.session_state["aic_custom_prompt_enabled"] = custom_prompt_enabled

                if custom_prompt_enabled:
                    custom_prompt = st.text_area(
                        "System prompt",
                        value=st.session_state.get("aic_custom_prompt", ""),
                        height=150,
                        placeholder="Enter your custom prompt. Use {codes_list} for code list and {training_examples} for training examples.",
                        key="aic_custom_prompt_textarea"
                    )
                    st.session_state["aic_custom_prompt"] = custom_prompt
                    st.caption("Placeholders: `{codes_list}`, `{training_examples}`")

                st.divider()

                # Code Definitions
                st.markdown("**Code Definitions**")
                st.caption("Add definitions, examples, and keywords for each code to improve AI accuracy.")

                code_definitions = st.session_state.get("aic_code_definitions", {})
                for code in codes:
                    with st.expander(f"📝 {code}", expanded=False):
                        if code not in code_definitions:
                            code_definitions[code] = {"definition": "", "examples": "", "keywords": ""}

                        code_definitions[code]["definition"] = st.text_input(
                            "Definition",
                            value=code_definitions[code].get("definition", ""),
                            key=f"aic_def_{code}",
                            placeholder=f"What does '{code}' mean?"
                        )
                        code_definitions[code]["examples"] = st.text_input(
                            "Examples",
                            value=code_definitions[code].get("examples", ""),
                            key=f"aic_ex_{code}",
                            placeholder="Example texts that should get this code"
                        )
                        code_definitions[code]["keywords"] = st.text_input(
                            "Keywords",
                            value=code_definitions[code].get("keywords", ""),
                            key=f"aic_kw_{code}",
                            placeholder="Keywords that indicate this code (comma-separated)"
                        )

                st.session_state["aic_code_definitions"] = code_definitions

            # Batch mode controls
            if ai_mode == "batch":
                col_batch, col_progress = st.columns([1, 2])
                with col_batch:
                    batch_status = st.session_state.get("aic_ai_batch_status", "idle")
                    if batch_status == "running":
                        st.button("Running...", disabled=True, key="aic_batch_btn")
                    elif batch_status == "complete":
                        suggestions = st.session_state.get("aic_ai_suggestions", {})
                        success_count = sum(1 for s in suggestions.values() if s and not s.get("error"))
                        error_count = sum(1 for s in suggestions.values() if s and s.get("error"))
                        if error_count > 0:
                            st.warning(f"Batch complete: {success_count} success, {error_count} errors")
                        else:
                            st.success(f"Batch complete! {success_count} rows processed")
                        if st.button("Re-run Batch", key="aic_rerun_batch"):
                            st.session_state["aic_ai_suggestions"] = {}
                            st.session_state["aic_ai_batch_status"] = "idle"
                            st.rerun()
                    else:
                        if st.button("Run AI Batch", type="primary", key="aic_run_batch"):
                            # Process batch directly in config (no need to wait for render_results)
                            provider = st.session_state.get("aic_ai_provider", "")
                            model = st.session_state.get("aic_ai_model", "")

                            if not provider:
                                st.error("No AI provider selected! Please select a provider above.")
                            elif not model:
                                st.error("No AI model selected! Please enter a model name above.")
                            elif df is None or df.empty:
                                st.error("No data loaded! Please load data in step 1.")
                            elif not text_cols:
                                st.error("No text columns selected! Please select columns in step 3.")
                            else:
                                st.info(f"Starting batch with provider: {provider}, model: {model}")
                                with st.spinner(f"Processing {len(df)} rows with {provider}..."):
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                    try:
                                        loop.run_until_complete(self._process_batch(df, text_cols))
                                    except Exception as e:
                                        st.error(f"Batch processing error: {str(e)}")
                                    finally:
                                        loop.close()
                                st.rerun()
                with col_progress:
                    if batch_status == "running":
                        progress = st.session_state.get("aic_batch_progress", 0)
                        st.progress(progress)
                        st.caption(f"Processing... {int(progress * 100)}%")
                    elif batch_status == "complete":
                        st.progress(1.0)
                        # Show debug info
                        debug_info = st.session_state.get("aic_batch_debug", "")
                        if debug_info:
                            st.caption(f"Debug: {debug_info}")
                        # Show first error if any
                        first_error = st.session_state.get("aic_batch_first_error", "")
                        if first_error:
                            st.error(f"First error: {first_error[:200]}")

        return ToolConfig(
            is_valid=True,
            config_data={
                "df": df,
                "codes": codes,
                "text_cols": text_cols
            }
        )

    def _show_immersive_dialog(self, df, codes, text_cols, total_rows):
        """Show immersive coding as a dialog with AI suggestions"""
        @st.dialog("AI-Assisted Coding", width="large")
        def immersive_dialog():
            current_row = st.session_state["aic_current_row"]
            light_mode = st.session_state.get("aic_light_mode", True)
            context_rows = st.session_state.get("aic_context_rows", 2)
            ai_display = st.session_state.get("aic_ai_display", "ai_first")

            # Header row
            head_col1, head_col2, head_col3 = st.columns([2, 1, 1])
            with head_col1:
                coded_count = self._count_coded_rows()
                progress_pct = coded_count / total_rows if total_rows > 0 else 0
                st.markdown(f"**Row {current_row + 1}/{total_rows}** | {coded_count} coded ({progress_pct:.0%})")
            with head_col2:
                if st.button("Save", key="aic_imm_save", use_container_width=True):
                    st.session_state["aic_imm_save_requested"] = True
                    st.rerun()
            with head_col3:
                if st.button("Close", key="aic_imm_exit", use_container_width=True):
                    st.session_state["aic_imm_close_requested"] = True
                    st.rerun()

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

            # Show threshold status indicators
            is_flagged = current_row in st.session_state.get("aic_flagged_rows", set())
            is_auto_accepted = current_row in st.session_state.get("aic_auto_accepted_rows", set())

            if is_auto_accepted:
                st.success("✓ Auto-accepted (high confidence)")
            elif is_flagged:
                st.warning("⚠️ Flagged for review (low confidence)")

            # AI Suggestion display (if ai_first mode)
            suggestion = st.session_state.get("aic_ai_suggestions", {}).get(current_row, {})
            if ai_display == "ai_first" and suggestion and suggestion.get("codes"):
                st.markdown("**AI Suggestions:**")
                sugg_cols = st.columns(len(suggestion["codes"]) + 1)
                for i, code in enumerate(suggestion["codes"]):
                    conf = suggestion.get("confidence", {}).get(code, 0)
                    with sugg_cols[i]:
                        color = self._get_code_color(code)
                        st.markdown(
                            f'<div style="background-color: {color}; padding: 6px 10px; '
                            f'border-radius: 5px; text-align: center;">'
                            f'{code}<br><small>{conf*100:.0f}%</small></div>',
                            unsafe_allow_html=True
                        )
                with sugg_cols[-1]:
                    if st.button("Accept All", key="aic_imm_accept_all", use_container_width=True):
                        self._accept_ai_suggestions(current_row)
                        # Move to next row
                        if current_row < total_rows - 1:
                            st.session_state["aic_current_row"] = current_row + 1
                        st.rerun()
                if suggestion.get("reasoning"):
                    st.caption(f"Reasoning: {suggestion['reasoning']}")
                st.divider()

            # Text display
            start_row = max(0, current_row - context_rows)
            end_row = min(total_rows, current_row + context_rows + 1)

            text_html = []
            for row_idx in range(start_row, end_row):
                is_current = row_idx == current_row
                text_content = self._get_row_text_html(df, text_cols, row_idx)
                highlighted_text = self._highlight_text(text_content)
                row_codes = self._get_applied_codes(row_idx)

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
                        f'<strong>Row {row_idx + 1}</strong> {badges}<br/>'
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

            st.markdown(
                f'<div style="min-height: 200px; max-height: 300px; overflow-y: auto; '
                f'padding: 5px; margin: 5px 0;">{"".join(text_html)}</div>',
                unsafe_allow_html=True
            )

            # Code buttons (with inline badges if that mode is selected)
            def add_code_immersive(row_idx, code):
                self._add_code(row_idx, code)

            num_codes = len(codes)
            if num_codes > 0:
                code_cols = st.columns(num_codes)
                for i, code in enumerate(codes):
                    color = self._get_code_color(code)
                    with code_cols[i]:
                        st.markdown(f'<div style="background:{color}; height:4px; border-radius:2px; margin-bottom:2px;"></div>', unsafe_allow_html=True)

                        # Inline badge mode
                        label = code
                        if ai_display == "inline_badges" and suggestion:
                            conf = suggestion.get("confidence", {}).get(code, 0)
                            if conf > 0:
                                label = f"{code} ({conf*100:.0f}%)"

                        st.button(
                            label,
                            key=f"aic_imm_code_{current_row}_{code}",
                            use_container_width=True,
                            on_click=add_code_immersive,
                            args=(current_row, code)
                        )

            # Navigation
            is_disabled_prev = current_row <= 0
            is_disabled_next = current_row >= total_rows - 1

            def go_prev5():
                st.session_state["aic_current_row"] = max(0, st.session_state["aic_current_row"] - 5)

            def go_prev():
                st.session_state["aic_current_row"] = max(0, st.session_state["aic_current_row"] - 1)

            def go_next():
                st.session_state["aic_current_row"] = min(total_rows - 1, st.session_state["aic_current_row"] + 1)
                self._save_progress()
                # Trigger AI suggestion for next row in per_row mode
                if st.session_state.get("aic_ai_mode") == "per_row":
                    st.session_state["aic_fetch_suggestion"] = True

            def go_next5():
                st.session_state["aic_current_row"] = min(total_rows - 1, st.session_state["aic_current_row"] + 5)

            st.button("Next", disabled=is_disabled_next, key="aic_imm_next_big", type="primary", use_container_width=True, on_click=go_next)

            nav1, nav2, nav3, nav4 = st.columns([1, 1, 1, 1])
            with nav1:
                st.button("<<", disabled=is_disabled_prev, key="aic_imm_prev5", use_container_width=True, on_click=go_prev5)
            with nav2:
                st.button("< Prev", disabled=is_disabled_prev, key="aic_imm_prev", use_container_width=True, on_click=go_prev)
            with nav3:
                st.button("Next >", disabled=is_disabled_next, key="aic_imm_next", use_container_width=True, on_click=go_next)
            with nav4:
                st.button(">>", disabled=is_disabled_next, key="aic_imm_next5", use_container_width=True, on_click=go_next5)

            # Applied codes section
            def remove_code_immersive(row_idx, position):
                self._remove_code_at(row_idx, position)

            applied_codes = self._get_applied_codes(current_row)
            st.markdown('<div style="min-height: 50px;">', unsafe_allow_html=True)
            if applied_codes:
                app_cols = st.columns(min(len(applied_codes), 6))
                for i, code in enumerate(applied_codes[:6]):
                    color = self._get_code_color(code)
                    with app_cols[i]:
                        st.markdown(f'<span style="background-color: {color}; padding: 2px 6px; border-radius: 3px; font-size: 0.85em;">{code}</span>', unsafe_allow_html=True)
                        st.button("x", key=f"aic_imm_rm_{current_row}_{i}", on_click=remove_code_immersive, args=(current_row, i))
            st.markdown('</div>', unsafe_allow_html=True)

            # Progress bar
            coded_count = self._count_coded_rows()
            progress = coded_count / total_rows if total_rows > 0 else 0
            st.progress(progress)

        immersive_dialog()

    async def execute(
        self,
        config: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> ToolResult:
        """Execute is not used for manual coding - all work happens in render_results."""
        return ToolResult(
            success=True,
            data=config["df"],
            stats={
                "total_rows": len(config["df"]),
                "codes_defined": len(config["codes"])
            }
        )

    def render_results(self, result: ToolResult):
        """Render the AI-assisted coding interface"""
        from ui.components.download_buttons import render_download_buttons

        if not result.success:
            st.error(f"Error: {result.error_message}")
            return

        df = st.session_state["aic_df"]
        codes = st.session_state["aic_codes"]
        text_cols = st.session_state.get("aic_text_cols", [])
        total_rows = len(df)

        # Handle immersive dialog button actions
        if st.session_state.get("aic_imm_close_requested"):
            st.session_state["aic_imm_close_requested"] = False
            st.session_state["aic_immersive_active"] = False
            st.session_state["aic_immersive_trigger"] = False
            st.rerun()

        if st.session_state.get("aic_imm_save_requested"):
            st.session_state["aic_imm_save_requested"] = False
            if not st.session_state.get("aic_current_session"):
                st.session_state["aic_current_session"] = self._generate_session_name()
            self._save_session()
            st.toast("Session saved")

        # Handle manual AI suggestion request
        force_fetch = st.session_state.pop("aic_force_fetch", False)

        # Fetch AI suggestion for current row if in per_row mode
        if st.session_state.get("aic_ai_mode") == "per_row" and st.session_state.get("aic_ai_provider"):
            current_row = st.session_state["aic_current_row"]
            should_fetch = force_fetch or (current_row not in st.session_state.get("aic_ai_suggestions", {}))
            if should_fetch:
                text = self._get_row_text(df, text_cols, current_row)
                with st.spinner(f"Getting AI suggestion for row {current_row + 1}..."):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(self._get_ai_suggestion(current_row, text))
                        if result and result.get("error"):
                            st.warning(f"AI: {result.get('reasoning', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"AI Error: {str(e)}")
                    finally:
                        loop.close()

        # Session management bar
        session_col1, session_col2, session_col3, session_col4, session_col5 = st.columns([2, 1, 1, 1, 1])
        with session_col1:
            current_session = st.session_state.get("aic_current_session")
            coded_count = self._count_coded_rows()
            ai_count = len(st.session_state.get("aic_ai_suggestions", {}))

            # Build session info string
            session_info = f"{coded_count}/{total_rows} coded, {ai_count} AI"

            # Add threshold stats if enabled
            if st.session_state.get("aic_thresholds_enabled", False):
                flagged_count = len(st.session_state.get("aic_flagged_rows", set()))
                auto_accepted_count = len(st.session_state.get("aic_auto_accepted_rows", set()))
                if flagged_count > 0 or auto_accepted_count > 0:
                    session_info += f" | ✓{auto_accepted_count} ⚠️{flagged_count}"

            if current_session:
                st.markdown(f"**Session:** `{current_session}` ({session_info})")
            else:
                st.markdown(f"**New Session** ({session_info})")
        with session_col2:
            if st.button("Save", key="aic_save_session", use_container_width=True):
                st.session_state["aic_show_save_dialog"] = True
        with session_col3:
            if st.button("Load", key="aic_show_load", use_container_width=True):
                st.session_state["aic_show_sessions"] = not st.session_state.get("aic_show_sessions", False)
                st.session_state["aic_show_save_dialog"] = False
        with session_col4:
            if st.button("Analytics", key="aic_show_analytics", use_container_width=True):
                st.session_state["aic_show_analytics_panel"] = not st.session_state.get("aic_show_analytics_panel", False)
        with session_col5:
            pass  # Immersive mode disabled

        # Save dialog
        if st.session_state.get("aic_show_save_dialog"):
            save_col1, save_col2, save_col3 = st.columns([3, 1, 1])
            with save_col1:
                default_name = st.session_state.get("aic_current_session") or self._generate_session_name()
                session_name = st.text_input("Session name", value=default_name, key="aic_session_name_input", label_visibility="collapsed")
            with save_col2:
                if st.button("Save", key="aic_do_save", type="primary", use_container_width=True):
                    if session_name.strip():
                        st.session_state["aic_current_session"] = session_name.strip()
                        self._save_session(session_name.strip())
                        st.session_state["aic_show_save_dialog"] = False
                        st.toast(f"Saved: {session_name.strip()}")
                        st.rerun()
            with save_col3:
                if st.button("Cancel", key="aic_cancel_save", use_container_width=True):
                    st.session_state["aic_show_save_dialog"] = False
                    st.rerun()

        # Session browser
        if st.session_state.get("aic_show_sessions"):
            sessions = self._list_sessions()
            if sessions:
                st.markdown("---")
                st.markdown("##### Saved Sessions")
                for sess in sessions[:10]:
                    sess_name = sess['name']
                    s_col1, s_col2, s_col3, s_col4 = st.columns([3, 1, 1, 1])
                    with s_col1:
                        saved_time = sess.get("saved_at", "")[:16].replace("T", " ")
                        ai_badge = " [AI]" if sess.get("has_ai") else ""
                        st.markdown(f"`{sess_name}` - {sess['coded_count']}/{sess['total_rows']} coded{ai_badge} ({saved_time})")
                    with s_col2:
                        if st.button("Load", key=f"aic_load_{sess_name}", use_container_width=True):
                            if self._load_session(sess_name):
                                st.session_state["aic_show_sessions"] = False
                                st.toast(f"Loaded: {sess_name}")
                                st.rerun()
                    with s_col3:
                        if st.button("Rename", key=f"aic_rename_btn_{sess_name}", use_container_width=True):
                            st.session_state["aic_renaming_session"] = sess_name
                    with s_col4:
                        if st.button("Delete", key=f"aic_del_{sess_name}", use_container_width=True):
                            self._delete_session(sess_name)
                            st.rerun()
                    if st.session_state.get("aic_renaming_session") == sess_name:
                        r_col1, r_col2, r_col3 = st.columns([3, 1, 1])
                        with r_col1:
                            new_name = st.text_input("New name", value=sess_name, key=f"aic_rename_input_{sess_name}", label_visibility="collapsed")
                        with r_col2:
                            if st.button("Save", key=f"aic_rename_save_{sess_name}", type="primary", use_container_width=True):
                                if new_name.strip() and new_name.strip() != sess_name:
                                    if self._rename_session(sess_name, new_name.strip()):
                                        st.session_state["aic_renaming_session"] = None
                                        st.toast(f"Renamed to: {new_name.strip()}")
                                        st.rerun()
                                    else:
                                        st.toast("Rename failed")
                                else:
                                    st.session_state["aic_renaming_session"] = None
                                    st.rerun()
                        with r_col3:
                            if st.button("Cancel", key=f"aic_rename_cancel_{sess_name}", use_container_width=True):
                                st.session_state["aic_renaming_session"] = None
                                st.rerun()
                st.markdown("---")
            else:
                st.info("No saved sessions found")
                st.session_state["aic_show_sessions"] = False

        # Immersive mode
        trigger = st.session_state.get("aic_immersive_trigger", False)
        active = st.session_state.get("aic_immersive_active", False)

        if trigger or active:
            st.session_state["aic_immersive_trigger"] = False
            st.session_state["aic_immersive_active"] = True
            self._show_immersive_dialog(df, codes, text_cols, total_rows)

        # Word highlighter
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
                        value=st.session_state.get("aic_highlights", {}).get(code, ""),
                        key=f"aic_hl_{code}",
                        placeholder="yes, agree, ok",
                        label_visibility="collapsed"
                    )
                    if "aic_highlights" not in st.session_state:
                        st.session_state["aic_highlights"] = {}
                    st.session_state["aic_highlights"][code] = words

        # Main coding interface
        @st.fragment
        def coding_interface():
            current_row = st.session_state["aic_current_row"]
            context_rows = st.session_state.get("aic_context_rows", 2)
            light_mode = st.session_state.get("aic_light_mode", True)
            ai_display = st.session_state.get("aic_ai_display", "ai_first")

            # Options row
            opt_col1, opt_col2, opt_col3, opt_col4, opt_col5 = st.columns([1, 1, 1, 1, 1])
            with opt_col1:
                new_light_mode = st.toggle("Light mode", value=light_mode, key="aic_theme_toggle")
                if new_light_mode != light_mode:
                    st.session_state["aic_light_mode"] = new_light_mode
                    st.rerun()
            with opt_col2:
                old_horiz = st.session_state.get("aic_horizontal_codes", False)
                new_horiz = st.toggle("Horizontal codes", value=old_horiz, key="aic_horiz_top")
                if new_horiz != old_horiz:
                    st.session_state["aic_horizontal_codes"] = new_horiz
                    st.rerun()
            with opt_col3:
                old_above = st.session_state.get("aic_buttons_above", False)
                new_above = st.toggle("Buttons above text", value=old_above, key="aic_above_toggle")
                if new_above != old_above:
                    st.session_state["aic_buttons_above"] = new_above
                    st.rerun()
            with opt_col4:
                auto_adv = st.toggle("Auto-advance", value=st.session_state.get("aic_auto_advance", False), key="aic_auto")
                st.session_state["aic_auto_advance"] = auto_adv
            with opt_col5:
                ctx = st.number_input("Context", min_value=0, max_value=5,
                                     value=st.session_state.get("aic_context_rows", 2), key="aic_ctx")
                st.session_state["aic_context_rows"] = ctx

            # Colors
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

            horizontal_mode = st.session_state.get("aic_horizontal_codes", False)
            buttons_above = st.session_state.get("aic_buttons_above", False)

            # Get AI suggestion for current row
            suggestion = st.session_state.get("aic_ai_suggestions", {}).get(current_row, {})
            has_provider = bool(st.session_state.get("aic_ai_provider"))

            # Show threshold status indicators
            is_flagged = current_row in st.session_state.get("aic_flagged_rows", set())
            is_auto_accepted = current_row in st.session_state.get("aic_auto_accepted_rows", set())

            if is_auto_accepted:
                st.success("✓ Auto-accepted (high confidence)")
            elif is_flagged:
                st.warning("⚠️ Flagged for review (low confidence)")

            # AI First display mode - show suggestions before text
            if ai_display == "ai_first":
                if suggestion and suggestion.get("codes") and not suggestion.get("error"):
                    st.markdown("**AI Suggestions:**")
                    sugg_cols = st.columns(min(len(suggestion["codes"]) + 1, 6))
                    for i, code in enumerate(suggestion["codes"][:5]):
                        conf = suggestion.get("confidence", {}).get(code, 0)
                        with sugg_cols[i]:
                            color = self._get_code_color(code)
                            st.markdown(
                                f'<div style="background-color: {color}; padding: 8px 12px; '
                                f'border-radius: 5px; text-align: center; margin-bottom: 5px;">'
                                f'<strong>{code}</strong><br><small>{conf*100:.0f}%</small></div>',
                                unsafe_allow_html=True
                            )
                    with sugg_cols[-1]:
                        if st.button("Accept All", key="aic_accept_all", use_container_width=True):
                            self._accept_ai_suggestions(current_row)
                            # Move to next row
                            if current_row < total_rows - 1:
                                st.session_state["aic_current_row"] = current_row + 1
                            st.rerun()
                    if suggestion.get("reasoning"):
                        st.caption(f"Reasoning: {suggestion['reasoning']}")
                    st.divider()
                elif suggestion and suggestion.get("error"):
                    st.warning(f"AI Error: {suggestion.get('reasoning', 'Unknown error')}")
                    if st.button("Retry AI Suggestion", key="aic_retry_ai"):
                        # Clear the cached error and retry
                        if current_row in st.session_state.get("aic_ai_suggestions", {}):
                            del st.session_state["aic_ai_suggestions"][current_row]
                        st.rerun()
                    st.divider()
                elif has_provider and st.session_state.get("aic_ai_mode") == "per_row":
                    ai_col1, ai_col2 = st.columns([3, 1])
                    with ai_col1:
                        st.info("No AI suggestion yet for this row")
                    with ai_col2:
                        if st.button("Get AI Suggestion", key="aic_get_suggestion", type="primary"):
                            st.session_state["aic_force_fetch"] = True
                            st.rerun()
                    st.divider()

            def render_text_display():
                start_row = max(0, current_row - context_rows)
                end_row = min(total_rows, current_row + context_rows + 1)

                text_container_html = []
                for row_idx in range(start_row, end_row):
                    is_current = row_idx == current_row
                    text_content = self._get_row_text_html(df, text_cols, row_idx)
                    highlighted_text = self._highlight_text(text_content)

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
                            f'<strong>Row {row_idx + 1} of {total_rows}</strong>{codes_badge}<br/>'
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

                st.markdown(
                    f'<div style="min-height: 300px; padding-right: 10px;">{"".join(text_container_html)}</div>',
                    unsafe_allow_html=True
                )

            def render_horizontal_controls():
                is_disabled = current_row >= total_rows - 1

                num_codes = len(codes)
                if num_codes > 0:
                    code_cols = st.columns(num_codes)
                    for i, code in enumerate(codes):
                        color = self._get_code_color(code)
                        with code_cols[i]:
                            st.markdown(f'<div style="background:{color}; height:4px; border-radius:2px; margin-bottom:2px;"></div>', unsafe_allow_html=True)

                            # Inline badge mode
                            label = code
                            if ai_display == "inline_badges" and suggestion:
                                conf = suggestion.get("confidence", {}).get(code, 0)
                                if conf > 0:
                                    label = f"{code} ({conf*100:.0f}%)"

                            st.button(
                                label,
                                key=f"aic_code_{current_row}_{code}",
                                type="secondary",
                                use_container_width=True,
                                on_click=self._add_code,
                                args=(current_row, code)
                            )

                if st.button("Next", key="aic_next_horiz", type="primary", use_container_width=True, disabled=is_disabled):
                    st.session_state["aic_current_row"] = min(total_rows - 1, current_row + 1)
                    if st.session_state.get("aic_autosave_enabled", True):
                        self._save_progress()
                    st.rerun()

                def go_prev():
                    st.session_state["aic_current_row"] = max(0, st.session_state["aic_current_row"] - 1)

                def go_next():
                    st.session_state["aic_current_row"] = min(total_rows - 1, st.session_state["aic_current_row"] + 1)

                nav1, nav2, nav3, nav4 = st.columns([1, 1, 1, 1])
                with nav1:
                    st.button("<<", disabled=current_row <= 0, key="aic_prev5_h",
                              on_click=lambda: st.session_state.update({"aic_current_row": max(0, current_row - 5)}))
                with nav2:
                    st.button("< Prev", disabled=current_row <= 0, key="aic_prev_h", on_click=go_prev)
                with nav3:
                    st.button("Next >", disabled=current_row >= total_rows - 1, key="aic_next_h", on_click=go_next)
                with nav4:
                    st.button(">>", disabled=current_row >= total_rows - 1, key="aic_next5_h",
                              on_click=lambda: st.session_state.update({"aic_current_row": min(total_rows - 1, current_row + 5)}))

            def render_applied_and_progress():
                applied_codes = self._get_applied_codes(current_row)

                if applied_codes:
                    st.markdown("##### Applied")
                    for i, code in enumerate(applied_codes):
                        color = self._get_code_color(code)
                        col_up, col_down, col_code, col_remove = st.columns([1, 1, 10, 1])
                        with col_up:
                            st.button("^", key=f"aic_up_{current_row}_{i}", disabled=i == 0,
                                      on_click=self._move_code_up, args=(current_row, i))
                        with col_down:
                            st.button("v", key=f"aic_down_{current_row}_{i}", disabled=i == len(applied_codes) - 1,
                                      on_click=self._move_code_down, args=(current_row, i))
                        with col_code:
                            st.markdown(
                                f'<span style="background-color: {color}; padding: 2px 6px; '
                                f'border-radius: 3px;">{i + 1}. {code}</span>',
                                unsafe_allow_html=True
                            )
                        with col_remove:
                            st.button("x", key=f"aic_remove_{current_row}_{i}",
                                      on_click=self._remove_code_at, args=(current_row, i))

                coded_count = self._count_coded_rows()
                progress = coded_count / total_rows if total_rows > 0 else 0
                st.caption(f"**{coded_count}/{total_rows}** coded")
                st.progress(progress)

            if horizontal_mode:
                if buttons_above:
                    render_horizontal_controls()
                    render_text_display()
                else:
                    render_text_display()
                    render_horizontal_controls()
                render_applied_and_progress()
            else:
                # Vertical layout
                col_text, col_codes = st.columns([5, 1])
                with col_text:
                    render_text_display()
                with col_codes:
                    st.markdown("##### Codes")

                    applied_codes = self._get_applied_codes(current_row)

                    is_disabled = current_row >= total_rows - 1
                    if st.button("Next", key="aic_next_top", type="primary", use_container_width=True, disabled=is_disabled):
                        st.session_state["aic_current_row"] = min(total_rows - 1, current_row + 1)
                        if st.session_state.get("aic_autosave_enabled", True):
                            self._save_progress()
                        st.rerun()

                    for i, code in enumerate(codes):
                        color = self._get_code_color(code)
                        st.markdown(f'<div style="background:{color}; height:4px; border-radius:2px; margin-bottom:2px;"></div>', unsafe_allow_html=True)

                        label = code
                        if ai_display == "inline_badges" and suggestion:
                            conf = suggestion.get("confidence", {}).get(code, 0)
                            if conf > 0:
                                label = f"{code} ({conf*100:.0f}%)"

                        st.button(
                            label,
                            key=f"aic_code_{current_row}_{code}",
                            type="secondary",
                            use_container_width=True,
                            on_click=self._add_code,
                            args=(current_row, code)
                        )

                    if st.button("Next", key="aic_next_bottom", type="primary", use_container_width=True, disabled=is_disabled):
                        st.session_state["aic_current_row"] = min(total_rows - 1, current_row + 1)
                        if st.session_state.get("aic_autosave_enabled", True):
                            self._save_progress()
                        st.rerun()

                    def go_prev_v():
                        st.session_state["aic_current_row"] = max(0, st.session_state["aic_current_row"] - 1)

                    def go_next_v():
                        st.session_state["aic_current_row"] = min(total_rows - 1, st.session_state["aic_current_row"] + 1)

                    nav1, nav2, nav3, nav4 = st.columns([1, 1, 1, 1])
                    with nav1:
                        st.button("<<", disabled=current_row <= 0, key="aic_prev5_v",
                                  on_click=lambda: st.session_state.update({"aic_current_row": max(0, current_row - 5)}))
                    with nav2:
                        st.button("< Prev", disabled=current_row <= 0, key="aic_prev_v", on_click=go_prev_v)
                    with nav3:
                        st.button("Next >", disabled=current_row >= total_rows - 1, key="aic_next_v", on_click=go_next_v)
                    with nav4:
                        st.button(">>", disabled=current_row >= total_rows - 1, key="aic_next5_v",
                                  on_click=lambda: st.session_state.update({"aic_current_row": min(total_rows - 1, current_row + 5)}))

                    if applied_codes:
                        st.markdown("---")
                        st.markdown("##### Applied")
                        for i, code in enumerate(applied_codes):
                            color = self._get_code_color(code)
                            col_up, col_down, col_code, col_remove = st.columns([1, 1, 6, 1])
                            with col_up:
                                st.button("^", key=f"aic_up_{current_row}_{i}", disabled=i == 0,
                                          on_click=self._move_code_up, args=(current_row, i))
                            with col_down:
                                st.button("v", key=f"aic_down_{current_row}_{i}", disabled=i == len(applied_codes) - 1,
                                          on_click=self._move_code_down, args=(current_row, i))
                            with col_code:
                                st.markdown(
                                    f'<span style="background-color: {color}; padding: 2px 6px; '
                                    f'border-radius: 3px;">{i + 1}. {code}</span>',
                                    unsafe_allow_html=True
                                )
                            with col_remove:
                                st.button("x", key=f"aic_remove_{current_row}_{i}",
                                          on_click=self._remove_code_at, args=(current_row, i))

                    st.markdown("---")
                    coded_count = self._count_coded_rows()
                    progress = coded_count / total_rows if total_rows > 0 else 0
                    st.caption(f"**{coded_count}/{total_rows}** coded")
                    st.progress(progress)

        coding_interface()

        # Analytics panel (before export)
        if st.session_state.get("aic_show_analytics_panel"):
            self._render_analytics_panel(df, codes, total_rows)

        # Export section
        st.divider()
        st.subheader("Export Results")

        coded_count = self._count_coded_rows()
        if coded_count == 0:
            st.info("Code some rows before exporting")
        else:
            export_format = st.radio(
                "Export format",
                ["Standard (human codes only)", "With AI (human + AI codes)", "Full Analytics (data + JSON report)"],
                horizontal=True,
                key="aic_export_format"
            )

            export_df = df.copy()

            if export_format == "Standard (human codes only)":
                export_df["applied_codes"] = export_df.index.map(
                    lambda idx: "; ".join(st.session_state["aic_coding_data"].get(idx, []))
                )
                export_df["code_count"] = export_df.index.map(
                    lambda idx: len(st.session_state["aic_coding_data"].get(idx, []))
                )

            elif export_format == "With AI (human + AI codes)":
                export_df["human_codes"] = export_df.index.map(
                    lambda idx: "; ".join(st.session_state["aic_coding_data"].get(idx, []))
                )
                export_df["ai_codes"] = export_df.index.map(
                    lambda idx: "; ".join(st.session_state.get("aic_ai_suggestions", {}).get(idx, {}).get("codes", []))
                )
                export_df["ai_confidence"] = export_df.index.map(
                    lambda idx: json.dumps(st.session_state.get("aic_ai_suggestions", {}).get(idx, {}).get("confidence", {}))
                )
                export_df["agreement"] = export_df.index.map(
                    lambda idx: set(st.session_state["aic_coding_data"].get(idx, [])) ==
                               set(st.session_state.get("aic_ai_suggestions", {}).get(idx, {}).get("codes", []))
                )

            else:  # Full Analytics
                export_df["human_codes"] = export_df.index.map(
                    lambda idx: "; ".join(st.session_state["aic_coding_data"].get(idx, []))
                )
                export_df["ai_codes"] = export_df.index.map(
                    lambda idx: "; ".join(st.session_state.get("aic_ai_suggestions", {}).get(idx, {}).get("codes", []))
                )

            with st.expander("Preview export data"):
                st.dataframe(export_df)

            render_download_buttons(export_df, filename_prefix="ai_coded_results")

    def _render_analytics_panel(self, df, codes, total_rows):
        """Render the inter-rater reliability analytics panel"""
        st.markdown("---")
        st.subheader("AI-Human Agreement Analysis")

        human_codes = st.session_state.get("aic_coding_data", {})
        ai_suggestions = st.session_state.get("aic_ai_suggestions", {})

        if not ai_suggestions:
            st.info("No AI suggestions yet. Run batch processing or navigate through rows to get AI suggestions.")
            return

        # Calculate metrics
        metrics = calculate_all_metrics(human_codes, ai_suggestions, codes, total_rows)

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Exact Agreement", f"{metrics.exact_agreement:.1f}%",
                     help="Percentage of rows where codes match exactly")
        with col2:
            kappa_interp = interpret_kappa(metrics.cohens_kappa)
            st.metric("Cohen's Kappa", f"{metrics.cohens_kappa:.3f}",
                     help=f"Agreement corrected for chance ({kappa_interp})")
        with col3:
            st.metric("Jaccard Index", f"{metrics.jaccard_index:.3f}",
                     help="Set-based similarity (0=no overlap, 1=perfect)")

        # Coverage
        st.markdown("### Coverage")
        cov_col1, cov_col2, cov_col3 = st.columns(3)
        with cov_col1:
            st.metric("Total Rows", metrics.total_rows)
        with cov_col2:
            st.metric("Human Coded", metrics.coded_rows)
        with cov_col3:
            st.metric("AI Suggested", metrics.ai_suggested_rows)

        # Per-code metrics
        st.markdown("### Per-Code Metrics")

        code_metrics_data = []
        for code in codes:
            code_metrics_data.append({
                "Code": code,
                "Precision": f"{metrics.per_code_precision.get(code, 0)*100:.1f}%",
                "Recall": f"{metrics.per_code_recall.get(code, 0)*100:.1f}%",
                "F1": f"{metrics.per_code_f1.get(code, 0)*100:.1f}%"
            })
        st.dataframe(pd.DataFrame(code_metrics_data), use_container_width=True)

        # Confusion matrix
        st.markdown("### Confusion Matrix")
        st.caption("Rows = AI Suggested, Columns = Human Applied")

        try:
            import plotly.express as px
            fig = px.imshow(
                metrics.confusion_matrix.values,
                labels=dict(x="Human Applied", y="AI Suggested", color="Count"),
                x=metrics.confusion_matrix.columns.tolist(),
                y=metrics.confusion_matrix.index.tolist(),
                color_continuous_scale="Blues",
                text_auto=True
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.dataframe(metrics.confusion_matrix)

        # Disagreement analysis
        st.markdown("### Disagreement Analysis")

        disagreements = get_disagreement_analysis(human_codes, ai_suggestions, total_rows)

        if not disagreements:
            st.success("No disagreements found!")
        else:
            st.warning(f"{len(disagreements)} rows with disagreements")

            # Show first 10 disagreements
            with st.expander(f"View disagreements ({len(disagreements)} total)"):
                for dis in disagreements[:10]:
                    row_idx = dis["row_idx"]
                    st.markdown(f"**Row {row_idx + 1}** ({dis['type']})")

                    dis_col1, dis_col2 = st.columns(2)
                    with dis_col1:
                        st.markdown(f"Human: {', '.join(dis['human_codes']) or 'None'}")
                    with dis_col2:
                        st.markdown(f"AI: {', '.join(dis['ai_codes']) or 'None'}")

                    if dis["only_in_human"]:
                        st.caption(f"Only human: {', '.join(dis['only_in_human'])}")
                    if dis["only_in_ai"]:
                        st.caption(f"Only AI: {', '.join(dis['only_in_ai'])}")
                    if dis["ai_reasoning"]:
                        st.caption(f"AI reasoning: {dis['ai_reasoning']}")

                    st.markdown("---")

        st.markdown("---")
