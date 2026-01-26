"""
Process Documents Tool
Tool for extracting structured data from documents using AI
"""

import streamlit as st
import pandas as pd
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List

from .base import BaseTool, ToolConfig, ToolResult
from core.providers import LLMProvider, PROVIDER_CONFIGS
from core.processing import DocumentProcessor, ProcessingConfig, build_results_dataframe
from core.document_reader import (
    get_files_from_folder, SUPPORTED_EXTENSIONS,
    get_extension_display_groups, get_supported_extensions
)
from core.document_templates import (
    get_template_names, get_template_prompt, get_template_columns,
    get_template_description, DOCUMENT_TEMPLATES
)
from database import get_db, RunStatus, LogLevel
from ui.state import (
    get_selected_provider, get_effective_api_key, get_selected_model,
    get_current_settings, set_current_session_id, get_current_session_id
)


class ProcessDocumentsTool(BaseTool):
    """Tool for extracting structured data from documents using AI"""

    id = "process_documents"
    name = "Process Documents"
    description = "Extract structured data from documents using AI"
    icon = ":material/description:"
    category = "Processing"

    def __init__(self):
        self.db = get_db()

    def render_config(self) -> ToolConfig:
        """Render document processing configuration UI"""

        # Step 1: Document Selection
        st.header("1. Select Documents")

        input_method = st.radio(
            "Input Method",
            ["Folder Path", "Upload Files"],
            horizontal=True,
            key="doc_input_method"
        )

        files = []
        uploaded_files = []
        folder_path = ""
        use_uploaded = False

        if input_method == "Folder Path":
            col1, col2 = st.columns([4, 1])
            with col1:
                folder_path = st.text_input(
                    "Folder Path",
                    placeholder="/path/to/documents",
                    help="Enter full path to folder containing documents",
                    key="doc_folder_path"
                )
            with col2:
                st.write("")
                st.write("")
                if st.button("Browse", help="Quick access to common folders"):
                    st.session_state.show_folder_browse = not st.session_state.get("show_folder_browse", False)

            if st.session_state.get("show_folder_browse", False):
                home = str(Path.home())
                common_paths = [
                    home,
                    f"{home}/Documents",
                    f"{home}/Downloads",
                    f"{home}/Desktop",
                    os.getcwd()
                ]
                valid_paths = [p for p in common_paths if os.path.exists(p)]
                selected_path = st.selectbox("Quick Access", valid_paths)
                if st.button("Use This Path"):
                    st.session_state.doc_folder_path = selected_path
                    st.session_state.show_folder_browse = False
                    st.rerun()

            # File type filters
            st.write("**File Types:**")
            ft_cols = st.columns(5)
            with ft_cols[0]:
                include_txt = st.checkbox("TXT/MD", value=True, key="doc_include_txt")
            with ft_cols[1]:
                include_pdf = st.checkbox("PDF", value=True, key="doc_include_pdf")
            with ft_cols[2]:
                include_docx = st.checkbox("DOCX", value=True, key="doc_include_docx")
            with ft_cols[3]:
                include_data = st.checkbox("JSON/CSV", value=False, key="doc_include_data")
            with ft_cols[4]:
                include_web = st.checkbox("HTML/XML", value=False, key="doc_include_web")

            # Build extensions list
            selected_extensions = []
            if include_txt:
                selected_extensions.extend(['.txt', '.md'])
            if include_pdf:
                selected_extensions.append('.pdf')
            if include_docx:
                selected_extensions.extend(['.docx', '.doc'])
            if include_data:
                selected_extensions.extend(['.json', '.csv'])
            if include_web:
                selected_extensions.extend(['.html', '.htm', '.xml'])

            if folder_path:
                try:
                    files = get_files_from_folder(folder_path, selected_extensions)
                    if files:
                        st.success(f"Found **{len(files)}** documents")
                        with st.expander(f"File List ({len(files)} files)", expanded=False):
                            file_df = pd.DataFrame({
                                "File Name": [os.path.basename(f) for f in files],
                                "Type": [Path(f).suffix for f in files],
                                "Path": files
                            })
                            st.dataframe(file_df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("No matching documents found in folder")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        else:  # Upload Files
            use_uploaded = True
            uploaded_files = st.file_uploader(
                "Upload Documents",
                type=['txt', 'md', 'pdf', 'docx', 'doc', 'json', 'csv', 'html', 'htm', 'xml'],
                accept_multiple_files=True,
                help="Select multiple files to process",
                key="doc_file_upload"
            )
            if uploaded_files:
                st.success(f"Uploaded **{len(uploaded_files)}** files")
                with st.expander(f"Uploaded Files ({len(uploaded_files)})", expanded=False):
                    upload_df = pd.DataFrame({
                        "File Name": [f.name for f in uploaded_files],
                        "Size": [f"{f.size/1024:.1f} KB" for f in uploaded_files],
                        "Type": [Path(f.name).suffix for f in uploaded_files]
                    })
                    st.dataframe(upload_df, use_container_width=True, hide_index=True)

        # Check if we have documents
        has_documents = len(files) > 0 or len(uploaded_files) > 0

        # Step 2: Processing Template
        st.header("2. Processing Template")

        template_names = get_template_names()
        template_choice = st.selectbox(
            "Template",
            template_names,
            help="Select a template or choose 'Custom' to create your own",
            key="doc_template_choice"
        )

        # Show template description
        description = get_template_description(template_choice)
        if description:
            st.caption(description)

        # Auto-update when template changes
        if "doc_last_template" not in st.session_state:
            st.session_state.doc_last_template = template_choice
        if st.session_state.doc_last_template != template_choice:
            st.session_state.doc_system_prompt = get_template_prompt(template_choice)
            st.session_state.doc_csv_columns = get_template_columns(template_choice)
            st.session_state.doc_last_template = template_choice
            # Let natural page flow handle the update instead of forcing rerun

        # Get default values from template
        default_prompt = get_template_prompt(template_choice)
        default_columns = get_template_columns(template_choice)

        system_prompt = st.text_area(
            "Processing Instructions",
            height=200,
            value=st.session_state.get("doc_system_prompt", default_prompt),
            placeholder="Enter instructions for processing each document...",
            key="doc_system_prompt"
        )

        # CSV column header input
        st.write("**Output Columns:**")
        csv_columns = st.text_input(
            "Column Headers (comma-separated)",
            value=st.session_state.get("doc_csv_columns", default_columns),
            key="doc_csv_columns",
            help="These will be the column headers in your output CSV"
        )

        if csv_columns:
            st.info(f"Output: `document_name` + `{csv_columns}`")

        # Template Import/Export
        with st.expander("Import/Export Templates"):
            col_imp, col_exp = st.columns(2)

            with col_exp:
                st.write("**Export Current as Template:**")
                template_name_export = st.text_input(
                    "Template Name",
                    value="My Custom Template",
                    key="doc_template_name_export"
                )
                if st.button("Export to JSON", key="doc_export_btn"):
                    export_data = {
                        "name": template_name_export,
                        "prompt": system_prompt,
                        "columns": csv_columns
                    }
                    json_str = json.dumps(export_data, indent=2)
                    st.download_button(
                        "Download Template JSON",
                        json_str,
                        f"{template_name_export.replace(' ', '_')}.json",
                        "application/json"
                    )

            with col_imp:
                st.write("**Import Template:**")
                uploaded_template = st.file_uploader(
                    "Upload JSON template",
                    type=['json'],
                    key="doc_template_upload"
                )
                if uploaded_template:
                    try:
                        template_data = json.load(uploaded_template)
                        if st.button("Apply Template", key="doc_apply_template_btn"):
                            st.session_state.doc_system_prompt = template_data.get("prompt", "")
                            st.session_state.doc_csv_columns = template_data.get("columns", "column1,column2,column3")
                            st.success(f"Loaded template: {template_data.get('name', 'Unknown')}")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Invalid template file: {e}")

        # Validation
        if not has_documents:
            return ToolConfig(
                is_valid=False,
                error_message="Please select documents to process"
            )

        if not system_prompt or not system_prompt.strip():
            return ToolConfig(
                is_valid=False,
                error_message="Please enter processing instructions",
                config_data={
                    "files": files,
                    "uploaded_files": uploaded_files,
                    "use_uploaded": use_uploaded,
                    "folder_path": folder_path
                }
            )

        if not csv_columns or not csv_columns.strip():
            return ToolConfig(
                is_valid=False,
                error_message="Please enter output column names",
                config_data={
                    "files": files,
                    "uploaded_files": uploaded_files,
                    "use_uploaded": use_uploaded,
                    "folder_path": folder_path,
                    "system_prompt": system_prompt
                }
            )

        # Get settings from session state
        provider = get_selected_provider()
        api_key = get_effective_api_key()
        model = get_selected_model()

        if PROVIDER_CONFIGS[provider].requires_api_key and not api_key:
            return ToolConfig(
                is_valid=False,
                error_message="Please enter an API key for the selected provider",
                config_data={
                    "files": files,
                    "uploaded_files": uploaded_files,
                    "use_uploaded": use_uploaded,
                    "folder_path": folder_path,
                    "system_prompt": system_prompt,
                    "csv_columns": csv_columns
                }
            )

        # Build config
        config_data = {
            "files": files,
            "uploaded_files": uploaded_files,
            "use_uploaded": use_uploaded,
            "folder_path": folder_path,
            "system_prompt": system_prompt,
            "csv_columns": csv_columns,
            "provider": provider,
            "api_key": api_key,
            "base_url": st.session_state.get("base_url"),
            "model": model,
            "temperature": st.session_state.get("temperature", 0.0),
            "max_tokens": st.session_state.get("max_tokens", 4096),
            "json_mode": False,  # Document processing uses CSV output
            "max_concurrency": st.session_state.get("max_concurrency", 5),
            "auto_retry": st.session_state.get("auto_retry", True),
            "max_retries": st.session_state.get("max_retries", 3),
            "realtime_progress": st.session_state.get("realtime_progress", True),
            "save_path": st.session_state.get("save_path", ""),
            "test_batch_size": st.session_state.get("test_batch_size", 5),
        }

        return ToolConfig(is_valid=True, config_data=config_data)

    async def execute(
        self,
        config: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> ToolResult:
        """Execute the document processing operation"""

        is_test = config.get("is_test", False)
        test_batch_size = config.get("test_batch_size", 5)
        use_uploaded = config.get("use_uploaded", False)

        # Prepare document list
        if use_uploaded:
            doc_list = [(f.name, f) for f in config["uploaded_files"]]
        else:
            doc_list = [(os.path.basename(f), f) for f in config["files"]]

        # Validate document list is not empty
        if not doc_list:
            return ToolResult(
                success=False,
                error_message="No documents to process. Please select files or a folder with documents."
            )

        # Limit for test
        if is_test:
            doc_list = doc_list[:test_batch_size]

        run_type = "test" if is_test else "full"

        # Create or get session
        session_id = get_current_session_id()
        if not session_id:
            session = self.db.create_session("process_documents", get_current_settings())
            session_id = session.session_id
            set_current_session_id(session_id)
            self.db.log(LogLevel.INFO, f"Created new session: {session.name}",
                       session_id=session_id)

        # Create run
        run = self.db.create_run(
            session_id=session_id,
            run_type=run_type,
            provider=config["provider"].value,
            model=config["model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            system_prompt=config["system_prompt"],
            schema={"type": "document_processing", "columns": config["csv_columns"]},
            variables={},
            input_file=config.get("folder_path", "uploaded"),
            input_rows=len(doc_list),
            json_mode=False,
            max_concurrency=config["max_concurrency"],
            auto_retry=config["auto_retry"],
            max_retry_attempts=config["max_retries"],
            run_settings=get_current_settings()
        )

        self.db.log(LogLevel.INFO, f"Started {run_type} run with {len(doc_list)} documents",
                   {"provider": config["provider"].value, "model": config["model"]},
                   run_id=run.run_id, session_id=session_id)

        # Create processor
        processing_config = ProcessingConfig(
            provider=config["provider"],
            api_key=config["api_key"],
            base_url=config["base_url"],
            model=config["model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            json_mode=False,
            max_concurrency=config["max_concurrency"],
            auto_retry=config["auto_retry"],
            max_retries=config["max_retries"],
            save_path=config["save_path"],
            realtime_progress=config.get("realtime_progress", True) if not is_test else True
        )

        processor = DocumentProcessor(processing_config, run.run_id, session_id)

        # Run processing
        try:
            result = await processor.process(
                doc_list,
                config["system_prompt"],
                config["csv_columns"],
                use_uploaded,
                progress_callback
            )

            # Update run status
            self.db.update_run_status(
                run.run_id, RunStatus.COMPLETED,
                success_count=result.success_count,
                error_count=result.error_count,
                retry_count=result.retry_count,
                avg_latency=result.avg_latency,
                total_duration=result.total_duration
            )

            self.db.log(LogLevel.INFO, f"Run completed: {result.success_count} success, {result.error_count} errors",
                       {"duration": result.total_duration, "avg_latency": result.avg_latency},
                       run_id=run.run_id)

            # Build results DataFrame
            result_df = build_results_dataframe(result.results, config["csv_columns"])

            return ToolResult(
                success=True,
                data=result_df,
                stats={
                    "success": result.success_count,
                    "errors": result.error_count,
                    "retries": result.retry_count,
                    "avg_latency": f"{result.avg_latency:.2f}s",
                    "duration": f"{result.total_duration:.1f}s"
                }
            )

        except Exception as e:
            self.db.update_run_status(run.run_id, RunStatus.FAILED)
            self.db.log(LogLevel.ERROR, f"Run failed: {str(e)}", run_id=run.run_id)
            return ToolResult(success=False, error_message=str(e))

    def render_results(self, result: ToolResult):
        """Render document processing results"""
        import streamlit as st
        from ui.components.result_inspector import render_result_inspector
        from ui.components.download_buttons import render_download_buttons

        if not result.success:
            st.error(f"Document processing failed: {result.error_message}")
            return

        # Success message with stats
        stats = result.stats
        st.success(
            f"Complete! "
            f" {stats.get('success', 0)} | "
            f" {stats.get('errors', 0)} | "
            f" {stats.get('retries', 0)} retries | "
            f"Avg: {stats.get('avg_latency', '0s')} | "
            f"Total: {stats.get('duration', '0s')}"
        )

        # Data preview
        st.dataframe(result.data, use_container_width=True, hide_index=True)

        # Result inspector
        st.divider()
        render_result_inspector(result.data)

        # Download buttons
        st.divider()
        render_download_buttons(result.data, filename_prefix="docuai_results")
