#!/bin/bash
# Start Handai v4.0
cd "$(dirname "$0")"
streamlit run app.py "$@"
