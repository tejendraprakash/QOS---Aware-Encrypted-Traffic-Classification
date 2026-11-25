"""
Reusable UI components and layout elements for the QoS Encrypted Traffic AI Research Lab Dashboard.
"""

from typing import Optional, Dict, Any, Union
import streamlit as st


def apply_custom_css():
    """Inject polished, modern academic laboratory CSS styling."""
    custom_css = """
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    code, pre, .stCodeBlock {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main container styling & padding */
    .main {
        background-color: #F8FAFC !important;
    }

    .main .block-container {
        padding-top: 1.75rem;
        padding-bottom: 3.5rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 1440px;
    }

    /* Navigation Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #172554 !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    section[data-testid="stSidebar"] * {
        color: #E2E8F0;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(226, 232, 240, 0.2) !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        color: #E2E8F0 !important;
        font-weight: 500;
        border-radius: 6px;
        padding: 4px 8px;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
        color: #FFFFFF !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border-radius: 6px;
    }

    /* Laboratory Hero Header */
    .lab-header {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
        border-radius: 8px;
        padding: 1.75rem 2rem;
        margin-bottom: 1.75rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        color: #0F172A;
    }

    .lab-header-title {
        font-size: 1.95rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin: 0 0 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        color: #0F172A;
    }

    .lab-header-subtitle {
        font-size: 0.96rem;
        color: #475569;
        margin-bottom: 0.85rem;
        line-height: 1.5;
    }

    .lab-header-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.85rem;
        margin-top: 0.85rem;
        padding-top: 0.85rem;
        border-top: 1px solid #E2E8F0;
        font-size: 0.82rem;
        color: #475569;
    }

    .lab-meta-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-weight: 500;
        color: #334155;
    }

    /* Professional KPI Cards */
    .kpi-card {
        background: #FFFFFF;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        padding: 1.3rem 1.45rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
        height: 100%;
        position: relative;
        overflow: hidden;
    }

    .kpi-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.07);
    }

    .kpi-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        margin-bottom: 0.45rem;
    }

    .kpi-value-row {
        display: flex;
        align-items: baseline;
        gap: 0.65rem;
        margin-bottom: 0.35rem;
    }

    .kpi-value {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.025em;
    }

    .kpi-delta {
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.2rem;
        border-radius: 6px;
        padding: 0.2rem 0.45rem;
    }

    .kpi-delta.positive {
        color: #16A34A;
        background: #DCFCE7;
        border: 1px solid #BBF7D0;
    }

    .kpi-delta.negative {
        color: #DC2626;
        background: #FEE2E2;
        border: 1px solid #FECACA;
    }

    .kpi-delta.neutral {
        color: #475569;
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
    }

    .kpi-help {
        font-size: 0.77rem;
        color: #64748B;
        margin-top: 0.4rem;
        line-height: 1.35;
    }

    /* Primary Buttons */
    .stButton > button {
        background: #2563EB !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: 1px solid #2563EB !important;
        border-radius: 6px !important;
        padding: 0.5rem 1.35rem !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background: #1D4ED8 !important;
        border-color: #1D4ED8 !important;
        box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.1) !important;
    }

    /* Styled Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
        font-weight: 600;
        color: #64748B;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        color: #2563EB !important;
        border-bottom: 2px solid #2563EB !important;
        background: transparent !important;
    }

    /* Styled Expander and DataFrames */
    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        background: #FFFFFF !important;
    }

    /* Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .badge-success {
        background-color: #DCFCE7;
        color: #16A34A;
        border: 1px solid #BBF7D0;
    }

    .badge-warning {
        background-color: #FEF3C7;
        color: #D97706;
        border: 1px solid #FDE68A;
    }

    .badge-danger {
        background-color: #FEE2E2;
        color: #DC2626;
        border: 1px solid #FECACA;
    }

    .badge-info {
        background-color: #DBEAFE;
        color: #2563EB;
        border: 1px solid #BFDBFE;
    }

    .badge-secondary {
        background-color: #F1F5F9;
        color: #475569;
        border: 1px solid #E2E8F0;
    }

    /* Section Header */
    .section-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #E2E8F0;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .section-desc {
        font-size: 0.85rem;
        color: #475569;
        margin: 0.25rem 0 0 0;
    }

    /* Information Callout Box */
    .info-callout {
        background: #FFFFFF;
        border-left: 4px solid #2563EB;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin-bottom: 1.25rem;
        color: #334155;
        font-size: 0.88rem;
    }

    /* Custom Footer */
    .lab-footer {
        margin-top: 3.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid #E2E8F0;
        text-align: center;
        color: #64748B;
        font-size: 0.82rem;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_header(
    title: str = "QoS-Aware Encrypted Traffic AI Research Lab",
    subtitle: str = "Live Evaluation, Classification & End-to-End Scheduling Platform",
    dataset_name: str = "ISCX VPN-nonVPN (Track B Cleaned)",
    model_name: str = "Phase 7.7 Champion (LightGBM 8-Feature)",
    flow_count: Optional[int] = 11727,
    active_page: Optional[str] = None
):
    """Render the master research laboratory header with system metadata badges."""
    flow_str = f"{flow_count:,} Flows" if flow_count else "11,727 Verified Flows"
    page_badge = f'<span class="lab-meta-tag">📍 View: <b>{active_page}</b></span>' if active_page else ""

    html = (
        f'<div class="lab-header">'
        f'<div class="lab-header-title"><span>📡</span> {title}</div>'
        f'<div class="lab-header-subtitle">{subtitle}</div>'
        f'<div class="lab-header-meta">'
        f'<span class="lab-meta-tag">🗄️ Dataset: <b>{dataset_name}</b></span>'
        f'<span class="lab-meta-tag">🤖 Champion Model: <b>{model_name}</b></span>'
        f'<span class="lab-meta-tag">📊 Scope: <b>{flow_str}</b></span>'
        f'<span class="lab-meta-tag">⚡ Architecture: <b>Early 5-Pkt Early Ingress</b></span>'
        f'{page_badge}'
        f'</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_kpi_card(
    title: str,
    value: Union[str, int, float],
    delta: Optional[str] = None,
    delta_type: str = "positive",
    help_text: Optional[str] = None,
    icon: Optional[str] = None,
    border_accent: Optional[str] = None
):
    """
    Render a polished KPI card with optional delta indicator, tooltip, and icon.
    
    delta_type: 'positive' (green), 'negative' (red), 'neutral' (slate)
    """
    icon_html = f'<span style="margin-right: 0.35rem;">{icon}</span>' if icon else ""
    delta_html = ""
    if delta is not None:
        delta_class = delta_type if delta_type in ["positive", "negative", "neutral"] else "neutral"
        delta_html = f'<span class="kpi-delta {delta_class}">{delta}</span>'
    
    help_html = f'<div class="kpi-help">{help_text}</div>' if help_text else ""
    border_style = f' style="border-top: 3px solid {border_accent};"' if border_accent else ""

    card_html = (
        f'<div class="kpi-card"{border_style}>'
        f'<div class="kpi-title">{icon_html}{title}</div>'
        f'<div class="kpi-value-row">'
        f'<span class="kpi-value">{value}</span>'
        f'{delta_html}'
        f'</div>'
        f'{help_html}'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_status_badge(status: str, message: Optional[str] = None) -> str:
    """
    Render or return HTML for a status badge.
    Status can be 'PASSED', 'FAILED', 'WARNING', 'ACTIVE', 'MET', 'VIOLATED', 'INFO'.
    """
    status_lower = status.lower()
    if status_lower in ["passed", "met", "active", "optimal", "clean", "success"]:
        badge_class = "badge-success"
        icon = "✓"
    elif status_lower in ["warning", "marginal", "caution"]:
        badge_class = "badge-warning"
        icon = "⚠"
    elif status_lower in ["failed", "violated", "error", "degraded", "drop"]:
        badge_class = "badge-danger"
        icon = "✕"
    elif status_lower in ["info", "running", "in_progress"]:
        badge_class = "badge-info"
        icon = "ℹ"
    else:
        badge_class = "badge-secondary"
        icon = "•"

    display_text = message if message else status
    badge_html = f'<span class="badge {badge_class}">{icon} {display_text}</span>'
    return badge_html


def render_sla_badge(sla_met: bool, label: Optional[str] = None):
    """Convenience helper to render SLA compliance pill badge."""
    if sla_met:
        status_text = label if label else "SLA COMPLIANT"
        st.markdown(f'<span class="badge badge-success">✓ {status_text}</span>', unsafe_allow_html=True)
    else:
        status_text = label if label else "SLA VIOLATED"
        st.markdown(f'<span class="badge badge-danger">✕ {status_text}</span>', unsafe_allow_html=True)


def render_section_title(title: str, subtitle: Optional[str] = None, badge_text: Optional[str] = None, badge_type: str = "info"):
    """Render a structured section header with optional description and badge."""
    badge_html = ""
    if badge_text:
        badge_class = f"badge-{badge_type}"
        badge_html = f'<div><span class="badge {badge_class}">{badge_text}</span></div>'
    
    sub_html = f'<p class="section-desc">{subtitle}</p>' if subtitle else ""
    html = (
        f'<div class="section-header">'
        f'<div><h3 class="section-title">{title}</h3>{sub_html}</div>'
        f'{badge_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_callout(text: str, title: Optional[str] = None, icon: str = "💡", accent_color: str = "#2563EB"):
    """Render a callout banner for notes, methodological considerations, or warnings."""
    title_html = f'<div style="font-weight: 700; margin-bottom: 0.35rem; color: #0F172A;">{icon} {title}</div>' if title else ""
    html = (
        f'<div class="info-callout" style="border-left-color: {accent_color};">'
        f'{title_html}'
        f'<div>{text}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_footer():
    """Render the dashboard footer with provenance and citation details."""
    html = (
        f'<div class="lab-footer">'
        f'<div><b>QoS-Aware Encrypted Traffic AI Research Laboratory</b> &bull; Phase 7.7 Champion LightGBM Architecture</div>'
        f'<div style="margin-top: 0.35rem; font-size: 0.75rem;">'
        f'Reproducibility Manifest: <code>output/phase7_7/phase7_7_champion_model.json</code> &bull; '
        f'Strict Leakage Isolation & Nested Stratified GroupKFold &bull; IEEE / ACM Telecommunications Research Standard'
        f'</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)
