import streamlit as st


GITHUB_REPOSITORY_URL = (
    "https://github.com/siddharthapramanik771/HeartDiseaseRiskAnalysis"
)


def apply_page_styles() -> None:
    st.markdown(
        """
<style>
    :root {
        --app-panel: rgba(255, 255, 255, 0.88);
        --app-panel-soft: rgba(248, 250, 252, 0.94);
        --app-line: rgba(100, 116, 139, 0.24);
        --app-shadow: rgba(15, 23, 42, 0.10);
        --app-accent: #0f766e;
        --app-accent-strong: #155e75;
        --app-rose: #be123c;
        --app-amber: #b45309;
        --sidebar-bg-start: #132238;
        --sidebar-bg-end: #17454c;
        --sidebar-ink: #f8fbfc;
        --sidebar-muted: #d9e8ee;
        --tab-active-bg: #123549;
        --tab-active-ink: #ffffff;
    }

    .stApp {
        background:
            linear-gradient(180deg, rgba(15, 118, 110, 0.07), transparent 22rem),
            #f6f7fb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1260px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--sidebar-bg-start) 0%, var(--sidebar-bg-end) 100%);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: var(--sidebar-ink);
    }

    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
        color: var(--sidebar-muted);
    }

    [data-testid="stSidebar"] [data-testid="stDownloadButton"] button,
    [data-testid="stSidebar"] [data-testid="stLinkButton"] a {
        background: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.22);
        color: #132238 !important;
        width: 100%;
    }

    [data-testid="stSidebar"] [data-testid="stDownloadButton"] button *,
    [data-testid="stSidebar"] [data-testid="stLinkButton"] a * {
        color: #132238 !important;
    }

    .hero {
        border: 1px solid rgba(255, 255, 255, 0.32);
        border-radius: 8px;
        padding: 2.1rem 2.35rem;
        background:
            radial-gradient(circle at 86% 22%, rgba(250, 204, 21, 0.20), transparent 22%),
            linear-gradient(135deg, rgba(19, 34, 56, 0.98), rgba(15, 118, 110, 0.88));
        box-shadow: 0 18px 45px var(--app-shadow);
        color: #ffffff;
        margin-bottom: 1.1rem;
        overflow: hidden;
    }

    .hero__eyebrow {
        color: #b8eee6;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0;
        margin-bottom: 0.55rem;
        text-transform: uppercase;
    }

    .hero h1 {
        color: #ffffff;
        font-size: clamp(2.05rem, 4vw, 4rem);
        line-height: 1.02;
        margin: 0 0 0.85rem;
    }

    .hero p {
        color: rgba(255, 255, 255, 0.88);
        font-size: 1.05rem;
        line-height: 1.62;
        max-width: 820px;
        margin: 0 0 1.25rem;
    }

    .hero__actions {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        align-items: center;
    }

    .hero__link {
        background: #ffffff;
        border-radius: 8px;
        color: #132238 !important;
        display: inline-flex;
        font-weight: 800;
        padding: 0.72rem 1rem;
        text-decoration: none;
    }

    .hero__note {
        color: #e2fffb;
        font-size: 0.95rem;
        font-weight: 650;
    }

    .status-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.85rem;
        margin: 0.9rem 0 1.35rem;
    }

    .status-tile,
    div[data-testid="stMetric"],
    div[data-testid="stForm"] {
        background: var(--app-panel);
        border: 1px solid var(--app-line);
        border-radius: 8px;
        box-shadow: 0 10px 28px var(--app-shadow);
        color: inherit;
    }

    .status-tile {
        padding: 1rem 1.05rem;
    }

    .status-tile span {
        display: block;
        font-size: 0.78rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
        opacity: 0.72;
        text-transform: uppercase;
    }

    .status-tile strong {
        display: block;
        font-size: 1.35rem;
        line-height: 1.2;
    }

    .status-tile small {
        display: block;
        margin-top: 0.3rem;
        opacity: 0.68;
    }

    div[data-testid="stMetric"] {
        padding: 0.9rem 1rem;
    }

    div[data-testid="stForm"] {
        padding: 1.15rem;
    }

    div[data-testid="stDataFrame"],
    div[data-testid="stAlert"] {
        border-radius: 8px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.45rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: var(--app-panel);
        border: 1px solid var(--app-line);
        border-radius: 8px;
        color: inherit !important;
        font-weight: 750;
        padding: 0.65rem 1rem;
    }

    .stTabs [data-baseweb="tab"] * {
        color: inherit !important;
    }

    .stTabs [aria-selected="true"] {
        background: var(--tab-active-bg);
        color: var(--tab-active-ink) !important;
    }

    .stTabs [aria-selected="true"] * {
        color: var(--tab-active-ink) !important;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, var(--app-accent), var(--app-accent-strong));
        border: 0;
        border-radius: 8px;
        color: #ffffff !important;
        font-weight: 800;
        min-height: 2.85rem;
        box-shadow: 0 10px 22px rgba(15, 118, 110, 0.24);
    }

    .stButton > button *,
    .stFormSubmitButton > button * {
        color: #ffffff !important;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        border: 0;
        color: #ffffff !important;
        transform: translateY(-1px);
    }

    @media (max-width: 780px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            padding: 1.45rem;
        }

        .status-strip {
            grid-template-columns: 1fr;
        }
    }
</style>
""",
        unsafe_allow_html=True,
    )
