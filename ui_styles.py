"""Shared Streamlit CSS for WRC app pages."""

WRC_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700&family=Quicksand:wght@400;500;600;700&display=swap');

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background: linear-gradient(135deg, #FFF8F3 0%, #FFF5EE 50%, #FEF3E8 100%);
        font-family: 'Nunito', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFF8F3 0%, #FFE8D6 100%);
        border-right: 1px solid #F5D5C8;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    h1, h2, h3 {
        font-family: 'Quicksand', sans-serif !important;
        color: #5D4E6D !important;
    }

    .hero-container {
        background: linear-gradient(135deg, #E8D5E0 0%, #D4C1EC 50%, #C9B8DB 100%);
        border-radius: 24px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(93, 78, 109, 0.1);
        text-align: center;
    }

    .hero-title {
        font-family: 'Quicksand', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #4A3F5C;
        margin-bottom: 0.25rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #6B5B7A;
        margin: 0;
    }

    .panel-note {
        background: #eef5ff;
        border: 1px solid #c5d9f7;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        color: #1e4f8a;
        margin: 1rem 0;
    }

    .status-pending {
        background: #fff3e0;
        color: #e65100;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        font-weight: 600;
        display: inline-block;
    }

    .status-approved {
        background: #e8f5e9;
        color: #2e7d32;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        font-weight: 600;
        display: inline-block;
    }

    .status-rejected {
        background: #fdecea;
        color: #c62828;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        font-weight: 600;
        display: inline-block;
    }

    .match-card {
        background: white;
        border: 1px solid #F0E6ED;
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 12px rgba(93, 78, 109, 0.05);
    }

    .match-card.ccsf {
        border-left: 4px solid #9B7AA5;
    }
</style>
"""
