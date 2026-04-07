"""
WRC Resource Finder - Web Interface
A Streamlit app for searching Women's Resource Center resources
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from database import ResourceDatabase
from rag_system import RAGSystem

# Page config
st.set_page_config(
    page_title="WRC Resource Finder",
    page_icon="💜",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Warm, welcoming design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700&family=Quicksand:wght@400;500;600;700&display=swap');
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main app background - warm cream */
    .stApp {
        background: linear-gradient(135deg, #FFF8F3 0%, #FFF5EE 50%, #FEF3E8 100%);
        font-family: 'Nunito', sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFF8F3 0%, #FFE8D6 100%);
        border-right: 1px solid #F5D5C8;
    }
    
    /* Main container padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Quicksand', sans-serif !important;
        color: #5D4E6D !important;
    }
    
    /* Hero section */
    .hero-container {
        background: linear-gradient(135deg, #E8D5E0 0%, #D4C1EC 50%, #C9B8DB 100%);
        border-radius: 24px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(93, 78, 109, 0.1);
        text-align: center;
    }
    
    .hero-title {
        font-family: 'Quicksand', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #4A3F5C;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-family: 'Nunito', sans-serif;
        font-size: 1.1rem;
        color: #6B5B7A;
        margin-bottom: 0;
    }
    
    /* Search container */
    .search-container {
        background: white;
        border-radius: 20px;
        padding: 1.5rem 2rem;
        margin: -1rem auto 2rem auto;
        max-width: 800px;
        box-shadow: 0 4px 20px rgba(93, 78, 109, 0.08);
        border: 1px solid #F0E6ED;
    }
    
    /* Quick action buttons */
    .stButton > button {
        font-family: 'Nunito', sans-serif !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        border: none !important;
    }
    
    /* Primary search button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #9B7AA5 0%, #7D6B8A 100%) !important;
        color: white !important;
        padding: 0.6rem 2rem !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #8A6994 0%, #6C5A79 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(155, 122, 165, 0.3) !important;
    }
    
    /* Quick search buttons */
    .quick-btn button {
        background: #FFF !important;
        color: #5D4E6D !important;
        border: 2px solid #E8D5E0 !important;
        padding: 0.8rem 1rem !important;
    }
    
    .quick-btn button:hover {
        background: #F9F0F5 !important;
        border-color: #D4C1EC !important;
        transform: translateY(-2px) !important;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        font-family: 'Nunito', sans-serif !important;
        border-radius: 12px !important;
        border: 2px solid #E8D5E0 !important;
        padding: 0.8rem 1rem !important;
        font-size: 1rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #9B7AA5 !important;
        box-shadow: 0 0 0 3px rgba(155, 122, 165, 0.15) !important;
    }
    
    /* Resource card styling */
    .resource-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #F0E6ED;
        box-shadow: 0 2px 12px rgba(93, 78, 109, 0.05);
        transition: all 0.2s ease;
    }
    
    .resource-card:hover {
        box-shadow: 0 8px 24px rgba(93, 78, 109, 0.1);
        transform: translateY(-2px);
    }
    
    .resource-card-ccsf {
        border-left: 4px solid #9B7AA5;
        background: linear-gradient(90deg, #FAF5FC 0%, #FFFFFF 20%);
    }
    
    .resource-title {
        font-family: 'Quicksand', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: #4A3F5C;
        margin-bottom: 0.5rem;
    }
    
    .resource-type-badge {
        display: inline-block;
        background: #F3EBF6;
        color: #7D6B8A;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    
    .ccsf-badge {
        display: inline-block;
        background: linear-gradient(135deg, #9B7AA5 0%, #7D6B8A 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .contact-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0;
        color: #5D4E6D;
        font-size: 0.95rem;
    }
    
    .match-score {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        color: #2E7D32;
        padding: 0.4rem 0.8rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    
    .status-current {
        background: #E8F5E9;
        color: #2E7D32;
        padding: 0.25rem 0.6rem;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .status-outdated {
        background: #FFF3E0;
        color: #E65100;
        padding: 0.25rem 0.6rem;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Stats cards */
    .stat-card {
        background: white;
        border-radius: 16px;
        padding: 1.25rem;
        text-align: center;
        border: 1px solid #F0E6ED;
        box-shadow: 0 2px 8px rgba(93, 78, 109, 0.05);
    }
    
    .stat-number {
        font-family: 'Quicksand', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #9B7AA5;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #7D6B8A;
        font-weight: 500;
    }
    
    /* Section headers */
    .section-header {
        font-family: 'Quicksand', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #5D4E6D;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E8D5E0;
    }
    
    /* Results count */
    .results-count {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        color: #2E7D32;
        padding: 0.75rem 1.25rem;
        border-radius: 12px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-family: 'Nunito', sans-serif !important;
        font-weight: 600 !important;
        color: #5D4E6D !important;
        background: white !important;
        border-radius: 12px !important;
    }
    
    /* Radio buttons */
    .stRadio > label {
        font-family: 'Nunito', sans-serif !important;
        font-weight: 500 !important;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: #9B7AA5 !important;
    }
    
    /* Checkbox */
    .stCheckbox > label {
        font-family: 'Nunito', sans-serif !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #7D6B8A;
        font-size: 0.9rem;
        margin-top: 3rem;
        border-top: 1px solid #E8D5E0;
    }
    
    /* Hide default streamlit elements we're replacing */
    .stAlert {
        border-radius: 12px !important;
    }
    
    /* Multiselect */
    .stMultiSelect > div > div {
        border-radius: 12px !important;
    }
    
    /* Data table */
    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# Initialize RAG system
@st.cache_resource
def init_rag():
    if not Path('wrc_resources.db').exists():
        st.error("Database not found! Please run the notebook first to process images.")
        return None
    return RAGSystem('wrc_resources.db', use_local_embeddings=True)


# Initialize
rag = init_rag()
if rag is None:
    st.stop()

# Get stats
with ResourceDatabase('wrc_resources.db') as db:
    stats = db.get_statistics()
    categories = list(stats.get('resources_by_category', {}).keys())

# Hero Section
st.markdown("""
<div class="hero-container">
    <div class="hero-title">💜 Women's Resource Center</div>
    <div class="hero-subtitle">City College of San Francisco • Find the support you need</div>
</div>
""", unsafe_allow_html=True)

# Stats Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{stats.get('total_resources', 0)}</div>
        <div class="stat-label">Resources</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{stats.get('current_resources', 0)}</div>
        <div class="stat-label">Current</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{len(categories)}</div>
        <div class="stat-label">Categories</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">24/7</div>
        <div class="stat-label">Available</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Quick Search Buttons
st.markdown('<div class="section-header">🔍 Quick Search</div>', unsafe_allow_html=True)

quick_searches = [
    ("🏠 Housing", "emergency housing assistance"),
    ("🛡️ Safety", "domestic violence support"),
    ("⚖️ Legal", "legal aid services"),
    ("🍎 Food", "food assistance"),
    ("👶 Childcare", "childcare services"),
    ("🏥 Health", "healthcare services"),
    ("💼 Jobs", "employment assistance"),
    ("📚 Education", "education support"),
]

cols = st.columns(4)
for i, (label, query) in enumerate(quick_searches):
    with cols[i % 4]:
        st.markdown('<div class="quick-btn">', unsafe_allow_html=True)
        if st.button(label, key=f"quick_{i}", use_container_width=True):
            st.session_state.search_query = query
        st.markdown('</div>', unsafe_allow_html=True)

# Search Section
st.markdown('<div class="section-header">Search Resources</div>', unsafe_allow_html=True)

# Search options in columns
col_search, col_options = st.columns([3, 1])

with col_search:
    query = st.text_input(
        "What do you need help with?",
        placeholder="Describe what you're looking for... (e.g., 'help with childcare for working mothers')",
        key="search_query",
        label_visibility="collapsed"
    )

with col_options:
    search_method = st.selectbox(
        "Search type",
        ["🤖 AI Search", "🔤 Keyword"],
        label_visibility="collapsed"
    )

# Filters row
col_filter1, col_filter2 = st.columns([1, 1])

with col_filter1:
    current_only = st.checkbox("Current resources only", value=True)

with col_filter2:
    num_results = st.select_slider(
        "Number of results",
        options=[5, 10, 15, 20, 30],
        value=10
    )

# Search button - full width
search_clicked = st.button("🔍 Search Resources", type="primary", use_container_width=True)

# Category filter is applied after search (optional)
category_filter = []

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

# Perform search
if search_clicked and query:
    with st.spinner('Finding resources...'):
        if search_method == "🤖 AI Search":
            results = rag.search(query, top_k=num_results, current_only=current_only)
        else:
            with ResourceDatabase('wrc_resources.db') as db:
                results = db.search_resources(query, current_only=current_only)[:num_results]
        
        # Apply category filter
        if category_filter:
            results = [r for r in results if r.get('resource_type') in category_filter]
        
        # Mark CCSF resources
        for result in results:
            org_name = result.get('organization_name') or ''
            result['is_ccsf'] = (result.get('binder_name') == 'CCSF_Website' or '(CCSF)' in org_name)
        
        st.session_state.search_results = results


# Helper functions for image rotation
def rotate_left(key):
    st.session_state[key] = (st.session_state.get(key, 0) + 90) % 360

def rotate_right(key):
    st.session_state[key] = (st.session_state.get(key, 0) - 90) % 360


# Display results
if st.session_state.search_results:
    results = st.session_state.search_results
    
    st.markdown(f'<div class="results-count">✨ Found {len(results)} resources</div>', unsafe_allow_html=True)
    
    for i, result in enumerate(results, 1):
        is_ccsf = result.get('is_ccsf', False)
        org_name = result.get('organization_name', 'Unknown Organization')
        resource_type = result.get('resource_type', 'general')
        
        # Build card header
        ccsf_indicator = "🎓 " if is_ccsf else ""
        card_class = "resource-card resource-card-ccsf" if is_ccsf else "resource-card"
        
        with st.expander(f"{ccsf_indicator}{org_name}", expanded=(i <= 3)):
            # Card content
            col_main, col_side = st.columns([3, 1])
            
            with col_main:
                # Badges row
                badges_html = f'<span class="resource-type-badge">{resource_type}</span>'
                if is_ccsf:
                    badges_html += '<span class="ccsf-badge">🎓 CCSF Campus</span>'
                st.markdown(badges_html, unsafe_allow_html=True)
                
                # Description
                if result.get('description'):
                    st.markdown(f"*{result['description'][:300]}{'...' if len(result.get('description', '')) > 300 else ''}*")
                
                st.markdown("**Contact Information**")
                
                # Contact details in a clean format
                contact_html = ""
                if result.get('phone'):
                    contact_html += f'<div class="contact-item">📱 {result["phone"]}</div>'
                if result.get('email'):
                    contact_html += f'<div class="contact-item">📧 {result["email"]}</div>'
                if result.get('website'):
                    contact_html += f'<div class="contact-item">🌐 <a href="{result["website"]}" target="_blank">{result["website"]}</a></div>'
                if result.get('address'):
                    contact_html += f'<div class="contact-item">📍 {result["address"]}</div>'
                if result.get('hours'):
                    contact_html += f'<div class="contact-item">🕐 {result["hours"]}</div>'
                
                if contact_html:
                    st.markdown(contact_html, unsafe_allow_html=True)
                else:
                    st.markdown("_Contact information not available_")
                
                # Eligibility
                if result.get('eligibility'):
                    st.markdown(f"**Eligibility:** {result['eligibility'][:200]}{'...' if len(result.get('eligibility', '')) > 200 else ''}")
            
            with col_side:
                # Match score
                if 'similarity_score' in result and result['similarity_score'] is not None:
                    score = result['similarity_score']
                    st.markdown(f'<div class="match-score">{score:.0%} match</div>', unsafe_allow_html=True)
                
                # Status
                if result.get('is_current'):
                    st.markdown('<div class="status-current">✓ Current</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-outdated">⚠ May be outdated</div>', unsafe_allow_html=True)
            
            # Expandable sections for image and OCR
            st.markdown("---")
            
            col_img, col_ocr = st.columns(2)
            
            with col_img:
                file_path = result.get('file_path', '')
                # Only show image viewer for actual image files (not manual_entry or text imports)
                is_image_file = (
                    file_path and 
                    file_path != 'manual_entry' and 
                    not file_path.endswith('.txt') and
                    not file_path.startswith('web_import/')
                )
                if is_image_file:
                    show_image = st.checkbox("🖼️ View original image", key=f"img_{result['resource_id']}_{i}")
                    if show_image:
                        try:
                            from PIL import Image
                            from pillow_heif import register_heif_opener
                            register_heif_opener()
                            
                            img_key = f"rot_{result['resource_id']}_{i}"
                            if img_key not in st.session_state:
                                st.session_state[img_key] = 0
                            
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.button("↺", key=f"l_{i}", on_click=rotate_left, args=(img_key,))
                            with c2:
                                st.button("↻", key=f"r_{i}", on_click=rotate_right, args=(img_key,))
                            with c3:
                                if st.button("Reset", key=f"rs_{i}"):
                                    st.session_state[img_key] = 0
                            
                            img = Image.open(file_path)
                            if st.session_state[img_key] != 0:
                                img = img.rotate(st.session_state[img_key], expand=True)
                            st.image(img, use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not load image: {e}")
            
            with col_ocr:
                if result.get('ocr_text'):
                    show_ocr = st.checkbox("📄 View OCR text", key=f"ocr_{result['resource_id']}_{i}")
                    if show_ocr:
                        st.text_area(
                            "Extracted text",
                            result['ocr_text'][:1500],
                            height=200,
                            label_visibility="collapsed"
                        )

elif search_clicked:
    st.info("No resources found. Try different search terms or broaden your filters.")

# Browse All Section
st.markdown('<div class="section-header">📋 Browse All Resources</div>', unsafe_allow_html=True)

if st.checkbox("Show all resources"):
    with ResourceDatabase('wrc_resources.db') as db:
        all_resources = db.get_all_resources(current_only=current_only)
    
    df = pd.DataFrame(all_resources)
    
    display_cols = ['organization_name', 'resource_type', 'phone', 'address', 'is_current']
    available_cols = [col for col in display_cols if col in df.columns]
    
    st.dataframe(
        df[available_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "organization_name": st.column_config.TextColumn("Organization", width="large"),
            "resource_type": st.column_config.TextColumn("Type", width="small"),
            "phone": st.column_config.TextColumn("Phone", width="medium"),
            "address": st.column_config.TextColumn("Address", width="large"),
            "is_current": st.column_config.CheckboxColumn("Current", width="small")
        }
    )
    
    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download CSV",
        data=csv,
        file_name="wrc_resources.csv",
        mime="text/csv"
    )

# Footer
st.markdown("""
<div class="footer">
    <div>💜 Women's Resource Center @ City College of San Francisco</div>
    <div style="font-size: 0.8rem; margin-top: 0.5rem; opacity: 0.7;">
        Powered by AI • Helping connect people with resources
    </div>
</div>
""", unsafe_allow_html=True)
