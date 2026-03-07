import streamlit as st

# Import copied modules (from v2_toolkit/modules)
from modules.text_checker import check_text_authenticity
from modules.image_checker import check_image_authenticity
from modules.background_remover import remove_background
from modules.image_enhancer import enhance_image
from streamlit_image_comparison import image_comparison
from auth import init_db, login_user, register_user
init_db()

import cv2
import numpy as np


def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

import streamlit as st

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "trial_used" not in st.session_state:
    st.session_state.trial_used = False

tool = None


# ==================================
# PAGE CONFIG
# ==================================
st.set_page_config(
    page_title="AI Forensics Toolkit",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI Forensics & Enhancement Toolkit (v2)")

st.caption(
    "Development Version — Experimental Toolkit Expansion"
)

st.divider()

st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f1117 0%, #141922 100%);
}

.stMetric {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.05);
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg,#4cc9f0,#4361ee);
}

h1, h2, h3 {
    letter-spacing: -0.5px;
}
</style>
""", unsafe_allow_html=True)

# NEW UI BLOCK (paste here)
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: #0e1117;
    border-right: 1px solid rgba(255,255,255,0.05);
}

[data-testid="stSidebar"] h1 {
    font-size: 22px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    input:-webkit-autofill {
        background-color: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==================================
# SIDEBAR NAVIGATION
# ==================================
st.sidebar.title("V2 Toolkit")

if not st.session_state.logged_in:

    page = st.sidebar.radio(
        "Welcome",
        ["Try Demo", "Login", "Register"]
    )

    if page == "Try Demo":

        if st.session_state.trial_used:
            st.warning(
                "Trial finished. Please login or register to continue using V2 Toolkit."
            )
        else:
            st.sidebar.info("Trial Mode: One tool use allowed")

            tool = st.sidebar.selectbox(
                "Select Trial Tool",
                [
                    "🏠 Dashboard",
                    "✍️ AI Text Detector"
                ]
            )

    elif page == "Login":

        st.subheader("Login")

        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid username or password")

    elif page == "Register":

        st.subheader("Register")

        username = st.text_input("Create Username", key="register_user")
        password = st.text_input("Create Password", type="password", key="register_pass")

        if st.button("Create Account"):

            if username.strip() == "" or password.strip() == "":
                st.error("Username and password cannot be empty")

            else:
                if register_user(username, password):
                    st.success("Account created. You can now login.")
                else:
                    st.error("Username already exists")

else:

    st.sidebar.success("Logged in")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    tool = st.sidebar.radio(
        "Select Tool",
        [
            "🏠 Dashboard",
            "✍️ AI Text Detector",
            "🖼️ AI Image Detector",
            "🪄 Image Enhancer",
            "🧽 Background Remover"
        ]
    )


if not st.session_state.logged_in and tool == "✍️ AI Text Detector":

    st.title("Demo Mode - AI Text Detector")

    # BLOCK DEMO IF ALREADY USED
    if st.session_state.trial_used:
        st.warning(
            "Trial finished. Please login or register to continue using V2 Toolkit."
        )
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload a text file (.txt)",
        type=["txt"],
        key="demo_text_upload"
    )

    if uploaded_file:
        text = uploaded_file.read().decode("utf-8", errors="ignore")

        if st.button("Analyze Text"):

            with st.spinner("Analyzing text..."):
                result = check_text_authenticity(text)

            # MARK TRIAL AS USED
            st.session_state.trial_used = True

            ai_probability = result["risk"]

            st.metric("AI Probability", f"{ai_probability}%")

            if ai_probability > 85:
                st.error("Likely AI Generated")
            elif ai_probability < 25:
                st.success("Likely Human Written")
            else:
                st.warning("Uncertain Result")

    st.caption("Demo mode allows only one analysis. Login for full toolkit access.")
    st.stop()

if tool == "🏠 Dashboard":

    st.markdown("""
    <div style='text-align:center;padding:80px 20px;'>
        <h1 style='font-size:48px;font-weight:700;margin-bottom:10px;'>
            AI Forensics Toolkit
        </h1>
        <p style='font-size:20px;color:#9aa0a6;max-width:700px;margin:auto;'>
            Advanced deep-learning powered authenticity analysis for text and images.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style='background:#161b22;padding:30px;border-radius:20px;'>
            <h3>🖼️ Image Analysis</h3>
            <p style='color:#9aa0a6;'>
                Detect AI-generated imagery using ResNet deep embeddings.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background:#161b22;padding:30px;border-radius:20px;'>
            <h3>✍️ Text Analysis</h3>
            <p style='color:#9aa0a6;'>
                Identify machine-generated writing using transformer models.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ==================================
# TEXT TOOL
# ==================================
elif tool == "✍️ AI Text Detector":
    st.subheader("📝 Text Authorship Analysis")

    uploaded_file = st.file_uploader(
        "Upload a text file (.txt)",
        type=["txt"],
        key="main_text_upload"
    )

    if uploaded_file:
        text = uploaded_file.read().decode("utf-8", errors="ignore")
        result = check_text_authenticity(text)

        ai_probability = result["risk"]

        with st.container():
            st.markdown("## 🔍 Detection Result")
            st.markdown("---")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.metric("AI Probability", f"{ai_probability}%")

            with col2:
                st.progress(ai_probability / 100)

            st.markdown("---")

            if ai_probability > 85:
                st.markdown(
                    "<div style='background-color:#3a0d0d;padding:15px;border-radius:12px;text-align:center;'>"
                    "<h3 style='color:#ff4d4d;'>🔴 Likely AI-Generated</h3></div>",
                    unsafe_allow_html=True
                )
            elif ai_probability < 25:
                st.markdown(
                    "<div style='background-color:#0d3a1f;padding:15px;border-radius:12px;text-align:center;'>"
                    "<h3 style='color:#00ff88;'>🟢 Likely Human-Created</h3></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div style='background-color:#3a330d;padding:15px;border-radius:12px;text-align:center;'>"
                    "<h3 style='color:#ffd166;'>🟡 Mixed / Uncertain Signals</h3></div>",
                    unsafe_allow_html=True
                )
            st.caption("Powered by DistilBERT embeddings + RandomForest.")

# ==================================
# IMAGE TOOL
# ==================================
elif tool == "🖼️ AI Image Detector":
    st.subheader("🖼️ Image Authorship Analysis")

    uploaded_file = st.file_uploader(
        "Upload an image (.jpg, .png)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)
        result = check_image_authenticity(uploaded_file)
        ai_probability = result["ai_probability"]

        with st.container():
            st.markdown("## 🔍 Detection Result")
            st.markdown("---")

            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                st.metric("AI Probability", f"{ai_probability}%")
                st.progress(ai_probability / 100)

            st.markdown("---")

            if ai_probability > 85:
                st.markdown(
                    "<div style='background-color:#3a0d0d;padding:15px;border-radius:12px;text-align:center;'>"
                    "<h3 style='color:#ff4d4d;'>🔴 Likely AI-Generated</h3></div>",
                    unsafe_allow_html=True
                )
            elif ai_probability < 25:
                st.markdown(
                    "<div style='background-color:#0d3a1f;padding:15px;border-radius:12px;text-align:center;'>"
                    "<h3 style='color:#00ff88;'>🟢 Likely Human-Created</h3></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div style='background-color:#3a330d;padding:15px;border-radius:12px;text-align:center;'>"
                    "<h3 style='color:#ffd166;'>🟡 Mixed / Uncertain Signals</h3></div>",
                    unsafe_allow_html=True
                )

            st.caption("Powered by ResNet18 embeddings + RandomForest.")

# ==================================
# BACKGROUND REMOVER TOOL
# ==================================
elif tool == "🧽 Background Remover":
    st.subheader("🖼 Background Removal Tool")

    uploaded_file = st.file_uploader(
        "Upload an image (.jpg, .png)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        st.image(uploaded_file, caption="Original Image", use_container_width=True)

        if st.button("Remove Background"):
            with st.spinner("Removing background..."):
                result_image = remove_background(uploaded_file)

            if result_image is not None:
                with st.container():
                    st.markdown("## 🧽 Background Removal Result")
                    st.markdown("---")

                    col1, col2, col3 = st.columns([1, 3, 1])

                with col2:
                        st.image(result_image, use_container_width=True)

                st.markdown("---")
                st.success("✨ Background successfully removed.")

# ==================================
# IMAGE ENHANCER TOOL
# ==================================
elif tool == "🪄 Image Enhancer":
    st.subheader("✨ Image Quality Enhancer")

    uploaded_file = st.file_uploader(
        "Upload an image (.jpg, .png)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        original_bytes = uploaded_file.read()

        # Decode original
        import cv2
        import numpy as np

        file_array = np.asarray(bytearray(original_bytes), dtype=np.uint8)
        original_image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

        h, w = original_image.shape[:2]

        st.image(original_image[:, :, ::-1], caption=f"Original ({w} x {h})", use_container_width=True)

        if st.button("Enhance Image"):

            with st.spinner("Enhancing image quality..."):
                enhanced_image = enhance_image(original_bytes)

            if enhanced_image is not None:

                eh, ew = enhanced_image.shape[:2]

                with st.container():
                    st.markdown("## 🪄 Enhancement Result")
                    st.markdown("---")

                # Center comparison slider
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col2:
                        image_comparison(
                            img1=original_image[:, :, ::-1],
                            img2=enhanced_image,
                            label1="Original",
                            label2="Enhanced",
                            width=700,
                    )

                    st.markdown("---")

                    # Resolution Metrics
                    colA, colB = st.columns(2)

                    with colA:
                        st.metric("Original Resolution", f"{w} x {h}")

                    with colB:
                        st.metric("Enhanced Resolution", f"{ew} x {eh}")

                # Sharpness Metrics
                original_sharpness = calculate_sharpness(original_image)
                enhanced_bgr = enhanced_image[:, :, ::-1]
                enhanced_sharpness = calculate_sharpness(enhanced_bgr)

                colC, colD = st.columns(2)

                with colC:
                    st.metric("Original Sharpness", f"{original_sharpness:.2f}")

                with colD:
                    st.metric("Enhanced Sharpness", f"{enhanced_sharpness:.2f}")

                st.markdown("---")
                st.success("✨ Image enhancement complete.")        