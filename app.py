import streamlit as st
import pandas as pd
import os
import sys
from src.logger import setup_logger
from src.recommendation_models import (
    RecommendationModel,
    reverse_encode_category,
    reverse_encode_review_rating,
)

# Thiết lập logger
logger = setup_logger(__name__, "app.log")

# Thêm đường dẫn src vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

st.set_page_config(page_title="🎯 Hybrid Recommendation System", layout="centered")

# Define data paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "shopping_behavior_updated.csv")
PROC_DATA_PATH = os.path.join(DATA_DIR, "shopping_behavior_processed.csv")


@st.cache_data
def load_data():
    logger.info(f"Loading data from {RAW_DATA_PATH} and {PROC_DATA_PATH}")
    try:
        df_processed = pd.read_csv(PROC_DATA_PATH)
        df_raw = pd.read_csv(RAW_DATA_PATH)
        logger.info("Data loaded successfully")
        return df_processed, df_raw
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


try:
    df_proc, df_raw = load_data()
    df_proc = reverse_encode_category(df_proc)
    df_proc = reverse_encode_review_rating(df_proc)
except Exception as e:
    logger.error(f"Error preprocessing data: {str(e)}")
    st.error(f"Không thể tải hoặc xử lý dữ liệu: {str(e)}")
    st.stop()

MIN_RATING, MAX_RATING = 0.0, 5.0


@st.cache_resource
def load_model():
    logger.info("Khởi tạo RecommendationModel...")
    try:
        model = RecommendationModel()
        logger.info("RecommendationModel được khởi tạo thành công")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"Không thể khởi tạo mô hình: {str(e)}")
        return None


model = load_model()
if model is None:
    st.stop()

st.sidebar.title("📦 Product Recommendation System")
page = st.sidebar.radio("📄 Chọn trang", ["Danh sách sản phẩm", "Gợi ý sản phẩm"])

if page == "Danh sách sản phẩm":
    st.title("📋 DANH SÁCH SẢN PHẨM")
    try:
        # Aggregate unique product info
        unique_products = (
            df_proc.groupby("Item Purchased")
            .agg(
                {
                    "Category": lambda x: ", ".join(sorted(x.dropna().unique())),
                    "Size": lambda x: ", ".join(sorted(x.dropna().unique())),
                    "Color": lambda x: ", ".join(sorted(x.dropna().unique())),
                }
            )
            .reset_index()
        )
        # Merge average ratings
        avg_ratings = (
            df_raw.groupby("Item Purchased")["Review Rating"]
            .mean()
            .round(1)
            .reset_index(name="Review Rating")
        )
        unique_products = unique_products.merge(avg_ratings, on="Item Purchased")
        st.subheader("🎯 Thông tin sản phẩm hiện có:")
        view_mode = st.radio("Kiểu hiển thị", ["Danh sách", "Lưới"], horizontal=True)

        def get_icon(item_name):
            name = item_name.lower()
            if "shirt" in name or "top" in name or "t-shirt" in name:
                return "👕"
            if "shoe" in name or "sneaker" in name or "boots" in name:
                return "👟"
            if "bag" in name or "handbag" in name or "tote" in name:
                return "👜"
            if "jacket" in name or "coat" in name:
                return "🧥"
            if "watch" in name:
                return "⌚"
            if "dress" in name or "skirt" in name:
                return "👗"
            if "ring" in name:
                return "💍"
            if "hat" in name or "cap" in name:
                return "🧢"
            if "sock" in name:
                return "🧦"
            if "sweater" in name:
                return "🧶"
            return "🛍️"

        if view_mode == "Danh sách":
            # Render as vertical list, not a table
            for idx, p in enumerate(unique_products.to_dict(orient="records"), 1):
                icon = get_icon(p["Item Purchased"])
                rating = (
                    "⭐" * int(round(p["Review Rating"]))
                    + f" ({p['Review Rating']:.1f})"
                )
                st.markdown(
                    f"""
                    <div style='display:flex;align-items:center;padding:11px 0;border-bottom:1px solid #ececec'>
                      <div style='font-size:1.7rem;width:42px;text-align:center'>{icon}</div>
                      <div style='flex:1'>
                        <b style='font-size:1.08rem'>{idx}. {p['Item Purchased']}</b><br>
                        <span style='color:#666;font-size:0.97rem'>Category: {p['Category']} | Size: {p['Size']} | Color: {p['Color']}</span><br>
                        <span style='color:#f59e42;font-size:1.02rem'>{rating}</span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            n_cols = 3
            prods = unique_products.to_dict(orient="records")
            for i in range(0, len(prods), n_cols):
                cols = st.columns(n_cols)
                for j in range(n_cols):
                    idx = i + j
                    if idx < len(prods):
                        p = prods[idx]
                        icon = get_icon(p["Item Purchased"])
                        st_html = f"""
                        <div style='background:#fff;padding:14px 10px 10px 10px;border-radius:12px;box-shadow:0 1px 4px #ccc;margin-bottom:14px;'>
                          <div style='font-size:2rem'>{icon}</div>
                          <div style='font-weight:700;font-size:1.05rem;margin-bottom:4px'>{p['Item Purchased']}</div>
                          <div style='color:#666;font-size:0.95rem;margin-bottom:2px'>Category: {p['Category']}</div>
                          <div style='color:#666;font-size:0.95rem;'>Size: {p['Size']}, Color: {p['Color']}</div>
                          <div style='margin-top:4px;font-size:1rem;color:#f59e42;'>⭐ {p['Review Rating']}</div>
                        </div>
                        """
                        cols[j].markdown(st_html, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Error displaying product list: {str(e)}")
        st.error(f"Lỗi khi hiển thị danh sách sản phẩm: {str(e)}")

else:
    st.markdown(
        "<h1 style='text-align:center;'>🎯 GỢI Ý SẢN PHẨM</h1>", unsafe_allow_html=True
    )
    mode = st.radio("Chọn phương pháp gợi ý:", ("Theo cá nhân", "Theo nội dung"))
    recs = []

    if mode == "Theo cá nhân":
        # --- PERSONALIZED RECOMMENDATION FORM ---
        with st.form("personal_form"):
            customer_ids = df_proc["Customer ID"].dropna().astype(int).unique()
            customer_ids = sorted(customer_ids)
            selected = st.selectbox(
                "Chọn Customer ID:", [""] + [str(cid) for cid in customer_ids]
            )
            top_n = st.slider(
                "Số lượng sản phẩm gợi ý:", min_value=1, max_value=10, value=5
            )
            col_submit, col_clear = st.columns([2, 1])
            submit_personal = col_submit.form_submit_button("🚀 Đề xuất")
            clear_personal = col_clear.form_submit_button("🧹 Xóa lựa chọn")
        if clear_personal:
            st.session_state["personal_selected"] = ""
            st.session_state["personal_top_n"] = 5
            st.session_state["personal_recs"] = []
            st.rerun()
        if "personal_selected" not in st.session_state:
            st.session_state["personal_selected"] = ""
        if "personal_top_n" not in st.session_state:
            st.session_state["personal_top_n"] = 5
        if "personal_recs" not in st.session_state:
            st.session_state["personal_recs"] = []
        if selected != st.session_state["personal_selected"]:
            st.session_state["personal_selected"] = selected
        if top_n != st.session_state["personal_top_n"]:
            st.session_state["personal_top_n"] = top_n

        if submit_personal and selected:
            with st.spinner("Đang tạo đề xuất..."):
                cid = int(selected)
                try:
                    logger.info(f"Đang tạo đề xuất cho Customer ID: {cid}")
                    recs = model.get_recommendations(cid, top_n)
                    st.session_state["personal_recs"] = recs
                except Exception as e:
                    logger.error(f"Lỗi khi đề xuất cho Customer ID {cid}: {str(e)}")
                    st.error(f"Lỗi khi đề xuất: {str(e)}")
        recs = st.session_state.get("personal_recs", [])
        if not selected:
            st.info("Vui lòng chọn Customer ID.")

    else:
        # --- CONTENT-BASED RECOMMENDATION FORM ---
        category_options = sorted(df_raw["Category"].dropna().unique())
        location_options = sorted(df_raw["Location"].dropna().unique())
        season_options = sorted(df_raw["Season"].dropna().unique())

        with st.form("content_form"):
            st.markdown("#### Thông tin sản phẩm")
            selected_categories = st.multiselect("Category", options=category_options)
            selected_locations = st.multiselect("Location", options=location_options)
            selected_seasons = st.multiselect("Season", options=season_options)
            st.markdown("#### Đánh giá")
            min_rating = st.slider(
                "Rating tối thiểu", MIN_RATING, MAX_RATING, MIN_RATING, step=0.1
            )
            top_n = st.slider(
                "Số lượng sản phẩm gợi ý", min_value=1, max_value=10, value=5
            )
            col_submit, col_clear = st.columns([2, 1])
            submit_content = col_submit.form_submit_button("🚀 Đề xuất")
            clear_content = col_clear.form_submit_button("🧹 Xóa tất cả bộ lọc")
        if clear_content:
            st.session_state["selected_categories"] = []
            st.session_state["selected_locations"] = []
            st.session_state["selected_seasons"] = []
            st.session_state["min_rating"] = MIN_RATING
            st.session_state["content_recs"] = []
            st.rerun()
        if "selected_categories" not in st.session_state:
            st.session_state["selected_categories"] = []
        if "selected_locations" not in st.session_state:
            st.session_state["selected_locations"] = []
        if "selected_seasons" not in st.session_state:
            st.session_state["selected_seasons"] = []
        if "min_rating" not in st.session_state:
            st.session_state["min_rating"] = MIN_RATING
        if "content_recs" not in st.session_state:
            st.session_state["content_recs"] = []

        # Sync current UI with session state
        if selected_categories != st.session_state["selected_categories"]:
            st.session_state["selected_categories"] = selected_categories
        if selected_locations != st.session_state["selected_locations"]:
            st.session_state["selected_locations"] = selected_locations
        if selected_seasons != st.session_state["selected_seasons"]:
            st.session_state["selected_seasons"] = selected_seasons
        if min_rating != st.session_state["min_rating"]:
            st.session_state["min_rating"] = min_rating

        # Assemble filters
        filters = {}
        if st.session_state["selected_categories"]:
            filters["Category"] = st.session_state["selected_categories"]
        if st.session_state["selected_locations"]:
            filters["Location"] = st.session_state["selected_locations"]
        if st.session_state["selected_seasons"]:
            filters["Season"] = st.session_state["selected_seasons"]
        if st.session_state["min_rating"] > MIN_RATING:
            filters["Review Rating"] = st.session_state["min_rating"]

        if submit_content:
            with st.spinner("Đang lọc..."):
                if not filters:
                    st.warning("Cần chọn ít nhất 1 trường lọc để hiển thị gợi ý.")
                    st.session_state["content_recs"] = []
                else:
                    try:
                        # Xử lý filter dạng list (multiselect)
                        df_filtered = df_raw.copy()
                        if "Category" in filters:
                            df_filtered = df_filtered[
                                df_filtered["Category"].isin(filters["Category"])
                            ]
                        if "Location" in filters:
                            df_filtered = df_filtered[
                                df_filtered["Location"].isin(filters["Location"])
                            ]
                        if "Season" in filters:
                            df_filtered = df_filtered[
                                df_filtered["Season"].isin(filters["Season"])
                            ]
                        if "Review Rating" in filters:
                            df_filtered = df_filtered[
                                df_filtered["Review Rating"] >= filters["Review Rating"]
                            ]
                        if not df_filtered.empty:
                            item_scores = (
                                df_filtered.groupby("Item Purchased")["Review Rating"]
                                .mean()
                                .reset_index()
                            )
                            item_scores = item_scores.sort_values(
                                by="Review Rating", ascending=False
                            )
                            recs = list(
                                zip(
                                    item_scores["Item Purchased"],
                                    item_scores["Review Rating"],
                                )
                            )[:top_n]
                            st.session_state["content_recs"] = recs
                        else:
                            st.session_state["content_recs"] = []
                    except Exception as e:
                        logger.error(f"Lỗi khi đề xuất với filter: {str(e)}")
                        st.error(f"Lỗi khi đề xuất: {str(e)}")
        recs = st.session_state.get("content_recs", [])
        if not filters:
            st.info("Chọn ít nhất 1 trường lọc để hiển thị gợi ý sản phẩm.")

    # --- DISPLAY RECOMMENDATIONS ---
    if recs:
        st.subheader("🔮 Top sản phẩm gợi ý:")
        view_mode = st.radio("Kiểu hiển thị", ["Danh sách", "Lưới"], horizontal=True)

        # Icon mapping
        def get_icon(name):
            name = name.lower()
            if "shirt" in name or "top" in name or "t-shirt" in name:
                return "👕"
            if "shoe" in name or "sneaker" in name or "boots" in name:
                return "👟"
            if "bag" in name or "handbag" in name or "tote" in name:
                return "👜"
            if "jacket" in name or "coat" in name:
                return "🧥"
            if "watch" in name:
                return "⌚"
            if "dress" in name or "skirt" in name:
                return "👗"
            if "ring" in name:
                return "💍"
            if "hat" in name or "cap" in name:
                return "🧢"
            if "sock" in name:
                return "🧦"
            if "sweater" in name:
                return "🧶"
            return "🛍️"

        ICON_TOOLTIP = {
            "👕": "Áo, T-shirt",
            "👗": "Váy, đầm",
            "👟": "Giày dép",
            "👜": "Túi xách",
            "🧥": "Áo khoác",
            "⌚": "Đồng hồ",
            "💍": "Nhẫn, phụ kiện",
            "🧢": "Mũ, nón",
            "🧦": "Tất/vớ",
            "🧶": "Áo len",
            "🛍️": "Sản phẩm khác",
        }

        if view_mode == "Danh sách":
            for idx, (item, score) in enumerate(recs):
                percent = int(round(score / 5.0 * 100))
                is_top = idx == 0
                card_color = "#ffe066" if is_top else "#f8f9fa"
                border = "3px solid #f59e42" if is_top else "1px solid #dee2e6"
                shadow = "0 4px 16px #e3e3e3" if is_top else "0 1px 3px #e3e3e3"
                icon = get_icon(item)
                st.markdown(
                    f"""<div style='background:{card_color};
                    padding:18px 12px 14px 12px;
                    margin-bottom:10px;
                    border-radius:13px;
                    border:{border};
                    box-shadow:{shadow};
                    transition:box-shadow 0.2s;display:flex;align-items:center;'>
                    <div style='font-size:2.3rem;margin-right:15px' title='{ICON_TOOLTIP.get(icon,"")}'>{icon}</div>
                    <div>
                    <strong style='font-size:1.05rem;'>{idx+1}. {item}</strong><br>
                    <span style='font-size:1.07rem;'>⭐ <b>{score:.1f} / 5.0</b></span>
                    &nbsp;<span style='color:#888;font-size:0.94rem;'>({percent}%)</span>
                    </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            n_cols = 3
            for i in range(0, len(recs), n_cols):
                cols = st.columns(n_cols)
                for j in range(n_cols):
                    idx = i + j
                    if idx < len(recs):
                        item, score = recs[idx]
                        percent = int(round(score / 5.0 * 100))
                        is_top = idx == 0
                        card_color = "#ffe066" if is_top else "#f8f9fa"
                        border = "3px solid #f59e42" if is_top else "1px solid #dee2e6"
                        shadow = "0 4px 16px #e3e3e3" if is_top else "0 1px 3px #e3e3e3"
                        icon = get_icon(item)
                        tooltip = ICON_TOOLTIP.get(icon, "")
                        cols[j].markdown(
                            f"""<div style='background:{card_color};
                            padding:18px 12px 14px 12px;
                            margin-bottom:10px;
                            border-radius:13px;
                            border:{border};
                            box-shadow:{shadow};
                            transition:box-shadow 0.2s;'>
                            <div style='font-size:2.3rem;margin-bottom:8px' title='{tooltip}'>{icon}</div>
                            <strong style='font-size:1.02rem;'>{item}</strong><br>
                            <span style='font-size:1.1rem;'>⭐ <b>{score:.1f} / 5.0</b></span>
                            &nbsp;<span style='color:#888;font-size:0.94rem;'>({percent}%)</span>
                            </div>""",
                            unsafe_allow_html=True,
                        )
    # Custom CSS
    st.markdown(
        """
        <style>
            html, body { font-family: system-ui, Arial, sans-serif !important; }
            .recommendation-item:hover { background-color: #eef4ff; }
            .stAlert, .stInfo, .stWarning { font-size: 1rem; }
            .stButton > button, .stRadio > div { font-size: 1.08rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
