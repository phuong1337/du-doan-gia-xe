import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(
    page_title="Dự đoán giá xe - Nhóm 7",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }
    .hero-card {
        background: radial-gradient(circle at top right, #2563eb 0%, #1e3a8a 45%, #0b1120 100%);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 18px;
        padding: 1.5rem 1.9rem;
        margin-bottom: 1.35rem;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.32);
    }
    .hero-pill {
        display: inline-block;
        background: rgba(96, 165, 250, 0.18);
        color: #bfdbfe;
        border: 1px solid rgba(147, 197, 253, 0.35);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.4px;
        padding: 3px 11px;
        border-radius: 999px;
        margin-bottom: 0.6rem;
    }
    .hero-card h1 {
        color: #ffffff !important;
        font-size: 1.72rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
    }
    .hero-card p {
        color: #e2e8f0 !important;
        margin: 0;
        font-size: 0.92rem;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0.4rem 0 0.75rem 0;
    }
    .result-box {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.42), rgba(15, 23, 42, 0.88));
        border: 1px solid rgba(96, 165, 250, 0.35);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        margin: 0.8rem 0 1rem 0;
    }
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.16);
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.84rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 750 !important;
    }
    .stButton > button[kind="primary"] {
        border-radius: 12px;
        height: 3rem;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.35);
    }
    .stButton > button:not([kind="primary"]) {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <span class="hero-pill">AI & MACHINE LEARNING • NHÓM 7</span>
        <h1>🚗 HỆ THỐNG DỰ ĐOÁN GIÁ XE Ô TÔ & PHÂN CỤM THỊ TRƯỜNG</h1>
        <p><b>Học phần:</b> Trí tuệ nhân tạo và ứng dụng &nbsp;•&nbsp; <b>GVHD:</b> Nguyễn Thị Thanh Hà &nbsp;•&nbsp; <b>Lớp:</b> 012008010412</p>
    </div>
    """,
    unsafe_allow_html=True,
)

TRAIN_FILES = ["nhom7_dudoangiaxe_train (1).xlsx", "nhom7_dudoangiaxe_train.xlsx"]
TEST_FILES = ["nhom7_dudoangiaxe_test (1).xlsx", "nhom7_dudoangiaxe_test.xlsx"]


def find_file(candidates):
    for name in candidates:
        if os.path.exists(name):
            return name
    return None


def clean_basic_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Mileage" in out.columns:
        out["Mileage"] = (
            out["Mileage"]
            .astype(str)
            .str.replace("km", "", case=False, regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        out["Mileage"] = pd.to_numeric(out["Mileage"], errors="coerce").fillna(0)

    if "Levy" in out.columns:
        out["Levy"] = (
            out["Levy"].astype(str).str.replace("-", "", regex=False).str.strip()
        )
        out["Levy"] = pd.to_numeric(out["Levy"], errors="coerce")
        median_levy = out["Levy"].median()
        out["Levy"] = out["Levy"].fillna(0 if pd.isna(median_levy) else median_levy)

    if "Engine volume" in out.columns:
        out["Engine volume"] = (
            out["Engine volume"]
            .astype(str)
            .str.replace("Turbo", "", case=False, regex=False)
            .str.strip()
        )
        out["Engine volume"] = pd.to_numeric(
            out["Engine volume"], errors="coerce"
        ).fillna(0.0)

    for num_col in ["Prod. year", "Cylinders", "Airbags", "Price"]:
        if num_col in out.columns:
            out[num_col] = pd.to_numeric(out[num_col], errors="coerce")

    return out


def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return df
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return df
    return df[
        (df[column] >= q1 - 1.5 * iqr) & (df[column] <= q3 + 1.5 * iqr)
    ].copy()


@st.cache_resource
def train_models_from_files():
    train_path = find_file(TRAIN_FILES)
    test_path = find_file(TEST_FILES)

    if not train_path:
        return None, "Không tìm thấy dữ liệu huấn luyện."

    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path) if test_path else None

    categorical_cols = [
        c
        for c in [
            "Manufacturer",
            "Model",
            "Category",
            "Leather interior",
            "Fuel type",
            "Gear box type",
            "Drive wheels",
            "Doors",
            "Wheel",
            "Color",
        ]
        if c in train_df.columns
    ]

    # Fit LabelEncoder theo thứ tự xuất hiện trên tập Train trước khi lọc ngoại lệ
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        vals = train_df[col].fillna("Missing").astype(str).unique()
        le.fit(vals)
        label_encoders[col] = le

    dt_df = clean_basic_columns(train_df)
    dt_df = dt_df.dropna(subset=["Price", "Prod. year"])
    dt_df = remove_outliers_iqr(dt_df, "Price")
    dt_df = remove_outliers_iqr(dt_df, "Mileage")

    for col in categorical_cols:
        le = label_encoders[col]
        known_set = set(le.classes_)
        fallback = le.classes_[0]
        dt_df[col] = (
            dt_df[col]
            .fillna("Missing")
            .astype(str)
            .apply(lambda x: x if x in known_set else fallback)
        )
        dt_df[col] = le.transform(dt_df[col])

    candidate_features = [
        "Levy",
        "Manufacturer",
        "Model",
        "Prod. year",
        "Category",
        "Leather interior",
        "Fuel type",
        "Engine volume",
        "Mileage",
        "Cylinders",
        "Gear box type",
        "Drive wheels",
        "Doors",
        "Wheel",
        "Color",
        "Airbags",
    ]
    features = [c for c in candidate_features if c in dt_df.columns]

    X_train = dt_df[features].fillna(0)
    y_train = dt_df["Price"]

    dt_model = DecisionTreeRegressor(max_depth=10, random_state=42)
    dt_model.fit(X_train, y_train)

    feat_imp = pd.DataFrame(
        {"Thuộc tính": features, "Mức độ quan trọng": dt_model.feature_importances_}
    ).sort_values(by="Mức độ quan trọng", ascending=False)

    test_metrics = None
    comparison_df = None
    if test_df is not None and not test_df.empty:
        t_df = clean_basic_columns(test_df)
        t_df = t_df.dropna(subset=["Price", "Prod. year"])
        t_df = remove_outliers_iqr(t_df, "Price")
        t_df = remove_outliers_iqr(t_df, "Mileage")

        for col in categorical_cols:
            if col in t_df.columns:
                le = label_encoders[col]
                known_set = set(le.classes_)
                fallback = le.classes_[0]
                t_df[col] = (
                    t_df[col]
                    .fillna("Missing")
                    .astype(str)
                    .apply(lambda x: x if x in known_set else fallback)
                )
                t_df[col] = le.transform(t_df[col])

        X_test = t_df[features].fillna(0)
        y_test = t_df["Price"]
        y_pred_test = dt_model.predict(X_test)

        test_metrics = {
            "samples": len(y_test),
            "mae": float(mean_absolute_error(y_test, y_pred_test)),
            "rmse": float(root_mean_squared_error(y_test, y_pred_test)),
            "r2": float(r2_score(y_test, y_pred_test)),
        }
        comparison_df = pd.DataFrame(
            {
                "Giá thực tế (USD)": y_test.values,
                "Giá dự đoán (USD)": np.round(y_pred_test, 2),
                "Sai lệch (USD)": np.round(np.abs(y_test.values - y_pred_test), 2),
            }
        )

    km_df = clean_basic_columns(train_df)
    cols_no_id = [c for c in km_df.columns if c != "ID"]
    km_df = km_df.drop_duplicates(subset=cols_no_id).copy()

    km_df = km_df[
        (km_df["Price"] > 500)
        & (km_df["Price"] < 100000)
        & (km_df["Mileage"] >= 1000)
        & (km_df["Mileage"] <= 500000)
        & (km_df["Engine volume"] > 0.6)
        & (km_df["Engine volume"] < 6.0)
        & (km_df["Prod. year"] > 1995)
    ].copy()

    km_df["Age"] = 2020 - km_df["Prod. year"]
    km_df["Log_Price"] = np.log(km_df["Price"])
    km_df["Log_Mileage"] = np.log(km_df["Mileage"])

    clustering_features = ["Log_Price", "Age", "Log_Mileage", "Engine volume"]
    km_df = km_df.dropna(subset=clustering_features).copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(km_df[clustering_features])

    kmeans = KMeans(n_clusters=5, n_init=20, random_state=42)
    raw_clusters = kmeans.fit_predict(X_scaled)
    km_df["Temp_Cluster"] = raw_clusters

    medians_by_raw = km_df.groupby("Temp_Cluster")["Price"].median().sort_values()
    cluster_mapping = {
        old_c: idx + 1 for idx, old_c in enumerate(medians_by_raw.index)
    }
    km_df["Cluster"] = km_df["Temp_Cluster"].map(cluster_mapping)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
    pca_df["Cluster"] = km_df["Cluster"].values

    cluster_names = {
        1: "Cụm 1 — Xe cũ giá siêu rẻ / Tiết kiệm kinh tế",
        2: "Cụm 2 — Xe phổ thông đời cũ",
        3: "Cụm 3 — Xe đời mới đại chúng",
        4: "Cụm 4 — Xe lướt cao cấp đời mới",
        5: "Cụm 5 — Xe SUV / Jeep hạng sang máy lớn",
    }

    summary_cols = [
        c
        for c in ["Price", "Age", "Mileage", "Engine volume", "Airbags"]
        if c in km_df.columns
    ]
    cluster_summary = km_df.groupby("Cluster")[summary_cols].median()
    cluster_counts = km_df["Cluster"].value_counts().sort_index()
    cluster_summary["Số lượng xe"] = cluster_counts
    cluster_summary["Tỷ lệ (%)"] = np.round((cluster_counts / len(km_df)) * 100, 2)

    artifacts = {
        "features": features,
        "categorical_cols": categorical_cols,
        "label_encoders": label_encoders,
        "dt_model": dt_model,
        "feat_imp": feat_imp,
        "test_metrics": test_metrics,
        "comparison_df": comparison_df,
        "scaler": scaler,
        "kmeans": kmeans,
        "cluster_mapping": cluster_mapping,
        "cluster_names": cluster_names,
        "cluster_summary": cluster_summary,
        "clustering_features": clustering_features,
        "pca_df": pca_df,
        "raw_test_df": test_df,
    }
    return artifacts, None


artifacts, err_msg = train_models_from_files()
if err_msg:
    st.error(err_msg)
    st.stop()

raw_test_df = artifacts["raw_test_df"]
les = artifacts["label_encoders"]

st.sidebar.header("⚙️ Tùy chỉnh")
vnd_rate = st.sidebar.number_input(
    "Tỷ giá (VNĐ / 1 USD)", min_value=20000, max_value=30000, value=25400, step=100
)
st.sidebar.markdown("---")
st.sidebar.caption("Tỷ lệ chia dữ liệu báo cáo: 80% Train — 20% Test")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🎯 Dự đoán & Phân cụm",
        "📂 Dự đoán hàng loạt",
        "📈 Hiệu năng Decision Tree",
        "🧩 Phân cụm K-Means (K=5)",
    ]
)


def get_options(col_name, fallback_list):
    if col_name in les:
        opts = [str(x) for x in les[col_name].classes_ if str(x) != "Missing"]
        return sorted(opts) if opts else fallback_list
    return fallback_list


def pick_existing(options, candidates):
    for item in candidates:
        if item in options:
            return item
    return options[0] if options else ""


def predict_single_car(input_dict):
    row_df = pd.DataFrame([input_dict])
    for col in artifacts["categorical_cols"]:
        if col in row_df.columns:
            le = les[col]
            val = str(row_df.loc[0, col])
            if val not in set(le.classes_):
                val = le.classes_[0]
            row_df[col] = le.transform([val])

    X_input = row_df[artifacts["features"]].fillna(0)
    pred_price = max(500.0, float(artifacts["dt_model"].predict(X_input)[0]))

    age_val = max(0.0, 2020.0 - float(input_dict["Prod. year"]))
    mileage_for_log = max(1000.0, float(input_dict["Mileage"]))
    km_input = pd.DataFrame(
        [
            {
                "Log_Price": np.log(pred_price),
                "Age": age_val,
                "Log_Mileage": np.log(mileage_for_log),
                "Engine volume": float(input_dict["Engine volume"]),
            }
        ]
    )[artifacts["clustering_features"]]

    km_scaled = artifacts["scaler"].transform(km_input)
    raw_c = int(artifacts["kmeans"].predict(km_scaled)[0])
    mapped_cluster = int(artifacts["cluster_mapping"][raw_c])
    return pred_price, mapped_cluster


m_opts = get_options("Manufacturer", ["TOYOTA", "HYUNDAI"])
md_opts = get_options("Model", ["Camry", "Elantra"])
cat_opts = get_options("Category", ["Sedan", "Jeep"])
lth_opts = get_options("Leather interior", ["Yes", "No"])
col_opts = get_options("Color", ["Black", "White", "Silver"])
fuel_opts = get_options("Fuel type", ["Petrol", "Diesel"])
gear_opts = get_options("Gear box type", ["Automatic", "Manual"])
drv_opts = get_options("Drive wheels", ["Front", "4x4"])
door_opts = get_options("Doors", ["04-May", "02-Mar"])
whl_opts = get_options("Wheel", ["Left wheel"])


def apply_preset(preset_type):
    if preset_type == "sedan":
        st.session_state["in_mfg"] = pick_existing(m_opts, ["HYUNDAI", "TOYOTA"])
        st.session_state["in_model"] = pick_existing(md_opts, ["Elantra", "Camry", "Sonata"])
        st.session_state["in_cat"] = pick_existing(cat_opts, ["Sedan"])
        st.session_state["in_year"] = 2015
        st.session_state["in_km"] = 110000
        st.session_state["in_eng"] = 1.8
        st.session_state["in_cyl"] = 4
        st.session_state["in_levy"] = 750
        st.session_state["in_airbags"] = 4
    elif preset_type == "suv":
        st.session_state["in_mfg"] = pick_existing(m_opts, ["LEXUS", "TOYOTA", "MERCEDES-BENZ"])
        st.session_state["in_model"] = pick_existing(md_opts, ["RX 350", "Land Cruiser", "GLE 350"])
        st.session_state["in_cat"] = pick_existing(cat_opts, ["Jeep", "SUV"])
        st.session_state["in_year"] = 2018
        st.session_state["in_km"] = 45000
        st.session_state["in_eng"] = 3.5
        st.session_state["in_cyl"] = 6
        st.session_state["in_levy"] = 1400
        st.session_state["in_airbags"] = 8
    elif preset_type == "new":
        st.session_state["in_mfg"] = pick_existing(m_opts, ["TOYOTA", "HYUNDAI", "KIA"])
        st.session_state["in_model"] = pick_existing(md_opts, ["Camry", "Tucson", "K5"])
        st.session_state["in_cat"] = pick_existing(cat_opts, ["Sedan", "Jeep"])
        st.session_state["in_year"] = 2020
        st.session_state["in_km"] = 18000
        st.session_state["in_eng"] = 2.0
        st.session_state["in_cyl"] = 4
        st.session_state["in_levy"] = 600
        st.session_state["in_airbags"] = 6


st.session_state.setdefault("in_mfg", pick_existing(m_opts, ["HYUNDAI", "TOYOTA"]))
st.session_state.setdefault("in_model", pick_existing(md_opts, ["Elantra", "Camry"]))
st.session_state.setdefault("in_cat", pick_existing(cat_opts, ["Sedan", "Jeep"]))
st.session_state.setdefault("in_leather", pick_existing(lth_opts, ["Yes", "No"]))
st.session_state.setdefault("in_color", pick_existing(col_opts, ["Silver", "Black", "White"]))
st.session_state.setdefault("in_year", 2016)
st.session_state.setdefault("in_km", 85000)
st.session_state.setdefault("in_eng", 2.0)
st.session_state.setdefault("in_cyl", 4)
st.session_state.setdefault("in_levy", 800)
st.session_state.setdefault("in_fuel", pick_existing(fuel_opts, ["Petrol", "Diesel"]))
st.session_state.setdefault("in_gear", pick_existing(gear_opts, ["Automatic", "Manual"]))
st.session_state.setdefault("in_drv", pick_existing(drv_opts, ["Front", "4x4"]))
st.session_state.setdefault("in_doors", pick_existing(door_opts, ["04-May", "02-Mar"]))
st.session_state.setdefault("in_wheel", pick_existing(whl_opts, ["Left wheel"]))
st.session_state.setdefault("in_airbags", 4)


with tab1:
    st.markdown('<div class="section-title">Chọn nhanh cấu hình mẫu</div>', unsafe_allow_html=True)
    p_col1, p_col2, p_col3, _ = st.columns([1.2, 1.2, 1.2, 2])
    p_col1.button("🚘 Sedan phổ thông", use_container_width=True, on_click=apply_preset, args=("sedan",))
    p_col2.button("🚙 SUV máy lớn", use_container_width=True, on_click=apply_preset, args=("suv",))
    p_col3.button("✨ Xe đời mới ít km", use_container_width=True, on_click=apply_preset, args=("new",))

    st.markdown('<div class="section-title">Thông số kỹ thuật xe</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        manufacturer = st.selectbox("Hãng xe", m_opts, key="in_mfg")
        model_name = st.selectbox("Dòng xe", md_opts, key="in_model")
        category = st.selectbox("Kiểu dáng", cat_opts, key="in_cat")
        leather = st.selectbox("Nội thất da", lth_opts, key="in_leather")
        color = st.selectbox("Màu sắc", col_opts, key="in_color")

    with c2:
        prod_year = st.number_input("Năm sản xuất", min_value=1990, max_value=2026, step=1, key="in_year")
        mileage = st.number_input("Quãng đường (km)", min_value=0, max_value=500000, step=1000, key="in_km")
        engine_volume = st.number_input("Dung tích động cơ (L)", min_value=0.5, max_value=7.0, step=0.1, format="%.1f", key="in_eng")
        cylinders = st.number_input("Số xi-lanh", min_value=2, max_value=16, step=1, key="in_cyl")
        levy = st.number_input("Thuế Levy (USD)", min_value=0, max_value=50000, step=50, key="in_levy")

    with c3:
        fuel_type = st.selectbox("Nhiên liệu", fuel_opts, key="in_fuel")
        gear_box = st.selectbox("Hộp số", gear_opts, key="in_gear")
        drive_wheels = st.selectbox("Hệ dẫn động", drv_opts, key="in_drv")
        doors = st.selectbox("Số cửa", door_opts, key="in_doors")
        wheel = st.selectbox("Vô lăng", whl_opts, key="in_wheel")

    airbags = st.slider("Số túi khí", min_value=0, max_value=16, key="in_airbags")

    current_car = {
        "Levy": levy,
        "Manufacturer": manufacturer,
        "Model": model_name,
        "Prod. year": prod_year,
        "Category": category,
        "Leather interior": leather,
        "Fuel type": fuel_type,
        "Engine volume": engine_volume,
        "Mileage": mileage,
        "Cylinders": cylinders,
        "Gear box type": gear_box,
        "Drive wheels": drive_wheels,
        "Doors": doors,
        "Wheel": wheel,
        "Color": color,
        "Airbags": airbags,
    }

    if st.button("🔍 DỰ ĐOÁN GIÁ XE & PHÂN CỤM", type="primary", use_container_width=True):
        pred_price, c_id = predict_single_car(current_car)
        st.session_state["last_result"] = {
            "car": current_car,
            "price": pred_price,
            "cluster": c_id,
        }

    if "last_result" in st.session_state:
        res = st.session_state["last_result"]
        pred_price = res["price"]
        c_id = res["cluster"]
        used_car = res["car"]
        cluster_label = artifacts["cluster_names"].get(c_id, f"Cụm {c_id}")

        if used_car != current_car:
            st.caption("ℹ️ Thông số đã thay đổi — bấm **DỰ ĐOÁN GIÁ XE & PHÂN CỤM** để cập nhật kết quả mới.")

        st.markdown(
            f"""
            <div class="result-box">
                <div style="color:#93c5fd; font-size:0.85rem; font-weight:600;">KẾT QUẢ ĐỊNH GIÁ & PHÂN KHÚC</div>
                <div style="color:#ffffff; font-size:1.25rem; font-weight:700; margin-top:0.2rem;">{cluster_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Giá dự đoán (USD)", f"{pred_price:,.2f} USD")
        m2.metric("Quy đổi tham khảo", f"{pred_price * vnd_rate:,.0f} đ")
        m3.metric("Phân khúc thị trường", f"Cụm {c_id} / 5")

        c_stats = artifacts["cluster_summary"].loc[c_id]
        compare_tbl = pd.DataFrame(
            {
                "Tiêu chí": ["Giá xe (USD)", "Số km (km)", "Dung tích (L)", "Túi khí"],
                "Xe đã dự đoán": [
                    f"{pred_price:,.2f}",
                    f"{int(used_car['Mileage']):,}",
                    f"{float(used_car['Engine volume']):.1f}",
                    f"{int(used_car['Airbags'])}",
                ],
                f"Trung vị Cụm {c_id}": [
                    f"{c_stats['Price']:,.2f}",
                    f"{c_stats['Mileage']:,.0f}",
                    f"{c_stats['Engine volume']:.1f}",
                    f"{c_stats['Airbags']:.0f}",
                ],
            }
        )
        st.markdown('<div class="section-title">Đối chiếu với mặt bằng phân khúc</div>', unsafe_allow_html=True)
        st.dataframe(compare_tbl, use_container_width=True, hide_index=True)

        with st.expander("⚖️ So sánh nhanh nếu đổi năm sản xuất hoặc số km"):
            sc1, sc2 = st.columns(2)
            alt_year = sc1.number_input("Năm sản xuất đối chiếu", min_value=1990, max_value=2026, value=min(2026, int(used_car["Prod. year"]) + 3), key="alt_y")
            alt_km = sc2.number_input("Số km đối chiếu", min_value=0, max_value=500000, value=max(0, int(used_car["Mileage"]) - 30000), step=5000, key="alt_k")
            alt_car = dict(used_car, **{"Prod. year": alt_year, "Mileage": alt_km})
            alt_price, alt_cid = predict_single_car(alt_car)
            diff = alt_price - pred_price
            st.write(
                f"Cấu hình đối chiếu (**{alt_year}**, **{alt_km:,} km**): **{alt_price:,.2f} USD** "
                f"({'+' if diff >= 0 else ''}{diff:,.2f} USD) — thuộc **Cụm {alt_cid}**."
            )

with tab2:
    st.subheader("Dự đoán hàng loạt từ Excel")
    col_u1, col_u2 = st.columns([2, 1])
    with col_u1:
        batch_file = st.file_uploader("Tải tệp Excel (.xlsx)", type=["xlsx"])
    with col_u2:
        use_sample_test = st.checkbox("Dùng thử 20 dòng mẫu", value=True)

    batch_df = None
    if batch_file is not None:
        batch_df = pd.read_excel(batch_file)
    elif use_sample_test and raw_test_df is not None:
        batch_df = raw_test_df.head(20).copy()

    if batch_df is not None:
        cleaned_batch = clean_basic_columns(batch_df)
        encoded_batch = cleaned_batch.copy()

        for col in artifacts["categorical_cols"]:
            if col in encoded_batch.columns:
                le = les[col]
                known_set = set(le.classes_)
                fallback = le.classes_[0]
                encoded_batch[col] = (
                    encoded_batch[col]
                    .fillna("Missing")
                    .astype(str)
                    .apply(lambda x: x if x in known_set else fallback)
                )
                encoded_batch[col] = le.transform(encoded_batch[col])

        for f in artifacts["features"]:
            if f not in encoded_batch.columns:
                encoded_batch[f] = 0

        preds = np.maximum(
            500.0, artifacts["dt_model"].predict(encoded_batch[artifacts["features"]].fillna(0))
        )
        ages = np.maximum(0.0, 2020.0 - cleaned_batch["Prod. year"].fillna(2015))
        mileages_log = np.log(np.maximum(1000.0, cleaned_batch["Mileage"].fillna(50000)))
        prices_log = np.log(preds)
        eng_vols = cleaned_batch["Engine volume"].fillna(2.0)

        km_batch = pd.DataFrame(
            {
                "Log_Price": prices_log,
                "Age": ages,
                "Log_Mileage": mileages_log,
                "Engine volume": eng_vols,
            }
        )
        km_batch_scaled = artifacts["scaler"].transform(km_batch)
        raw_c_batch = artifacts["kmeans"].predict(km_batch_scaled)
        mapped_c_batch = [artifacts["cluster_mapping"][int(c)] for c in raw_c_batch]

        result_df = batch_df.copy()
        result_df["Giá dự đoán (USD)"] = np.round(preds, 2)
        result_df["Phân cụm K-Means"] = [f"Cụm {c}" for c in mapped_c_batch]

        st.dataframe(result_df, use_container_width=True)
        st.download_button(
            label="📥 Tải kết quả (.CSV)",
            data=result_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="ket_qua_du_doan_nhom7.csv",
            mime="text/csv",
        )

with tab3:
    st.subheader("Hiệu năng mô hình Decision Tree Regressor")
    if artifacts["test_metrics"] is not None:
        m = artifacts["test_metrics"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Số mẫu đánh giá", f"{m['samples']:,} xe")
        k2.metric("MAE", f"{m['mae']:,.2f} USD")
        k3.metric("RMSE", f"{m['rmse']:,.2f} USD")
        k4.metric("R² Score", f"{m['r2']:.4f}")

    st.markdown("")
    c_left, c_right = st.columns([1.2, 1])
    with c_left:
        st.markdown('<div class="section-title">Top 10 thuộc tính ảnh hưởng mạnh nhất</div>', unsafe_allow_html=True)
        top10_imp = (
            artifacts["feat_imp"]
            .head(10)
            .sort_values(by="Mức độ quan trọng", ascending=True)
        )
        fig, ax = plt.subplots(figsize=(7, 4.4))
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
        ax.barh(top10_imp["Thuộc tính"], top10_imp["Mức độ quan trọng"], color="#60a5fa")
        ax.set_xlabel("Mức độ quan trọng", color="#cbd5e1")
        ax.tick_params(colors="#e2e8f0")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#475569")
        ax.spines["left"].set_color("#475569")
        ax.grid(axis="x", linestyle="--", alpha=0.25, color="#94a3b8")
        plt.tight_layout()
        st.pyplot(fig)

    with c_right:
        if artifacts["comparison_df"] is not None:
            st.markdown('<div class="section-title">Đối chiếu 10 mẫu đầu tiên</div>', unsafe_allow_html=True)
            st.dataframe(artifacts["comparison_df"].head(10), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Phân cụm thị trường bằng K-Means (K = 5)")
    col_km1, col_km2 = st.columns([1.15, 1])

    with col_km1:
        pca_df = artifacts["pca_df"]
        fig2, ax2 = plt.subplots(figsize=(7.4, 4.8))
        fig2.patch.set_alpha(0.0)
        ax2.set_facecolor("none")
        palette = ["#fb7185", "#38bdf8", "#4ade80", "#c084fc", "#fbbf24"]
        for idx, c_num in enumerate(sorted(pca_df["Cluster"].unique())):
            sub = pca_df[pca_df["Cluster"] == c_num]
            ax2.scatter(
                sub["PCA1"],
                sub["PCA2"],
                label=f"Cụm {c_num}",
                alpha=0.68,
                s=22,
                color=palette[idx % len(palette)],
            )
        ax2.set_title("Biểu đồ phân cụm K-Means (PCA 2D)", color="#f8fafc", fontsize=12)
        ax2.set_xlabel("PCA1", color="#cbd5e1")
        ax2.set_ylabel("PCA2", color="#cbd5e1")
        ax2.tick_params(colors="#e2e8f0")
        for spine in ["top", "right"]:
            ax2.spines[spine].set_visible(False)
        ax2.spines["bottom"].set_color("#475569")
        ax2.spines["left"].set_color("#475569")
        leg = ax2.legend(title="Phân khúc", facecolor="#0f172a", edgecolor="#334155")
        plt.setp(leg.get_texts(), color="#f8fafc")
        plt.setp(leg.get_title(), color="#94a3b8")
        ax2.grid(True, linestyle="--", alpha=0.2, color="#94a3b8")
        plt.tight_layout()
        st.pyplot(fig2)

    with col_km2:
        st.markdown('<div class="section-title">Giá trị trung vị theo từng cụm</div>', unsafe_allow_html=True)
        st.dataframe(artifacts["cluster_summary"].round(2), use_container_width=True)
        st.markdown(
            """
            - **Cụm 1:** Xe cũ giá siêu rẻ / Tiết kiệm kinh tế
            - **Cụm 2:** Xe phổ thông đời cũ
            - **Cụm 3:** Xe đời mới đại chúng
            - **Cụm 4:** Xe lướt cao cấp đời mới
            - **Cụm 5:** Xe SUV / Jeep hạng sang máy lớn
            """
        )

st.markdown("---")
st.caption("Trường Đại học Giao thông vận tải Thành phố Hồ Chí Minh (UTH) — Nhóm 7")