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
    page_title="Hệ thống AI Dự đoán giá xe - Nhóm 7",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #2563eb 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        color: white;
        font-size: 1.8rem;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #dbeafe;
        margin: 0;
        font-size: 0.95rem;
    }
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 12px 16px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-header">
        <h1>🚗 HỆ THỐNG AI DỰ ĐOÁN GIÁ XE Ô TÔ & PHÂN CỤM THỊ TRƯỜNG</h1>
        <p><b>Học phần:</b> Trí tuệ nhân tạo và ứng dụng &nbsp;|&nbsp; <b>GVHD:</b> Nguyễn Thị Thanh Hà &nbsp;|&nbsp; <b>Thực hiện:</b> Nhóm 7 (Lớp 012008010412)</p>
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
        return None, None, "Không tìm thấy dữ liệu huấn luyện."

    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path) if test_path else None

    dt_df = clean_basic_columns(train_df)
    dt_df = dt_df.dropna(subset=["Price", "Prod. year"])
    dt_df = remove_outliers_iqr(dt_df, "Price")
    dt_df = remove_outliers_iqr(dt_df, "Mileage")

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
        if c in dt_df.columns
    ]

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        dt_df[col] = le.fit_transform(dt_df[col].astype(str))
        label_encoders[col] = le

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
                "Sai lệch tuyệt đối (USD)": np.round(
                    np.abs(y_test.values - y_pred_test), 2
                ),
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
        1: "Cụm 1 - Xe cũ giá siêu rẻ / Tiết kiệm kinh tế",
        2: "Cụm 2 - Xe phổ thông đời cũ",
        3: "Cụm 3 - Xe đời mới đại chúng (Cận sang)",
        4: "Cụm 4 - Xe lướt cao cấp đời mới",
        5: "Cụm 5 - Xe SUV / Jeep hạng sang máy lớn",
    }

    summary_cols = [
        c
        for c in ["Price", "Age", "Mileage", "Engine volume", "Airbags"]
        if c in km_df.columns
    ]
    cluster_summary = km_df.groupby("Cluster")[summary_cols].median()
    cluster_counts = km_df["Cluster"].value_counts().sort_index()
    cluster_summary["Số lượng xe"] = cluster_counts
    cluster_summary["Tỷ lệ (%)"] = np.round(
        (cluster_counts / len(km_df)) * 100, 2
    )

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

st.sidebar.header("⚙️ Cấu hình hiển thị")
vnd_rate = st.sidebar.number_input(
    "Tỷ giá quy đổi (VNĐ / 1 USD)", min_value=20000, max_value=30000, value=25400, step=100
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🎯 Dự đoán & Phân cụm xe",
        "📂 Dự đoán hàng loạt (Excel)",
        "📈 Hiệu năng Decision Tree",
        "🧩 Khám phá 5 cụm K-Means",
    ]
)

les = artifacts["label_encoders"]


def get_options(col_name, fallback_list):
    if col_name in les:
        opts = [str(x) for x in les[col_name].classes_]
        return sorted(opts) if opts else fallback_list
    return fallback_list


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


with tab1:
    st.subheader("Nhập thông số kỹ thuật chiếc xe cần định giá")
    col1, col2, col3 = st.columns(3)

    with col1:
        manufacturer = st.selectbox(
            "Hãng xe (Manufacturer)", get_options("Manufacturer", ["TOYOTA", "HYUNDAI"])
        )
        model_name = st.selectbox(
            "Dòng xe (Model)", get_options("Model", ["Camry", "Elantra"])
        )
        category = st.selectbox(
            "Kiểu dáng (Category)", get_options("Category", ["Sedan", "Jeep"])
        )
        leather = st.selectbox(
            "Nội thất da (Leather interior)", get_options("Leather interior", ["Yes", "No"])
        )
        color = st.selectbox(
            "Màu sắc (Color)", get_options("Color", ["Black", "White", "Silver"])
        )

    with col2:
        prod_year = st.number_input(
            "Năm sản xuất (Prod. year)", min_value=1990, max_value=2026, value=2016
        )
        mileage = st.number_input(
            "Số km đã đi (Mileage)", min_value=0, max_value=500000, value=85000, step=1000
        )
        engine_volume = st.number_input(
            "Dung tích động cơ (Engine volume - L)",
            min_value=0.5,
            max_value=7.0,
            value=2.0,
            step=0.1,
        )
        cylinders = st.number_input(
            "Số xi-lanh (Cylinders)", min_value=2, max_value=16, value=4, step=1
        )
        levy = st.number_input(
            "Thuế Levy (USD)", min_value=0, max_value=50000, value=800, step=50
        )

    with col3:
        fuel_type = st.selectbox(
            "Loại nhiên liệu (Fuel type)", get_options("Fuel type", ["Petrol", "Diesel"])
        )
        gear_box = st.selectbox(
            "Loại hộp số (Gear box type)",
            get_options("Gear box type", ["Automatic", "Manual"]),
        )
        drive_wheels = st.selectbox(
            "Hệ dẫn động (Drive wheels)", get_options("Drive wheels", ["Front", "4x4"])
        )
        doors = st.selectbox(
            "Số cửa (Doors)", get_options("Doors", ["04-May", "02-Mar"])
        )
        wheel = st.selectbox(
            "Vị trí vô lăng (Wheel)", get_options("Wheel", ["Left wheel"])
        )

    airbags = st.slider("Số túi khí an toàn (Airbags)", min_value=0, max_value=16, value=4)

    if st.button("🔍 DỰ ĐOÁN GIÁ XE & PHÂN KHÚC NGAY", type="primary", use_container_width=True):
        user_car = {
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
        pred_price, c_id = predict_single_car(user_car)
        pred_vnd = pred_price * vnd_rate
        cluster_label = artifacts["cluster_names"].get(c_id, f"Cụm {c_id}")

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("💰 Giá dự đoán (USD)", f"{pred_price:,.2f} USD")
        m2.metric("🇻🇳 Quy đổi tham khảo (VNĐ)", f"{pred_vnd:,.0f} đ")
        m3.metric("🧩 Phân khúc thị trường", f"Cụm {c_id} / 5")

        st.info(f"**Chi tiết phân khúc K-Means:** {cluster_label}")

        c_stats = artifacts["cluster_summary"].loc[c_id]
        compare_tbl = pd.DataFrame(
            {
                "Tiêu chí": ["Giá xe (USD)", "Số km đã đi (km)", "Dung tích máy (L)", "Số túi khí"],
                "Xe của bạn": [
                    f"{pred_price:,.2f}",
                    f"{mileage:,}",
                    f"{engine_volume:.1f}",
                    f"{airbags}",
                ],
                f"Trung vị Cụm {c_id} trên thị trường": [
                    f"{c_stats['Price']:,.2f}",
                    f"{c_stats['Mileage']:,.0f}",
                    f"{c_stats['Engine volume']:.1f}",
                    f"{c_stats['Airbags']:.0f}",
                ],
            }
        )
        st.write("**Đối chiếu thông số xe của bạn với mặt bằng chung của phân khúc:**")
        st.table(compare_tbl)

with tab2:
    st.subheader("Dự đoán giá & phân cụm hàng loạt từ tệp Excel")
    st.write(
        "Tải lên tệp `.xlsx` chứa danh sách nhiều xe, hoặc dùng thử 20 mẫu trích từ tập kiểm thử."
    )

    col_up1, col_up2 = st.columns([2, 1])
    with col_up1:
        batch_file = st.file_uploader("Tải lên tệp Excel cần dự đoán (.xlsx)", type=["xlsx"])
    with col_up2:
        use_sample_test = st.checkbox("Dùng thử 20 dòng đầu mẫu", value=True)

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
                    .astype(str)
                    .apply(lambda x: x if x in known_set else fallback)
                )
                encoded_batch[col] = le.transform(encoded_batch[col])

        for f in artifacts["features"]:
            if f not in encoded_batch.columns:
                encoded_batch[f] = 0

        preds = artifacts["dt_model"].predict(encoded_batch[artifacts["features"]].fillna(0))
        preds = np.maximum(500.0, preds)

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

        csv_data = result_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 Tải xuống kết quả dự đoán (.CSV)",
            data=csv_data,
            file_name="ket_qua_du_doan_nhom7.csv",
            mime="text/csv",
        )

with tab3:
    st.subheader("Đánh giá hiệu năng mô hình Decision Tree Regressor")
    if artifacts["test_metrics"] is not None:
        m = artifacts["test_metrics"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Số mẫu đánh giá", f"{m['samples']:,} xe")
        k2.metric("Sai số tuyệt đối TB (MAE)", f"{m['mae']:,.2f} USD")
        k3.metric("Sai số toàn phương TB (RMSE)", f"{m['rmse']:,.2f} USD")
        k4.metric("Hệ số xác định (R² Score)", f"{m['r2']:.4f}")

    c_left, c_right = st.columns([1.2, 1])
    with c_left:
        st.write("**Top 10 thuộc tính ảnh hưởng mạnh nhất đến giá xe:**")
        top10_imp = artifacts["feat_imp"].head(10).sort_values(by="Mức độ quan trọng", ascending=True)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.barh(top10_imp["Thuộc tính"], top10_imp["Mức độ quan trọng"], color="#2563eb")
        ax.set_xlabel("Trọng số quan trọng (Feature Importance)")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

    with c_right:
        if artifacts["comparison_df"] is not None:
            st.write("**Đối chiếu 10 mẫu đầu tiên:**")
            st.dataframe(artifacts["comparison_df"].head(10), use_container_width=True)

with tab4:
    st.subheader("Trực quan hóa 5 phân khúc thị trường bằng K-Means & PCA")
    col_km1, col_km2 = st.columns([1.15, 1])

    with col_km1:
        pca_df = artifacts["pca_df"]
        fig2, ax2 = plt.subplots(figsize=(7.5, 5))
        colors = ["#e11d48", "#2563eb", "#16a34a", "#9333ea", "#f59e0b"]
        for idx, c_num in enumerate(sorted(pca_df["Cluster"].unique())):
            sub = pca_df[pca_df["Cluster"] == c_num]
            ax2.scatter(
                sub["PCA1"],
                sub["PCA2"],
                label=f"Cụm {c_num}",
                alpha=0.55,
                s=22,
                color=colors[idx % len(colors)],
            )
        ax2.set_title("Biểu đồ phân cụm K-Means (K = 5) giảm chiều PCA 2D")
        ax2.set_xlabel("Thành phần chính 1 (PCA1)")
        ax2.set_ylabel("Thành phần chính 2 (PCA2)")
        ax2.legend(title="Phân khúc")
        ax2.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig2)

    with col_km2:
        st.write("**Bảng thông số trung vị (Median) của 5 cụm:**")
        st.dataframe(artifacts["cluster_summary"].round(2), use_container_width=True)
        st.markdown(
            """
            - **Cụm 1:** Xe cũ giá siêu rẻ / Tiết kiệm kinh tế
            - **Cụm 2:** Xe phổ thông đời cũ (tuổi xe cao, chạy nhiều)
            - **Cụm 3:** Xe đời mới đại chúng (chiếm thị phần lớn nhất)
            - **Cụm 4:** Xe lướt cao cấp đời mới (ít km, đời rất gần)
            - **Cụm 5:** Xe SUV / Jeep hạng sang máy lớn (dung tích ~3.5L)
            """
        )

st.markdown("---")
st.caption("Trường Đại học Giao thông vận tải Thành phố Hồ Chí Minh (UTH) — Nhóm 7")