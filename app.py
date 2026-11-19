import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(
    page_title="Dự đoán giá xe & Phân cụm K-Means - Nhóm 7",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 HỆ THỐNG DỰ ĐOÁN GIÁ XE Ô TÔ VÀ PHÂN CỤM THỊ TRƯỜNG")
st.markdown("---")
st.markdown("- **Học phần:** Trí tuệ nhân tạo và ứng dụng[cite: 40]")
st.markdown("- **Giảng viên hướng dẫn:** Nguyễn Thị Thanh Hà[cite: 40]")
st.markdown("- **Nhóm thực hiện:** Nhóm 7 [cite: 40]")
st.markdown("---")

# Danh sách nhận diện chính xác các file Excel có trong repo của bạn
TRAIN_FILES = [
    "nhom7_dudoangiaxe_train (1).xlsx",
    "nhom7_dudoangiaxe_train.xlsx",
]
TEST_FILES = [
    "nhom7_dudoangiaxe_test (1).xlsx",
    "nhom7_dudoangiaxe_test.xlsx",
]


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
            out["Levy"]
            .astype(str)
            .str.replace("-", "", regex=False)
            .str.strip()
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
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()


@st.cache_resource
def train_models_from_files():
    train_path = find_file(TRAIN_FILES)
    test_path = find_file(TEST_FILES)

    if not train_path:
        return None, None, "Không tìm thấy file Train Excel trong kho lưu trữ!"

    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path) if test_path else None

    # 1. Huấn luyện Decision Tree Regressor theo báo cáo (max_depth=10, random_state=42)[cite: 54, 57]
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

    test_metrics = None
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

    # 2. Huấn luyện K-Means (K=5) theo báo cáo trên Log_Price, Age, Log_Mileage, Engine volume[cite: 60, 62, 72]
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

    final_k = 5
    kmeans = KMeans(n_clusters=final_k, n_init=20, random_state=42)
    raw_clusters = kmeans.fit_predict(X_scaled)
    km_df["Temp_Cluster"] = raw_clusters

    medians_by_raw = km_df.groupby("Temp_Cluster")["Price"].median().sort_values()
    cluster_mapping = {old_c: idx + 1 for idx, old_c in enumerate(medians_by_raw.index)}
    km_df["Cluster"] = km_df["Temp_Cluster"].map(cluster_mapping)

    cluster_names = {
        1: "Cụm 1 - Phân khúc xe cũ giá siêu rẻ / Tiết kiệm kinh tế",
        2: "Cụm 2 - Phân khúc xe phổ thông đời cũ",
        3: "Cụm 3 - Phân khúc xe đời mới đại chúng (Cận sang)",
        4: "Cụm 4 - Phân khúc xe lướt cao cấp đời mới",
        5: "Cụm 5 - Phân khúc xe SUV / Jeep hạng sang máy lớn",
    }

    summary_cols = [
        c for c in ["Price", "Age", "Mileage", "Engine volume", "Airbags"] if c in km_df.columns
    ]
    cluster_summary = km_df.groupby("Cluster")[summary_cols].median()

    artifacts = {
        "features": features,
        "categorical_cols": categorical_cols,
        "label_encoders": label_encoders,
        "dt_model": dt_model,
        "test_metrics": test_metrics,
        "scaler": scaler,
        "kmeans": kmeans,
        "cluster_mapping": cluster_mapping,
        "cluster_names": cluster_names,
        "cluster_summary": cluster_summary,
        "clustering_features": clustering_features,
    }
    return train_path, artifacts, None


train_path, artifacts, err_msg = train_models_from_files()

if err_msg:
    st.error(err_msg)
    st.stop()

st.success(f"Đã tải thành công file dữ liệu từ: **{train_path}**")

if artifacts["test_metrics"] is not None:
    m = artifacts["test_metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mẫu Test đánh giá", f"{m['samples']:,}")
    c2.metric("MAE (Test)", f"{m['mae']:,.2f} USD")
    c3.metric("RMSE (Test)", f"{m['rmse']:,.2f} USD")
    c4.metric("R² Score (Test)", f"{m['r2']:.4f}")

# Form nhập thông số kỹ thuật xe trên Sidebar
st.sidebar.header("🛠️ Nhập thông số kỹ thuật xe")
les = artifacts["label_encoders"]


def get_options(col_name, fallback_list):
    if col_name in les:
        opts = [str(x) for x in les[col_name].classes_]
        return sorted(opts) if opts else fallback_list
    return fallback_list


manufacturer = st.sidebar.selectbox("Hãng xe (Manufacturer)", get_options("Manufacturer", ["TOYOTA", "HYUNDAI"]))
model_name = st.sidebar.selectbox("Dòng xe (Model)", get_options("Model", ["Camry", "Elantra"]))
category = st.sidebar.selectbox("Kiểu dáng (Category)", get_options("Category", ["Sedan", "Jeep"]))
prod_year = st.sidebar.number_input("Năm sản xuất (Prod. year)", min_value=1990, max_value=2026, value=2015)
mileage = st.sidebar.number_input("Quãng đường đã đi (Mileage - km)", min_value=0, max_value=500000, value=90000, step=1000)
engine_volume = st.sidebar.number_input("Dung tích động cơ (Engine volume - L)", min_value=0.5, max_value=7.0, value=2.0, step=0.1)
levy = st.sidebar.number_input("Thuế Levy (USD)", min_value=0, max_value=50000, value=800, step=50)
cylinders = st.sidebar.number_input("Số xi-lanh (Cylinders)", min_value=2, max_value=16, value=4, step=1)
airbags = st.sidebar.slider("Số túi khí (Airbags)", min_value=0, max_value=16, value=4)
fuel_type = st.sidebar.selectbox("Loại nhiên liệu (Fuel type)", get_options("Fuel type", ["Petrol", "Diesel"]))
gear_box = st.sidebar.selectbox("Loại hộp số (Gear box type)", get_options("Gear box type", ["Automatic", "Manual"]))
drive_wheels = st.sidebar.selectbox("Hệ dẫn động (Drive wheels)", get_options("Drive wheels", ["Front", "4x4"]))
doors = st.sidebar.selectbox("Số cửa (Doors)", get_options("Doors", ["04-May", "02-Mar"]))
wheel = st.sidebar.selectbox("Vô lăng (Wheel)", get_options("Wheel", ["Left wheel"]))
color = st.sidebar.selectbox("Màu sắc (Color)", get_options("Color", ["Black", "White", "Silver"]))

if st.sidebar.button("🔍 Tiến hành dự đoán & phân cụm"):
    input_dict = {
        "Levy": levy,
        "Manufacturer": manufacturer,
        "Model": model_name,
        "Prod. year": prod_year,
        "Category": category,
        "Leather interior": "Yes",
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

    row_df = pd.DataFrame([input_dict])
    for col in artifacts["categorical_cols"]:
        if col in row_df.columns:
            le = les[col]
            val = str(row_df.loc[0, col])
            if val not in set(le.classes_):
                val = le.classes_[0]
            row_df[col] = le.transform([val])

    X_input = row_df[artifacts["features"]].fillna(0)
    predicted_price = float(artifacts["dt_model"].predict(X_input)[0])
    predicted_price = max(500.0, predicted_price)

    st.subheader("📊 Kết quả dự đoán từ mô hình thực tế")
    st.success(f"💰 **Giá xe dự đoán (Decision Tree Regressor):** **{predicted_price:,.2f} USD**")

    age_val = max(0.0, 2020.0 - float(prod_year))
    mileage_for_log = max(1000.0, float(mileage))
    price_for_log = max(500.0, predicted_price)

    km_input = pd.DataFrame(
        [
            {
                "Log_Price": np.log(price_for_log),
                "Age": age_val,
                "Log_Mileage": np.log(mileage_for_log),
                "Engine volume": float(engine_volume),
            }
        ]
    )[artifacts["clustering_features"]]

    km_scaled = artifacts["scaler"].transform(km_input)
    raw_c = int(artifacts["kmeans"].predict(km_scaled)[0])
    mapped_cluster = int(artifacts["cluster_mapping"][raw_c])
    cluster_desc = artifacts["cluster_names"].get(mapped_cluster, f"Cụm {mapped_cluster}")

    st.info(f"🧩 **Phân khúc thị trường (K-Means, K=5):** **{cluster_desc}**")

st.markdown("---")
st.subheader("📌 Bảng giá trị trung vị của 5 cụm K-Means từ dữ liệu huấn luyện")
st.dataframe(artifacts["cluster_summary"].round(2), use_container_width=True)
st.caption("Trường Đại học Giao thông vận tải Thành phố Hồ Chí Minh (UTH) — Nhóm 7[cite: 40]")