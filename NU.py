import streamlit as st
import pandas as pd
import numpy as np
import rasterio
import os
import joblib
import whitebox
import tempfile

# --- Page Configuration ---
st.set_page_config(
    page_title="Flood Prone Area Detector",
    page_icon="🌊",
    layout="centered"
)

# --- Initialize WhiteboxTools ---
@st.cache_resource
def init_whitebox():
    wbt = whitebox.WhiteboxTools()
    wbt.verbose = False
    return wbt

wbt = init_whitebox()

# --- Load Model ---
MODEL_PATH = r"C:\Users\AYA\Downloads\nu\\random_forest_model.pkl"

@st.cache_resource
def load_model():
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            #if hasattr(model, 'feature_names_in_'):
                #st.info(f"Model expects these features: {list(model.feature_names_in_)}")
            return model
        else:
            st.error(f"Model file not found at: {MODEL_PATH}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Feature Extraction Function ---
def extract_features(dem_path, model_expected_features=None):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            filled_path = os.path.join(temp_dir, "filled.tif")
            slope_path = os.path.join(temp_dir, "slope.tif")
            flowacc_path = os.path.join(temp_dir, "flow_acc.tif")
            temp_dem_path = os.path.join(temp_dir, "input_dem.tif")

            with rasterio.open(dem_path) as src:
                dem = src.read(1).astype(float)
                dem = np.where(dem == src.nodata, np.nan, dem)
                dem_filled = np.where(np.isnan(dem), np.nanmean(dem), dem)
                profile = src.profile
                profile.update(dtype=rasterio.float32, count=1)
                with rasterio.open(temp_dem_path, 'w', **profile) as dst:
                    dst.write(dem_filled.astype(np.float32), 1)

            wbt.fill_depressions(temp_dem_path, filled_path)
            wbt.slope(filled_path, slope_path, zfactor=1.0)
            wbt.d8_flow_accumulation(filled_path, flowacc_path, out_type="cells")

            with rasterio.open(slope_path) as s:
                slope = s.read(1)
            with rasterio.open(flowacc_path) as f:
                flow_accum = f.read(1)

            original_features = {
                'mean_elevation': np.nanmean(dem),
                'std_elevation': np.nanstd(dem),
                'min_elevation': np.nanmin(dem),
                'max_elevation': np.nanmax(dem),
                'range_elevation': np.nanmax(dem) - np.nanmin(dem),
                'elevation_25_percentile': np.nanpercentile(dem, 25),
                'elevation_75_percentile': np.nanpercentile(dem, 75),
                'mean_slope': np.nanmean(slope),
                'max_slope': np.nanmax(slope),
                'mean_flow_accum': np.nanmean(flow_accum),
                'max_flow_accum': np.nanmax(flow_accum),
                'percent_low_slope': np.sum(slope < 2) / slope.size if slope.size > 0 else 0
            }

            # Filter and order features based on model expectations
            if model_expected_features is not None:
                ordered_features = {
                    key: original_features[key]
                    for key in model_expected_features
                    if key in original_features
                }
                missing_features = [key for key in model_expected_features if key not in ordered_features]
                if missing_features:
                    st.error(f"Missing required features: {missing_features}")
                    return None
                return ordered_features

            return original_features

    except Exception as e:
        st.error(f"Error in feature extraction: {e}")
        return None

# --- Preprocessing ---
def preprocess_features(df):
    df = df.round(3)
    numeric_cols = df.select_dtypes(include=['number']).columns
    min_val = df[numeric_cols].min().min()
    if min_val <= -1:
        df[numeric_cols] = df[numeric_cols] + abs(min_val) + 1
    df[numeric_cols] = np.log1p(df[numeric_cols])
    return df

# --- Main App ---
def main():
    st.title("🌊 Flood Prone Area Detector")
    st.write("Upload a DEM (.tif) file to check if the area is flood prone.")

    model = load_model()
    if model is None:
        st.stop()

    expected_features = None
    if hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
        #st.success(f"Model loaded successfully! Expected features: {len(expected_features)}")
    else:
        st.error("Cannot determine model's expected features.")
        st.stop()

    uploaded_file = st.file_uploader("Choose a DEM file", type=['tif', 'tiff'])
    if uploaded_file is not None:
        with st.spinner("Processing DEM file, please wait..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_file_path = tmp_file.name

                features = extract_features(tmp_file_path, expected_features)
                os.unlink(tmp_file_path)

                if features is None:
                    st.error("Feature extraction failed or mismatch.")
                    st.stop()

                df = pd.DataFrame([features], columns=expected_features)

                #st.subheader("📊 Extracted Features")
                #st.dataframe(df)

                df_processed = preprocess_features(df.copy())

                #st.subheader("🔧 Preprocessed Features")
                #st.dataframe(df_processed)

                prediction = model.predict(df_processed)[0]
                prediction_proba = model.predict_proba(df_processed)[0]

                st.subheader("Prediction Result")
                if prediction == 1:
                    st.error("🌊 FLOOD PRONE AREA")
                else:
                    st.success("🏞️ NOT FLOOD PRONE")
                st.metric("Confidence", f"{max(prediction_proba):.1%}")

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

if __name__ == "__main__":
    main()
