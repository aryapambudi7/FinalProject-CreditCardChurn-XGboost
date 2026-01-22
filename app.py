import joblib
import pandas as pd
import streamlit as st

ARTIFACT_PATH = "xgb_artifact.pkl"


@st.cache_resource
def load_artifact():
    return joblib.load(ARTIFACT_PATH)


def preprocess_transform(X_raw: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    ord_cols = artifact["ord_cols"]
    nom_cols = artifact["nom_cols"]
    num_cols = artifact["num_cols"]
    feature_cols = artifact["feature_cols"]

    ord_encoder = artifact["ord_encoder"]
    oh_encoder = artifact["oh_encoder"]
    scaler = artifact["scaler"]

    X = X_raw[artifact["raw_input_cols"]].copy()

    X_ord = pd.DataFrame(
        ord_encoder.transform(X[ord_cols]),
        columns=ord_cols,
        index=X.index
    )

    X_oh = pd.DataFrame(
        oh_encoder.transform(X[nom_cols]),
        columns=oh_encoder.get_feature_names_out(nom_cols),
        index=X.index
    )

    X_to_scale = pd.concat([X[num_cols], X_ord], axis=1)
    X_scaled = pd.DataFrame(
        scaler.transform(X_to_scale),
        columns=X_to_scale.columns,
        index=X.index
    )

    X_final = pd.concat([X_scaled, X_oh], axis=1)
    X_final = X_final.reindex(columns=feature_cols, fill_value=0)

    return X_final


def main():
    st.set_page_config(page_title="Credit Card Churn — XGBoost", layout="wide")
    st.title("Credit Card Churn Prediction — XGBoost")

    artifact = load_artifact()
    model = artifact["model"]
    threshold = float(artifact.get("threshold", 0.5))

    raw_cols = artifact["raw_input_cols"]
    ord_cols = artifact["ord_cols"]
    nom_cols = artifact["nom_cols"]
    num_cols = artifact["num_cols"]

    ord_encoder = artifact["ord_encoder"]
    oh_encoder = artifact["oh_encoder"]

    ord_options = {col: list(cats) for col, cats in zip(ord_cols, ord_encoder.categories_)}
    nom_options = {col: list(cats) for col, cats in zip(nom_cols, oh_encoder.categories_)}

    # HANYA DUA INI yang dibuat kategorikal (dropdown angka)
    discrete_only = {
        "Dependent_count": list(range(0, 6)),            # 0–5
        "Total_Relationship_Count": list(range(1, 7)),   # 1–6
    }

    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction (CSV)"])

    # =========================
    # TAB 1: Single Prediction
    # =========================
    with tab1:
        input_data = {}
        colL, colR = st.columns(2)

        for i, c in enumerate(raw_cols):
            box = colL if i % 2 == 0 else colR

            # ORDINAL dropdown
            if c in ord_cols:
                opts = ord_options.get(c, ["Unknown"])
                default_idx = opts.index("Unknown") if "Unknown" in opts else 0
                input_data[c] = box.selectbox(c, opts, index=default_idx)

            # NOMINAL dropdown
            elif c in nom_cols:
                opts = nom_options.get(c, ["Unknown"])
                default_idx = opts.index("Unknown") if "Unknown" in opts else 0
                input_data[c] = box.selectbox(c, opts, index=default_idx)

            # HANYA 2 kolom ini → dropdown angka
            elif c in discrete_only:
                opts = discrete_only[c]
                input_data[c] = int(box.selectbox(c, opts, index=0))

            # NUMERIC biasa
            elif c in num_cols:
                input_data[c] = float(box.number_input(c, value=0.0))

            # fallback (kalau ada tipe lain)
            else:
                input_data[c] = box.text_input(c, value="")

        if st.button("Predict", type="primary"):
            try:
                X_input = pd.DataFrame([input_data], columns=raw_cols)

                # pastikan numeric valid
                for nc in num_cols:
                    X_input[nc] = pd.to_numeric(X_input[nc], errors="coerce")

                if X_input[num_cols].isna().any().any():
                    st.error("Ada numeric column yang kosong / bukan angka. Mohon isi semua numeric dengan angka.")
                    st.stop()

                X_final = preprocess_transform(X_input, artifact)
                proba = float(model.predict_proba(X_final)[0, 1])
                pred = int(proba >= threshold)

                st.metric("Churn Probability (Class 1)", f"{proba:.3f}")
                st.write("Prediction:", "🟥 Churn (1)" if pred == 1 else "🟦 Not Churn (0)")

            except Exception as e:
                st.error(f"Error: {e}")

    # =========================
    # TAB 2: Batch Prediction
    # =========================
    with tab2:
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            try:
                df = pd.read_csv(up)

                missing = [c for c in raw_cols if c not in df.columns]
                if missing:
                    st.error(f"Kolom kurang: {missing}")
                    st.stop()

                X_raw = df[raw_cols].copy()
                X_final = preprocess_transform(X_raw, artifact)

                probs = model.predict_proba(X_final)[:, 1]
                preds = (probs >= threshold).astype(int)

                out = df.copy()
                out["churn_probability"] = probs
                out["churn_prediction"] = preds

                st.success("✅ Prediksi selesai.")
                st.dataframe(out.head(50), use_container_width=True)

                st.download_button(
                    "Download Result CSV",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
