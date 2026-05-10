"""
app.py
------
Streamlit interactive demo for HBB Mutation Classifier.

Run with:
    streamlit run app.py
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocess import encode_features, filter_hbb, get_feature_matrix, load_clinvar, make_binary_label
from sequence_pipeline import (
    ALL_3MER,
    WINDOW,
    accession_to_chrom,
    apply_mutation,
    delta_kmer,
    fetch_sequence,
    parse_spdi,
)

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HBB Mutation Classifier",
    page_icon="🧬",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    .main-title { font-size: 2.5rem; font-weight: 800; color: #60a5fa; }
    .sub { color: #94a3b8; font-size: 1rem; }
    .metric-card {
        background: #1e293b; border-radius: 12px; padding: 1rem 1.5rem;
        border-left: 4px solid #2563eb;
    }
    .pathogenic { color: #f87171; font-weight: 700; font-size: 1.4rem; }
    .benign { color: #34d399; font-weight: 700; font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧬 HBB Mutation Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Beta-globin gene variant pathogenicity prediction · ClinVar × Machine Learning</div>', unsafe_allow_html=True)
st.markdown("---")

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About This Project")
    st.info("""
    This tool classifies HBB (beta-globin) gene mutations as **Pathogenic** or **Benign**
    using two complementary approaches:

    1. **Annotation Model** — Random Forest on ClinVar variant type + molecular consequence
    2. **Sequence Model** — k-mer frequency encoding of DNA context around the mutation

    Built to support research into beta-thalassemia, a common hemoglobinopathy in India.
    """)
    st.markdown("**Dataset:** ClinVar HBB variants")
    st.markdown("**Genome build:** GRCh38")
    st.markdown("**Sequence source:** Ensembl REST API")
    st.markdown("---")
    st.caption("Source: [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) · [Ensembl](https://rest.ensembl.org)")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Dataset Explorer", "🤖 Annotation Predictor", "🔬 Sequence Predictor"])

DATA_PATH = "data/clinvar_result.txt"

@st.cache_data
def load_data():
    raw = load_clinvar(DATA_PATH)
    hbb = filter_hbb(raw)
    labelled = make_binary_label(hbb)
    return raw, hbb, labelled

try:
    raw, hbb, labelled = load_data()
    data_ok = True
except Exception as e:
    data_ok = False
    st.error(f"Could not load data: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: Dataset Explorer
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    if not data_ok:
        st.warning("Place `clinvar_result.txt` in the `data/` directory and restart.")
    else:
        st.subheader("ClinVar HBB Variant Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total ClinVar Records", f"{len(raw):,}")
        c2.metric("HBB Variants", f"{len(hbb):,}")
        c3.metric("Pathogenic", f"{int((labelled['label']==1).sum()):,}")
        c4.metric("Benign", f"{int((labelled['label']==0).sum()):,}")

        st.markdown("#### Class Distribution")
        class_counts = labelled["label"].value_counts().rename({1: "Pathogenic", 0: "Benign"})
        st.bar_chart(class_counts)

        st.markdown("#### Variant Type Breakdown")
        vt = labelled["Variant type"].value_counts().head(12)
        st.bar_chart(vt)

        st.markdown("#### Molecular Consequence Breakdown")
        mc = labelled["Molecular consequence"].value_counts().head(12)
        st.bar_chart(mc)

        st.markdown("#### Sample Records")
        st.dataframe(
            labelled[["Variant type", "Molecular consequence", "Germline classification", "Canonical SPDI", "label"]]
            .head(20),
            use_container_width=True,
        )

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: Annotation Predictor
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Annotation-Based Prediction")
    st.markdown("Select the **variant type** and **molecular consequence** to get a pathogenicity prediction from the annotation model.")

    if not data_ok:
        st.warning("Data not loaded.")
    else:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        FEATURE_COLS = ["Variant type", "Molecular consequence"]

        @st.cache_resource
        def train_annotation_model():
            import pandas as pd
            from preprocess import encode_features, get_feature_matrix
            import numpy as np

            encoded = encode_features(labelled, FEATURE_COLS)
            X, y = get_feature_matrix(encoded)

            # Oversample minority
            rng = np.random.RandomState(42)
            minority = y[y == 1].index
            majority = y[y == 0].index
            n = len(majority) - len(minority)
            idx = rng.choice(minority, size=n, replace=True)
            X_bal = pd.concat([X, X.loc[idx]]).reset_index(drop=True)
            y_bal = pd.concat([y, y.loc[idx]]).reset_index(drop=True)

            X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal)
            clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
            clf.fit(X_train, y_train)
            return clf, list(X.columns)

        clf_ann, feature_names = train_annotation_model()

        variant_types = sorted(labelled["Variant type"].dropna().unique())
        mol_cons = sorted(labelled["Molecular consequence"].dropna().unique())

        col_a, col_b = st.columns(2)
        with col_a:
            sel_vt = st.selectbox("Variant Type", variant_types, index=variant_types.index("single nucleotide variant") if "single nucleotide variant" in variant_types else 0)
        with col_b:
            sel_mc = st.selectbox("Molecular Consequence", mol_cons, index=mol_cons.index("missense variant") if "missense variant" in mol_cons else 0)

        if st.button("Predict Pathogenicity", key="ann_predict"):
            # Build input vector
            input_dict = {f: 0 for f in feature_names}
            vt_col = f"Variant type_{sel_vt}"
            mc_col = f"Molecular consequence_{sel_mc}"
            if vt_col in input_dict:
                input_dict[vt_col] = 1
            if mc_col in input_dict:
                input_dict[mc_col] = 1

            X_input = pd.DataFrame([input_dict])
            prob = clf_ann.predict_proba(X_input)[0]
            pred = clf_ann.predict(X_input)[0]

            st.markdown("---")
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                label_html = '<div class="pathogenic">⚠️ PATHOGENIC</div>' if pred == 1 else '<div class="benign">✅ BENIGN</div>'
                st.markdown(f"**Prediction:** {label_html}", unsafe_allow_html=True)
            with col_r2:
                st.metric("Pathogenic Probability", f"{prob[1]:.1%}")
                st.metric("Benign Probability", f"{prob[0]:.1%}")

            st.progress(float(prob[1]), text=f"Pathogenic confidence: {prob[1]:.1%}")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: Sequence Predictor
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Sequence-Based Prediction (Live Ensembl Lookup)")
    st.markdown("""
    Enter a **Canonical SPDI** string from ClinVar. The app will:
    1. Fetch the reference genomic sequence from Ensembl
    2. Apply your mutation to generate the alternate sequence
    3. Compute k-mer frequency differences
    4. Return a pathogenicity prediction

    **Example SPDI:** `NC_000011.10:5226966:A:T`
    """)

    spdi_input = st.text_input("Canonical SPDI", value="NC_000011.10:5226966:A:T")
    label_input = st.radio("True label (for display only)", ["Unknown", "Pathogenic", "Benign"], horizontal=True)

    if not data_ok:
        st.warning("Data not loaded.")
    else:
        @st.cache_resource
        def train_seq_model_from_cache():
            """Train on cached k-mer features if available, otherwise return None."""
            X_path = "results/sequence_X.npy"
            y_path = "results/sequence_y.npy"
            if os.path.exists(X_path) and os.path.exists(y_path):
                X = np.load(X_path)
                y = np.load(y_path)
                clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
                clf.fit(X, y)
                return clf
            return None

        seq_clf = train_seq_model_from_cache()

        if st.button("Run Sequence Prediction", key="seq_predict"):
            parsed = parse_spdi(spdi_input)
            if parsed is None:
                st.error("Could not parse SPDI. Expected format: `NC_000011.10:5226966:A:T`")
            else:
                accession, pos, ref, alt = parsed
                chrom = accession_to_chrom(accession)
                start = pos - WINDOW + 1
                end = pos + WINDOW + 1

                with st.spinner(f"Fetching sequence chr{chrom}:{start}-{end} from Ensembl…"):
                    ref_seq = fetch_sequence(chrom, start, end)

                if ref_seq is None:
                    st.error("Could not fetch sequence. Check your internet connection or try again.")
                else:
                    mut_seq = apply_mutation(ref_seq, ref, alt, WINDOW)
                    if mut_seq is None:
                        st.error(f"REF mismatch: expected `{ref}` at position {WINDOW} but found `{ref_seq[WINDOW]}`")
                    else:
                        st.success("Sequence fetched and mutation applied!")
                        col_s1, col_s2 = st.columns(2)
                        with col_s1:
                            st.markdown("**Reference sequence (41 bp):**")
                            st.code(ref_seq, language="")
                        with col_s2:
                            st.markdown("**Mutated sequence (41 bp):**")
                            # Highlight changed position
                            highlighted = ref_seq[:WINDOW] + f"[{alt}]" + ref_seq[WINDOW+1:]
                            st.code(highlighted, language="")

                        feat = delta_kmer(ref_seq, mut_seq)
                        st.markdown("**Top shifted k-mers (mutated − reference):**")
                        kmer_df = pd.DataFrame({"k-mer": ALL_3MER, "Δ frequency": feat})
                        top_kmers = kmer_df.reindex(kmer_df["Δ frequency"].abs().nlargest(10).index)
                        st.dataframe(top_kmers.reset_index(drop=True), use_container_width=True)

                        if seq_clf is not None:
                            prob = seq_clf.predict_proba(feat.reshape(1, -1))[0]
                            pred = seq_clf.predict(feat.reshape(1, -1))[0]
                            st.markdown("---")
                            label_html = '<div class="pathogenic">⚠️ PATHOGENIC</div>' if pred == 1 else '<div class="benign">✅ BENIGN</div>'
                            st.markdown(f"**Sequence Model Prediction:** {label_html}", unsafe_allow_html=True)
                            st.progress(float(prob[1]), text=f"Pathogenic confidence: {prob[1]:.1%}")
                        else:
                            st.info("Run `python src/sequence_pipeline.py` first to train and cache the sequence model. The k-mer features above are still computed live.")
