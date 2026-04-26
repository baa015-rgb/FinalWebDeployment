import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------

st.set_page_config(
    page_title="Buckell Lending",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ------------------------------------------------------------
# Visual styling
# ------------------------------------------------------------

st.markdown(
    """
    <style>
        :root {
            --navy: #08233f;
            --blue: #123b63;
            --gold: #f2a900;
            --card: #ffffff;
            --border: #d8dee8;
            --text: #0f2742;
            --muted: #64748b;
        }

        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
        }

        .main .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
            max-width: 1480px;
        }

        .hero {
            background: linear-gradient(90deg, var(--navy), var(--blue));
            color: white;
            padding: 24px 30px;
            border-radius: 18px;
            margin-bottom: 18px;
            box-shadow: 0 12px 28px rgba(8, 35, 63, 0.18);
        }

        .hero-title {
            font-size: 32px;
            font-weight: 900;
            letter-spacing: 0.2px;
            margin: 0;
        }

        .hero-subtitle {
            font-size: 14px;
            color: #e5e7eb;
            margin-top: 4px;
            font-weight: 600;
        }

        .section-label {
            color: var(--text);
            font-weight: 900;
            font-size: 16px;
            margin-bottom: 6px;
        }

        .field-group-note {
            color: var(--muted);
            font-size: 12px;
            margin-top: -4px;
            margin-bottom: 10px;
        }

        .decision-card {
            color: white;
            border-radius: 20px;
            padding: 24px 26px;
            margin-bottom: 16px;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.16);
        }

        .approve { background: linear-gradient(90deg, #166534, #15803d); }
        .review { background: linear-gradient(90deg, #92400e, #d97706); }
        .reject { background: linear-gradient(90deg, #7f1d1d, #b91c1c); }
        .ready { background: linear-gradient(90deg, var(--navy), var(--blue)); }

        .decision-label {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            opacity: 0.88;
            font-weight: 850;
        }

        .decision-value {
            font-size: 44px;
            font-weight: 950;
            line-height: 1.05;
            margin-top: 4px;
        }

        .metric-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 20px 22px;
            margin-bottom: 14px;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
        }

        .metric-label {
            color: var(--muted);
            font-size: 12px;
            font-weight: 850;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 6px;
        }

        .metric-value {
            color: var(--navy);
            font-size: 34px;
            font-weight: 950;
            line-height: 1.1;
        }

        .metric-explain {
            color: #475569;
            font-size: 12px;
            line-height: 1.35;
            margin-top: 9px;
            font-weight: 550;
        }

        .model-detail-card {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 15px 17px;
            margin-top: 12px;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.04);
        }

        .model-detail-title {
            color: var(--navy);
            font-size: 13px;
            font-weight: 900;
            margin-bottom: 8px;
        }

        .model-detail-row {
            display: flex;
            justify-content: space-between;
            gap: 14px;
            padding: 7px 0;
            border-top: 1px solid #edf2f7;
            color: var(--text);
            font-size: 13px;
        }

        .model-detail-row:first-of-type {
            border-top: none;
        }

        .model-detail-row span {
            color: var(--muted);
            font-weight: 700;
        }

        .model-detail-row strong {
            color: var(--navy);
            font-weight: 850;
            text-align: right;
        }

        .footer-note {
            color: var(--muted);
            font-size: 12px;
            text-align: center;
            margin-top: 22px;
            padding-top: 12px;
            border-top: 1px solid #d8dee8;
        }

        /* Higher contrast Streamlit widgets */
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] label,
        [data-testid="stWidgetLabel"] p,
        div[data-testid="stSlider"] label,
        div[data-testid="stSlider"] p,
        div[data-testid="stSelectbox"] label,
        div[data-testid="stSelectbox"] p,
        div[data-testid="stSelectSlider"] label,
        div[data-testid="stSelectSlider"] p,
        div[data-testid="stRadio"] label,
        div[data-testid="stRadio"] p {
            color: var(--text) !important;
            font-weight: 850 !important;
        }

        div[data-testid="stSlider"] span,
        div[data-testid="stSlider"] div,
        div[data-testid="stSelectSlider"] span,
        div[data-testid="stSelectSlider"] div {
            color: var(--text) !important;
        }

        div[data-testid="stSlider"] [data-testid="stThumbValue"] {
            background: var(--navy) !important;
            color: #ffffff !important;
            border-radius: 8px !important;
        }

        div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 12px !important;
            color: var(--text) !important;
        }

        div[data-baseweb="select"] span,
        div[data-baseweb="select"] svg {
            color: var(--text) !important;
            fill: var(--text) !important;
        }

        div[data-baseweb="popover"],
        div[data-baseweb="popover"] * {
            color: var(--text) !important;
        }

        ul[role="listbox"],
        div[role="listbox"] {
            background: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 12px !important;
        }

        li[role="option"],
        div[role="option"] {
            color: var(--text) !important;
            background: #ffffff !important;
            font-weight: 700 !important;
        }

        li[role="option"]:hover,
        div[role="option"]:hover,
        li[aria-selected="true"],
        div[aria-selected="true"] {
            background: #e8eff7 !important;
            color: var(--navy) !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 18px !important;
            border-color: var(--border) !important;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
            background-color: white;
        }

        div[data-testid="stForm"] {
            border: none;
            padding: 0;
        }

        div[data-testid="stFormSubmitButton"] > button {
            background-color: var(--gold);
            color: #111827;
            font-weight: 900;
            border: none;
            border-radius: 12px;
            padding: 11px 20px;
            width: 100%;
        }

        div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: #ffc233;
            color: #111827;
            border: none;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Load model artifacts
# ------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    artifact_path = Path("buckell_lending_model_artifacts.pkl")

    if not artifact_path.exists():
        st.error("Model artifact file not found. Upload buckell_lending_model_artifacts.pkl to the app repository.")
        st.stop()

    with open(artifact_path, "rb") as file:
        return pickle.load(file)


artifacts = load_artifacts()


# ------------------------------------------------------------
# Feature engineering and preprocessing functions
# ------------------------------------------------------------

def clean_emp_length(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value == "10+ years":
        return 10
    if value == "< 1 year":
        return 0
    if value == "1 year":
        return 1

    return int(value.replace(" years", "").replace(" year", ""))


def safe_ratio(numerator, denominator):
    return np.where(
        (denominator == 0) | (pd.isna(denominator)),
        np.nan,
        numerator / denominator
    )


def add_log_features(df_input, cols):
    df_output = df_input.copy()

    for col in cols:
        if col in df_output.columns:
            df_output["log1p_" + col] = np.log1p(df_output[col].clip(lower=0))

    return df_output


def build_application_features(input_data, feature_set):
    temp = pd.DataFrame([input_data]).copy()

    temp["issue_d"] = pd.to_datetime(temp["issue_d"], errors="coerce")
    temp["earliest_cr_line"] = pd.to_datetime(temp["earliest_cr_line"], errors="coerce")

    temp["credit_history_years"] = (
        temp["issue_d"] - temp["earliest_cr_line"]
    ).dt.days / 365.25

    temp["fico_avg"] = (temp["fico_range_high"] + temp["fico_range_low"]) / 2
    temp["fico_spread"] = temp["fico_range_high"] - temp["fico_range_low"]

    temp["emp_length_num"] = temp["emp_length"].apply(clean_emp_length)

    grade_map = {
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "E": 5,
        "F": 6,
        "G": 7
    }

    temp["grade_num"] = temp["grade"].map(grade_map)

    if feature_set in ["engineered", "transformed"]:
        temp["monthly_income"] = temp["annual_inc"] / 12
        temp["loan_to_income"] = safe_ratio(temp["loan_amnt"], temp["annual_inc"])
        temp["funded_to_income"] = safe_ratio(temp["funded_amnt"], temp["annual_inc"])
        temp["installment_to_monthly_income"] = safe_ratio(temp["installment"], temp["monthly_income"])
        temp["revol_bal_to_income"] = safe_ratio(temp["revol_bal"], temp["annual_inc"])

        temp["int_rate_x_dti"] = temp["int_rate"] * temp["dti"]
        temp["int_rate_x_loan_to_income"] = temp["int_rate"] * temp["loan_to_income"]
        temp["dti_x_revol_util"] = temp["dti"] * temp["revol_util"]
        temp["fico_x_int_rate"] = temp["fico_avg"] * temp["int_rate"]
        temp["fico_x_dti"] = temp["fico_avg"] * temp["dti"]
        temp["loan_amnt_x_int_rate"] = temp["loan_amnt"] * temp["int_rate"]
        temp["installment_x_term"] = temp["installment"] * temp["term_num"]

        temp["fico_band"] = pd.cut(
            temp["fico_avg"],
            bins=[-np.inf, 660, 680, 700, 720, 740, 760, np.inf],
            labels=[
                "fico_under_660", "fico_660_679", "fico_680_699",
                "fico_700_719", "fico_720_739", "fico_740_759", "fico_760_plus"
            ]
        )

        temp["dti_band"] = pd.cut(
            temp["dti"],
            bins=[-np.inf, 10, 20, 30, 40, np.inf],
            labels=["dti_under_10", "dti_10_19", "dti_20_29", "dti_30_39", "dti_40_plus"]
        )

        temp["interest_rate_band"] = pd.cut(
            temp["int_rate"],
            bins=[-np.inf, 8, 12, 16, 20, 25, np.inf],
            labels=["rate_under_8", "rate_8_12", "rate_12_16", "rate_16_20", "rate_20_25", "rate_25_plus"]
        )

        temp["loan_to_income_band"] = pd.cut(
            temp["loan_to_income"],
            bins=[-np.inf, 0.10, 0.20, 0.30, 0.50, np.inf],
            labels=["lti_under_10", "lti_10_20", "lti_20_30", "lti_30_50", "lti_50_plus"]
        )

    if feature_set == "transformed":
        log_cols = [
            "loan_amnt", "funded_amnt", "installment", "annual_inc",
            "revol_bal", "dti", "loan_to_income",
            "installment_to_monthly_income", "revol_bal_to_income"
        ]

        temp = add_log_features(temp, log_cols)

        temp["grade_num_x_int_rate"] = temp["grade_num"] * temp["int_rate"]
        temp["grade_num_x_dti"] = temp["grade_num"] * temp["dti"]
        temp["fico_avg_x_loan_to_income"] = temp["fico_avg"] * temp["loan_to_income"]
        temp["credit_history_x_fico"] = temp["credit_history_years"] * temp["fico_avg"]

    drop_cols = [
        "id", "loan_status", "fully_paid_flag", "ret_PESS",
        "total_pymnt", "last_pymnt_d", "recoveries", "loan_length",
        "term", "emp_length", "issue_d", "earliest_cr_line",
        "fico_range_high", "fico_range_low"
    ]

    return temp.drop(columns=drop_cols, errors="ignore")


def apply_preprocessing(X, preprocess_artifacts):
    X_clean = X.copy()
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)

    numeric_cols = preprocess_artifacts["numeric_cols"]
    categorical_cols = preprocess_artifacts["categorical_cols"]
    dummy_columns = preprocess_artifacts["dummy_columns"]

    for col in numeric_cols:
        if col not in X_clean.columns:
            X_clean[col] = np.nan

    for col in categorical_cols:
        if col not in X_clean.columns:
            X_clean[col] = np.nan

    X_clean[numeric_cols] = X_clean[numeric_cols].fillna(preprocess_artifacts["numeric_medians"])

    if preprocess_artifacts.get("cap_outliers", False):
        lower_caps = preprocess_artifacts["numeric_lower_caps"]
        upper_caps = preprocess_artifacts["numeric_upper_caps"]

        X_clean[numeric_cols] = X_clean[numeric_cols].clip(
            lower=lower_caps,
            upper=upper_caps,
            axis=1
        )

    for col in categorical_cols:
        X_clean[col] = X_clean[col].astype("object")
        fill_value = preprocess_artifacts["category_fill_values"].get(col, "Missing")
        X_clean[col] = X_clean[col].fillna(fill_value)

    # Build the model matrix directly from the saved training dummy columns.
    # This avoids one-row get_dummies/drop_first issues in deployment.
    X_dummies = pd.DataFrame(
        0.0,
        index=X_clean.index,
        columns=dummy_columns
    )

    for col in numeric_cols:
        if col in X_dummies.columns:
            X_dummies[col] = X_clean[col].astype(float)

    for col in categorical_cols:
        for row_index, value in X_clean[col].items():
            dummy_col = f"{col}_{value}"
            if dummy_col in X_dummies.columns:
                X_dummies.at[row_index, dummy_col] = 1.0

    X_dummies = X_dummies.astype(float)

    X_scaled_array = preprocess_artifacts["scaler"].transform(X_dummies)

    X_scaled = pd.DataFrame(
        X_scaled_array,
        columns=X_dummies.columns,
        index=X_dummies.index
    )

    return X_dummies, X_scaled


def calculate_installment(loan_amount, annual_rate, term_months):
    monthly_rate = (annual_rate / 100) / 12

    if monthly_rate == 0:
        return loan_amount / term_months

    return loan_amount * monthly_rate / (1 - (1 + monthly_rate) ** (-term_months))


def make_recommendation(full_paid_probability, predicted_return):
    approve_threshold = artifacts["approve_probability_threshold"]
    reject_threshold = artifacts["reject_probability_threshold"]
    return_threshold = artifacts["return_threshold"]
    severe_return_threshold = -5.0

    if full_paid_probability >= approve_threshold and predicted_return >= return_threshold:
        return "APPROVE"

    if full_paid_probability < reject_threshold or predicted_return <= severe_return_threshold:
        return "REJECT"

    return "REVIEW"


def predict_application(input_data):
    class_features = build_application_features(
        input_data,
        artifacts["final_class_feature_set"]
    )

    class_dummies, class_scaled = apply_preprocessing(
        class_features,
        artifacts["final_class_preprocess_artifacts"]
    )

    if artifacts["final_class_uses_scaled"]:
        class_model_input = class_scaled
    else:
        class_model_input = class_dummies

    full_paid_probability = artifacts["final_class_model"].predict_proba(class_model_input)[0][1]

    reg_features = build_application_features(
        input_data,
        artifacts["final_reg_feature_set"]
    )

    reg_dummies, reg_scaled = apply_preprocessing(
        reg_features,
        artifacts["final_reg_preprocess_artifacts"]
    )

    if artifacts["final_reg_uses_scaled"]:
        reg_model_input = reg_scaled
    else:
        reg_model_input = reg_dummies

    predicted_return = artifacts["final_reg_model"].predict(reg_model_input)[0]
    recommendation = make_recommendation(full_paid_probability, predicted_return)

    return full_paid_probability, predicted_return, recommendation


def scenario_defaults(scenario):
    scenarios = {
        "Approve example": {
            "loan_amount": 12000,
            "term": "36 months",
            "int_rate": 8.5,
            "grade": "A",
            "purpose": "credit_card",
            "annual_inc": 95000,
            "emp_length": "10+ years",
            "home_ownership": "MORTGAGE",
            "verification_status": "Verified",
            "dti": 12.0,
            "fico_score": 740,
            "credit_history_years": 14,
            "open_acc": 11,
            "delinq_2yrs": 0,
            "pub_rec": 0,
            "revol_bal": 8500,
            "revol_util": 32.0
        },
        "Review example": {
            "loan_amount": 18000,
            "term": "36 months",
            "int_rate": 14.5,
            "grade": "C",
            "purpose": "debt_consolidation",
            "annual_inc": 68000,
            "emp_length": "5 years",
            "home_ownership": "RENT",
            "verification_status": "Source Verified",
            "dti": 24.0,
            "fico_score": 695,
            "credit_history_years": 8,
            "open_acc": 9,
            "delinq_2yrs": 0,
            "pub_rec": 0,
            "revol_bal": 14500,
            "revol_util": 58.0
        },
        "Reject example": {
            "loan_amount": 28000,
            "term": "60 months",
            "int_rate": 24.0,
            "grade": "F",
            "purpose": "small_business",
            "annual_inc": 52000,
            "emp_length": "< 1 year",
            "home_ownership": "RENT",
            "verification_status": "Not Verified",
            "dti": 36.0,
            "fico_score": 665,
            "credit_history_years": 4,
            "open_acc": 6,
            "delinq_2yrs": 2,
            "pub_rec": 1,
            "revol_bal": 23500,
            "revol_util": 89.0
        }
    }

    return scenarios.get(scenario, scenarios["Review example"])


def format_choice(value):
    return str(value).replace("_", " ").title()


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Buckell Lending</div>
        <div class="hero-subtitle">Loan Screening Dashboard</div>
    </div>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

with st.sidebar:
    st.markdown("### Model")
    st.write(f"Approve threshold: **{artifacts['approve_probability_threshold']:.0%}**")
    st.write(f"Reject threshold: **{artifacts['reject_probability_threshold']:.0%}**")
    st.write("Severe return trigger: **-5.00%**")

    st.markdown("---")
    st.caption("Decision-support prototype. Not an automatic underwriting system.")


# ------------------------------------------------------------
# Scenario state
# ------------------------------------------------------------

scenario = st.radio(
    "Scenario",
    ["Approve example", "Review example", "Reject example"],
    index=1,
    horizontal=True
)

if "last_scenario" not in st.session_state:
    st.session_state["last_scenario"] = scenario

if scenario != st.session_state["last_scenario"]:
    st.session_state["evaluation_result"] = None
    st.session_state["last_scenario"] = scenario

if "evaluation_result" not in st.session_state:
    st.session_state["evaluation_result"] = None

defaults = scenario_defaults(scenario)

key_prefix = scenario.replace(" ", "_").replace("-", "_").lower()

left_col, right_col = st.columns([1.15, 0.85], gap="large")


# ------------------------------------------------------------
# Input form
# ------------------------------------------------------------

with left_col:
    with st.form("loan_input_form"):
        with st.container(border=True):
            st.markdown('<div class="section-label">Loan Terms</div>', unsafe_allow_html=True)
            st.markdown('<div class="field-group-note">Set the requested loan structure and pricing.</div>', unsafe_allow_html=True)

            loan_col_1, loan_col_2 = st.columns(2)

            with loan_col_1:
                loan_amount = st.slider(
                    "Loan amount",
                    min_value=1000,
                    max_value=40000,
                    value=int(defaults["loan_amount"]),
                    step=500,
                    format="$%d",
                    key=f"{key_prefix}_loan_amount"
                )

                term = st.selectbox(
                    "Term",
                    ["36 months", "60 months"],
                    index=["36 months", "60 months"].index(defaults["term"]),
                    key=f"{key_prefix}_term"
                )

                grade = st.selectbox(
                    "Grade",
                    ["A", "B", "C", "D", "E", "F", "G"],
                    index=["A", "B", "C", "D", "E", "F", "G"].index(defaults["grade"]),
                    key=f"{key_prefix}_grade"
                )

            with loan_col_2:
                int_rate = st.slider(
                    "Interest rate",
                    min_value=5.0,
                    max_value=30.0,
                    value=float(defaults["int_rate"]),
                    step=0.1,
                    format="%.1f%%",
                    key=f"{key_prefix}_int_rate"
                )

                purpose_options = [
                    "debt_consolidation", "credit_card", "home_improvement", "major_purchase",
                    "small_business", "car", "medical", "moving", "vacation", "house",
                    "renewable_energy", "educational", "wedding", "other"
                ]

                purpose = st.selectbox(
                    "Purpose",
                    purpose_options,
                    index=purpose_options.index(defaults["purpose"]) if defaults["purpose"] in purpose_options else purpose_options.index("other"),
                    format_func=format_choice,
                    key=f"{key_prefix}_purpose"
                )

        with st.container(border=True):
            st.markdown('<div class="section-label">Borrower Profile</div>', unsafe_allow_html=True)
            st.markdown('<div class="field-group-note">Describe income, stability, and current debt burden.</div>', unsafe_allow_html=True)

            borrower_col_1, borrower_col_2 = st.columns(2)

            with borrower_col_1:
                annual_inc = st.slider(
                    "Annual income",
                    min_value=15000,
                    max_value=250000,
                    value=int(defaults["annual_inc"]),
                    step=5000,
                    format="$%d",
                    key=f"{key_prefix}_annual_inc"
                )

                emp_options = [
                    "< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
                    "6 years", "7 years", "8 years", "9 years", "10+ years"
                ]

                emp_length = st.selectbox(
                    "Employment length",
                    emp_options,
                    index=emp_options.index(defaults["emp_length"]),
                    key=f"{key_prefix}_emp_length"
                )

                home_options = ["RENT", "MORTGAGE", "OWN", "OTHER", "ANY", "NONE"]

                home_ownership = st.selectbox(
                    "Home ownership",
                    home_options,
                    index=home_options.index(defaults["home_ownership"]),
                    format_func=lambda x: x.title(),
                    key=f"{key_prefix}_home_ownership"
                )

            with borrower_col_2:
                verification_options = ["Not Verified", "Source Verified", "Verified"]

                verification_status = st.selectbox(
                    "Verification status",
                    verification_options,
                    index=verification_options.index(defaults["verification_status"]),
                    key=f"{key_prefix}_verification_status"
                )

                dti = st.slider(
                    "Debt-to-income ratio",
                    min_value=0.0,
                    max_value=45.0,
                    value=float(defaults["dti"]),
                    step=0.5,
                    format="%.1f",
                    key=f"{key_prefix}_dti"
                )

        with st.container(border=True):
            st.markdown('<div class="section-label">Credit Profile</div>', unsafe_allow_html=True)
            st.markdown('<div class="field-group-note">Summarize credit quality, credit history, and revolving usage.</div>', unsafe_allow_html=True)

            credit_col_1, credit_col_2 = st.columns(2)

            with credit_col_1:
                fico_score = st.slider(
                    "FICO score",
                    min_value=640,
                    max_value=850,
                    value=int(defaults["fico_score"]),
                    step=5,
                    key=f"{key_prefix}_fico_score"
                )

                credit_history_years = st.slider(
                    "Credit history years",
                    min_value=1,
                    max_value=40,
                    value=int(defaults["credit_history_years"]),
                    step=1,
                    key=f"{key_prefix}_credit_history_years"
                )

                open_acc = st.slider(
                    "Open accounts",
                    min_value=1,
                    max_value=40,
                    value=int(defaults["open_acc"]),
                    step=1,
                    key=f"{key_prefix}_open_acc"
                )

                delinq_2yrs = st.slider(
                    "Delinquencies, last 2 years",
                    min_value=0,
                    max_value=10,
                    value=int(defaults["delinq_2yrs"]),
                    step=1,
                    key=f"{key_prefix}_delinq_2yrs"
                )

            with credit_col_2:
                pub_rec = st.slider(
                    "Public records",
                    min_value=0,
                    max_value=8,
                    value=int(defaults["pub_rec"]),
                    step=1,
                    key=f"{key_prefix}_pub_rec"
                )

                revol_bal = st.slider(
                    "Revolving balance",
                    min_value=0,
                    max_value=80000,
                    value=int(defaults["revol_bal"]),
                    step=1000,
                    format="$%d",
                    key=f"{key_prefix}_revol_bal"
                )

                revol_util = st.slider(
                    "Revolving utilization",
                    min_value=0.0,
                    max_value=120.0,
                    value=float(defaults["revol_util"]),
                    step=1.0,
                    format="%.0f%%",
                    key=f"{key_prefix}_revol_util"
                )

        evaluate = st.form_submit_button("Evaluate Loan")


# ------------------------------------------------------------
# Prepare application record
# ------------------------------------------------------------

term_num = 36 if term == "36 months" else 60
installment = calculate_installment(loan_amount, int_rate, term_num)
issue_date = pd.Timestamp("2018-01-01")
earliest_credit_date = issue_date - pd.DateOffset(years=int(credit_history_years))

application_record = {
    "id": 0,
    "loan_amnt": loan_amount,
    "funded_amnt": loan_amount,
    "term": term,
    "term_num": term_num,
    "int_rate": int_rate,
    "installment": installment,
    "grade": grade,
    "emp_length": emp_length,
    "home_ownership": home_ownership,
    "annual_inc": annual_inc,
    "verification_status": verification_status,
    "purpose": purpose,
    "dti": dti,
    "delinq_2yrs": delinq_2yrs,
    "earliest_cr_line": earliest_credit_date.strftime("%Y-%m-%d"),
    "fico_range_low": fico_score - 2,
    "fico_range_high": fico_score + 2,
    "open_acc": open_acc,
    "pub_rec": pub_rec,
    "revol_bal": revol_bal,
    "revol_util": revol_util,
    "issue_d": issue_date.strftime("%Y-%m-%d")
}

if evaluate:
    try:
        full_paid_probability, predicted_return, recommendation = predict_application(application_record)

        st.session_state["evaluation_result"] = {
            "full_paid_probability": full_paid_probability,
            "predicted_return": predicted_return,
            "recommendation": recommendation,
            "installment": installment,
            "application_record": application_record
        }
    except Exception as error:
        st.session_state["evaluation_result"] = None
        st.error("Prediction failed. Check that the model artifact is in the same repository as app.py.")
        st.exception(error)


# ------------------------------------------------------------
# Output panel
# ------------------------------------------------------------

with right_col:
    result = st.session_state.get("evaluation_result")

    if result is None:
        st.markdown(
            """
            <div class="decision-card ready">
                <div class="decision-label">Status</div>
                <div class="decision-value">READY</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="model-detail-card">
                <div class="model-detail-title">Decision logic</div>
                <div class="model-detail-row"><span>Approve</span><strong>Probability ≥ {artifacts['approve_probability_threshold']:.0%} and return ≥ {artifacts['return_threshold']:.0f}%</strong></div>
                <div class="model-detail-row"><span>Review</span><strong>Between thresholds or mild negative return</strong></div>
                <div class="model-detail-row"><span>Reject</span><strong>Probability &lt; {artifacts['reject_probability_threshold']:.0%} or return ≤ -5%</strong></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        recommendation = result["recommendation"]
        full_paid_probability = result["full_paid_probability"]
        predicted_return = result["predicted_return"]
        installment = result["installment"]

        if recommendation == "APPROVE":
            decision_class = "approve"
        elif recommendation == "REVIEW":
            decision_class = "review"
        else:
            decision_class = "reject"

        st.markdown(
            f"""
            <div class="decision-card {decision_class}">
                <div class="decision-label">Recommendation</div>
                <div class="decision-value">{recommendation}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Full Repayment Probability</div>
                <div class="metric-value">{full_paid_probability:.1%}</div>
                <div class="metric-explain">Estimated chance that the borrower fully repays the loan.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Pessimistic Return</div>
                <div class="metric-value">{predicted_return:.2f}%</div>
                <div class="metric-explain">Estimated downside return rate. Negative values indicate predicted loss.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="model-detail-card">
                <div class="model-detail-title">Decision logic</div>
                <div class="model-detail-row"><span>Calculated installment</span><strong>${installment:,.2f}</strong></div>
                <div class="model-detail-row"><span>Approve</span><strong>Probability ≥ {artifacts['approve_probability_threshold']:.0%} and return ≥ {artifacts['return_threshold']:.0f}%</strong></div>
                <div class="model-detail-row"><span>Review</span><strong>Probability {artifacts['reject_probability_threshold']:.0%}–{artifacts['approve_probability_threshold']:.0%} or return -5% to 0%</strong></div>
                <div class="model-detail-row"><span>Reject</span><strong>Probability &lt; {artifacts['reject_probability_threshold']:.0%} or return ≤ -5%</strong></div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown(
    """
    <div class="footer-note">
        ANOP 330 Final Project | Brian Auerbach and Mitch Jackson
    </div>
    """,
    unsafe_allow_html=True
)
