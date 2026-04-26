
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
# Styling
# ------------------------------------------------------------

st.markdown(
    """
    <style>
        :root {
            --navy: #08233f;
            --blue: #143b63;
            --gold: #f2a900;
            --muted: #64748b;
            --border: #d7dce3;
        }

        .stApp {
            background: linear-gradient(180deg, #f7f8fb 0%, #eef1f5 100%);
            color: var(--navy);
        }

        .main .block-container {
            padding-top: 24px;
            max-width: 1220px;
        }

        .top-header {
            background: linear-gradient(90deg, #08233f, #143b63);
            color: white;
            padding: 26px 34px;
            border-radius: 18px;
            margin-bottom: 22px;
            box-shadow: 0 10px 25px rgba(8, 35, 63, 0.18);
        }

        .top-title {
            font-size: 34px;
            font-weight: 850;
            margin: 0;
        }

        .top-subtitle {
            font-size: 15px;
            color: #e5e7eb;
            margin-top: 5px;
            font-weight: 500;
        }

        .section-card {
            background: white;
            padding: 18px 20px 8px 20px;
            border-radius: 16px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
            margin-bottom: 16px;
        }

        .section-title {
            color: var(--navy);
            font-weight: 850;
            font-size: 16px;
            margin-bottom: 12px;
        }

        .result-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 20px 22px;
            margin-bottom: 14px;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
        }

        .result-label {
            color: var(--muted);
            font-size: 13px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 8px;
        }

        .result-value {
            color: var(--navy);
            font-size: 38px;
            font-weight: 900;
            line-height: 1.1;
        }

        .metric-note {
            color: #475569;
            font-size: 13px;
            line-height: 1.45;
            margin-top: 10px;
        }

        .decision-box {
            color: white;
            border-radius: 16px;
            padding: 24px 24px;
            margin-top: 8px;
            margin-bottom: 16px;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.12);
        }

        .approve { background: linear-gradient(90deg, #166534, #15803d); }
        .review { background: linear-gradient(90deg, #92400e, #d97706); }
        .reject { background: linear-gradient(90deg, #7f1d1d, #b91c1c); }

        .decision-label {
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            opacity: 0.92;
            font-weight: 800;
        }

        .decision-value {
            font-size: 46px;
            font-weight: 950;
            line-height: 1.05;
            margin-top: 4px;
        }

        .compact-rule {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            padding: 16px 18px;
            border-radius: 14px;
            color: #334155;
            font-size: 14px;
            line-height: 1.55;
            margin-top: 10px;
            box-shadow: 0 6px 14px rgba(15, 23, 42, 0.04);
        }

        .model-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 14px 16px;
            margin-top: 12px;
            color: #334155;
            font-size: 14px;
            line-height: 1.7;
        }

        .model-card strong {
            color: var(--navy);
        }

        div[data-testid="stFormSubmitButton"] > button {
            background-color: var(--gold) !important;
            color: #111827 !important;
            font-weight: 850 !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 11px 20px !important;
            width: 100% !important;
        }

        div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: #ffc233 !important;
            color: #111827 !important;
            border: none !important;
        }

        label, .stSelectbox label, .stSlider label {
            color: var(--navy) !important;
            font-weight: 750 !important;
        }

        div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: var(--navy) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
        }

        div[data-baseweb="select"] span {
            color: var(--navy) !important;
        }

        div[role="listbox"] {
            background-color: #ffffff !important;
        }

        div[role="option"] {
            color: var(--navy) !important;
            background-color: #ffffff !important;
        }

        div[role="option"]:hover {
            background-color: #eef2f7 !important;
        }

        [data-testid="stTickBar"],
        [data-testid="stSliderTickBarMin"],
        [data-testid="stSliderTickBarMax"] {
            display: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Load saved notebook artifacts
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
# Feature engineering and preprocessing
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

    grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
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
    """
    Apply the saved notebook preprocessing to a one-row Streamlit input.

    This is the key deployment fix:
    the app does not call get_dummies() on one row. Instead, it rebuilds the
    exact saved training dummy columns so dropdown categories affect the model.
    """
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
        X_clean[numeric_cols] = X_clean[numeric_cols].clip(
            lower=preprocess_artifacts["numeric_lower_caps"],
            upper=preprocess_artifacts["numeric_upper_caps"],
            axis=1
        )

    for col in categorical_cols:
        X_clean[col] = X_clean[col].astype("object")
        fill_value = preprocess_artifacts["category_fill_values"].get(col, "Missing")
        X_clean[col] = X_clean[col].fillna(fill_value)

    X_dummies = pd.DataFrame(0.0, index=X_clean.index, columns=dummy_columns)

    for col in numeric_cols:
        if col in X_dummies.columns:
            X_dummies[col] = pd.to_numeric(X_clean[col], errors="coerce").fillna(0).astype(float)

    for col in categorical_cols:
        for row_index, value in X_clean[col].items():
            dummy_col = f"{col}_{str(value)}"
            if dummy_col in X_dummies.columns:
                X_dummies.loc[row_index, dummy_col] = 1.0

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

    if full_paid_probability >= approve_threshold and predicted_return >= return_threshold:
        return "APPROVE"

    if full_paid_probability < reject_threshold or predicted_return < return_threshold:
        return "REJECT"

    return "REVIEW"


def predict_application(input_data):
    class_features = build_application_features(input_data, artifacts["final_class_feature_set"])
    class_dummies, class_scaled = apply_preprocessing(class_features, artifacts["final_class_preprocess_artifacts"])

    if artifacts["final_class_uses_scaled"]:
        class_model_input = class_scaled
    else:
        class_model_input = class_dummies

    full_paid_probability = artifacts["final_class_model"].predict_proba(class_model_input)[0][1]

    reg_features = build_application_features(input_data, artifacts["final_reg_feature_set"])
    reg_dummies, reg_scaled = apply_preprocessing(reg_features, artifacts["final_reg_preprocess_artifacts"])

    if artifacts["final_reg_uses_scaled"]:
        reg_model_input = reg_scaled
    else:
        reg_model_input = reg_dummies

    predicted_return = artifacts["final_reg_model"].predict(reg_model_input)[0]
    recommendation = make_recommendation(full_paid_probability, predicted_return)

    return full_paid_probability, predicted_return, recommendation


# ------------------------------------------------------------
# Scenario defaults and app state
# ------------------------------------------------------------

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
            "loan_amount": 22000,
            "term": "60 months",
            "int_rate": 18.5,
            "grade": "D",
            "purpose": "debt_consolidation",
            "annual_inc": 60000,
            "emp_length": "3 years",
            "home_ownership": "RENT",
            "verification_status": "Source Verified",
            "dti": 30.0,
            "fico_score": 680,
            "credit_history_years": 7,
            "open_acc": 8,
            "delinq_2yrs": 1,
            "pub_rec": 0,
            "revol_bal": 18000,
            "revol_util": 76.0
        },
        "Reject example": {
            "loan_amount": 40000,
            "term": "60 months",
            "int_rate": 30.0,
            "grade": "G",
            "purpose": "small_business",
            "annual_inc": 15000,
            "emp_length": "< 1 year",
            "home_ownership": "RENT",
            "verification_status": "Not Verified",
            "dti": 45.0,
            "fico_score": 640,
            "credit_history_years": 1,
            "open_acc": 1,
            "delinq_2yrs": 10,
            "pub_rec": 8,
            "revol_bal": 80000,
            "revol_util": 120.0
        }
    }

    return scenarios.get(scenario, scenarios["Review example"])


def reset_input_state(defaults):
    for key, value in defaults.items():
        st.session_state[key] = value


def format_purpose(value):
    return value.replace("_", " ").title()


def prepare_application_record(values):
    term_num = 36 if values["term"] == "36 months" else 60
    installment = calculate_installment(values["loan_amount"], values["int_rate"], term_num)

    issue_date = pd.Timestamp("2018-01-01")
    earliest_credit_date = issue_date - pd.DateOffset(years=int(values["credit_history_years"]))

    application_record = {
        "id": 0,
        "loan_amnt": values["loan_amount"],
        "funded_amnt": values["loan_amount"],
        "term": values["term"],
        "term_num": term_num,
        "int_rate": values["int_rate"],
        "installment": installment,
        "grade": values["grade"],
        "emp_length": values["emp_length"],
        "home_ownership": values["home_ownership"],
        "annual_inc": values["annual_inc"],
        "verification_status": values["verification_status"],
        "purpose": values["purpose"],
        "dti": values["dti"],
        "delinq_2yrs": values["delinq_2yrs"],
        "earliest_cr_line": earliest_credit_date.strftime("%Y-%m-%d"),
        "fico_range_low": values["fico_score"] - 2,
        "fico_range_high": values["fico_score"] + 2,
        "open_acc": values["open_acc"],
        "pub_rec": values["pub_rec"],
        "revol_bal": values["revol_bal"],
        "revol_util": values["revol_util"],
        "issue_d": issue_date.strftime("%Y-%m-%d")
    }

    return application_record, installment


# ------------------------------------------------------------
# Header and sidebar
# ------------------------------------------------------------

st.markdown(
    """
    <div class="top-header">
        <div class="top-title">Buckell Lending</div>
        <div class="top-subtitle">Loan Screening Dashboard</div>
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("### Model")
    st.write(f"Classifier: **{artifacts['final_class_model_name']}**")
    st.write(f"Regressor: **{artifacts['final_reg_model_name']}**")
    st.write(f"Approve threshold: **{artifacts['approve_probability_threshold']:.0%}**")
    st.write(f"Reject threshold: **{artifacts['reject_probability_threshold']:.0%}**")
    st.markdown("---")
    st.caption("Decision-support prototype. Not an automatic underwriting system.")


# ------------------------------------------------------------
# Scenario controls
# ------------------------------------------------------------

scenario_options = ["Review example", "Approve example", "Reject example", "Custom"]

if "scenario" not in st.session_state:
    st.session_state["scenario"] = "Review example"
    st.session_state["previous_scenario"] = "Review example"
    reset_input_state(scenario_defaults("Review example"))
    st.session_state["evaluated"] = False
    st.session_state["last_result"] = None

scenario = st.selectbox("Scenario", scenario_options, key="scenario")

if scenario != st.session_state.get("previous_scenario", scenario):
    if scenario != "Custom":
        reset_input_state(scenario_defaults(scenario))
    st.session_state["evaluated"] = False
    st.session_state["last_result"] = None
    st.session_state["previous_scenario"] = scenario


# ------------------------------------------------------------
# Input form
# ------------------------------------------------------------

purpose_options = [
    "debt_consolidation", "credit_card", "home_improvement", "major_purchase",
    "small_business", "car", "medical", "moving", "vacation", "house",
    "renewable_energy", "educational", "wedding", "other"
]
emp_options = ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"]
home_options = ["RENT", "MORTGAGE", "OWN", "OTHER", "ANY", "NONE"]
verification_options = ["Not Verified", "Source Verified", "Verified"]
grade_options = ["A", "B", "C", "D", "E", "F", "G"]
term_options = ["36 months", "60 months"]

left_col, right_col = st.columns([1.15, 0.85], gap="large")

with left_col:
    with st.form("loan_input_form"):
        st.markdown('<div class="section-card"><div class="section-title">Loan Terms</div>', unsafe_allow_html=True)
        loan_col_1, loan_col_2 = st.columns(2)

        with loan_col_1:
            st.slider("Loan amount", 1000, 40000, step=500, format="$%d", key="loan_amount")
            st.selectbox("Term", term_options, key="term")
            st.selectbox("Grade", grade_options, key="grade")

        with loan_col_2:
            st.slider("Interest rate", 5.0, 30.0, step=0.1, format="%.1f%%", key="int_rate")
            st.selectbox("Purpose", purpose_options, key="purpose", format_func=format_purpose)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card"><div class="section-title">Borrower Profile</div>', unsafe_allow_html=True)
        borrower_col_1, borrower_col_2 = st.columns(2)

        with borrower_col_1:
            st.slider("Annual income", 15000, 250000, step=5000, format="$%d", key="annual_inc")
            st.selectbox("Employment length", emp_options, key="emp_length")
            st.selectbox("Home ownership", home_options, key="home_ownership")

        with borrower_col_2:
            st.selectbox("Verification status", verification_options, key="verification_status")
            st.slider("Debt-to-income ratio", 0.0, 45.0, step=0.5, format="%.1f", key="dti")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card"><div class="section-title">Credit Profile</div>', unsafe_allow_html=True)
        credit_col_1, credit_col_2 = st.columns(2)

        with credit_col_1:
            st.slider("FICO score", 640, 850, step=5, key="fico_score")
            st.slider("Credit history years", 1, 40, step=1, key="credit_history_years")
            st.slider("Open accounts", 1, 40, step=1, key="open_acc")
            st.slider("Delinquencies, last 2 years", 0, 10, step=1, key="delinq_2yrs")

        with credit_col_2:
            st.slider("Public records", 0, 8, step=1, key="pub_rec")
            st.slider("Revolving balance", 0, 80000, step=1000, format="$%d", key="revol_bal")
            st.slider("Revolving utilization", 0.0, 120.0, step=1.0, format="%.0f%%", key="revol_util")

        st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("Evaluate Loan")


values = {
    "loan_amount": st.session_state["loan_amount"],
    "term": st.session_state["term"],
    "int_rate": st.session_state["int_rate"],
    "grade": st.session_state["grade"],
    "purpose": st.session_state["purpose"],
    "annual_inc": st.session_state["annual_inc"],
    "emp_length": st.session_state["emp_length"],
    "home_ownership": st.session_state["home_ownership"],
    "verification_status": st.session_state["verification_status"],
    "dti": st.session_state["dti"],
    "fico_score": st.session_state["fico_score"],
    "credit_history_years": st.session_state["credit_history_years"],
    "open_acc": st.session_state["open_acc"],
    "delinq_2yrs": st.session_state["delinq_2yrs"],
    "pub_rec": st.session_state["pub_rec"],
    "revol_bal": st.session_state["revol_bal"],
    "revol_util": st.session_state["revol_util"]
}

application_record, installment = prepare_application_record(values)

if submitted:
    try:
        full_paid_probability, predicted_return, recommendation = predict_application(application_record)
        st.session_state["evaluated"] = True
        st.session_state["last_result"] = {
            "full_paid_probability": full_paid_probability,
            "predicted_return": predicted_return,
            "recommendation": recommendation,
            "installment": installment
        }
    except Exception as error:
        st.session_state["evaluated"] = False
        st.session_state["last_result"] = None
        st.error("Prediction failed. Check that buckell_lending_model_artifacts.pkl is in the same repository as app.py.")
        st.exception(error)


# ------------------------------------------------------------
# Output panel
# ------------------------------------------------------------

with right_col:
    if not st.session_state.get("evaluated", False) or st.session_state.get("last_result") is None:
        st.markdown(
            """
            <div class="result-card">
                <div class="result-label">Status</div>
                <div class="result-value">Ready</div>
                <div class="metric-note">Select a scenario or adjust the controls, then evaluate the loan.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div class="compact-rule">
                <strong>Decision thresholds</strong><br>
                Approve: repayment probability ≥ {artifacts['approve_probability_threshold']:.0%} and nonnegative predicted return.<br>
                Review: borderline or mixed model output.<br>
                Reject: repayment probability &lt; {artifacts['reject_probability_threshold']:.0%} or negative predicted return.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        result = st.session_state["last_result"]
        full_paid_probability = result["full_paid_probability"]
        predicted_return = result["predicted_return"]
        recommendation = result["recommendation"]
        installment = result["installment"]

        decision_class = "approve" if recommendation == "APPROVE" else "review" if recommendation == "REVIEW" else "reject"

        st.markdown(
            f"""
            <div class="decision-box {decision_class}">
                <div class="decision-label">Recommendation</div>
                <div class="decision-value">{recommendation}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        metric_col_1, metric_col_2 = st.columns(2)

        with metric_col_1:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Full Repayment Probability</div>
                    <div class="result-value">{full_paid_probability:.1%}</div>
                    <div class="metric-note">Estimated chance that the borrower fully repays the loan.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with metric_col_2:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Predicted Pessimistic Return</div>
                    <div class="result-value">{predicted_return:.2f}%</div>
                    <div class="metric-note">Estimated downside return rate. Negative values indicate predicted loss.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        if recommendation == "APPROVE":
            rationale = "Repayment probability meets the approval threshold and predicted pessimistic return is nonnegative."
        elif recommendation == "REVIEW":
            rationale = "The application has borderline or mixed model output. Manual review is recommended."
        else:
            rationale = "The application falls below the repayment threshold or has negative predicted pessimistic return."

        st.markdown(
            f"""
            <div class="compact-rule">
                <strong>Decision basis</strong><br>
                {rationale}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="model-card">
                <strong>Model details</strong><br>
                Classifier: {artifacts['final_class_model_name']}<br>
                Regressor: {artifacts['final_reg_model_name']}<br>
                Calculated installment: ${installment:,.2f}<br>
                Approval threshold: {artifacts['approve_probability_threshold']:.0%}<br>
                Reject threshold: {artifacts['reject_probability_threshold']:.0%}
            </div>
            """,
            unsafe_allow_html=True
        )
