from __future__ import annotations

LOCATIONS = [
    {"uf": "R13", "city": "Metro-13", "lat": -23.5505, "lon": -46.6333, "weight": 0.22},
    {"uf": "R10", "city": "Metro-11", "lat": -22.9068, "lon": -43.1729, "weight": 0.12},
    {"uf": "R06", "city": "Metro-02", "lat": -19.9167, "lon": -43.9345, "weight": 0.10},
    {"uf": "R02", "city": "Metro-12", "lat": -12.9777, "lon": -38.5016, "weight": 0.08},
    {"uf": "R09", "city": "Metro-04", "lat": -25.4284, "lon": -49.2733, "weight": 0.08},
    {"uf": "R11", "city": "Metro-09", "lat": -30.0346, "lon": -51.2177, "weight": 0.07},
    {"uf": "R08", "city": "Metro-10", "lat": -8.0476, "lon": -34.8770, "weight": 0.06},
    {"uf": "R03", "city": "Metro-06", "lat": -3.7319, "lon": -38.5267, "weight": 0.06},
    {"uf": "R05", "city": "Metro-07", "lat": -16.6869, "lon": -49.2648, "weight": 0.05},
    {"uf": "R04", "city": "Metro-03", "lat": -15.7939, "lon": -47.8828, "weight": 0.05},
    {"uf": "R01", "city": "Metro-08", "lat": -3.1190, "lon": -60.0217, "weight": 0.04},
    {"uf": "R07", "city": "Metro-01", "lat": -1.4558, "lon": -48.4902, "weight": 0.04},
    {"uf": "R12", "city": "Metro-05", "lat": -27.5954, "lon": -48.5480, "weight": 0.03},
]

GENDERS = ["female", "male", "non_binary", "not_informed"]

INCOME_BANDS = [
    {"band": "ate_2_sm", "min": 1400.0, "max": 2800.0, "weight": 0.26},
    {"band": "2_a_4_sm", "min": 2800.0, "max": 5600.0, "weight": 0.34},
    {"band": "4_a_8_sm", "min": 5600.0, "max": 11200.0, "weight": 0.24},
    {"band": "8_a_15_sm", "min": 11200.0, "max": 21000.0, "weight": 0.11},
    {"band": "15_sm_mais", "min": 21000.0, "max": 42000.0, "weight": 0.05},
]

PRODUCTS = [
    {"product_type": "cartao_credito", "term_months": 24, "rate": 0.035, "weight": 0.34},
    {"product_type": "credito_pessoal", "term_months": 36, "rate": 0.028, "weight": 0.28},
    {"product_type": "veiculo", "term_months": 48, "rate": 0.022, "weight": 0.16},
    {"product_type": "consignado", "term_months": 60, "rate": 0.017, "weight": 0.14},
    {"product_type": "renegociacao", "term_months": 30, "rate": 0.025, "weight": 0.08},
]

CHANNELS = ["none", "sms", "whatsapp", "phone", "field_visit"]

PUBLIC_TABLES = ["borrowers", "loans", "actions", "monthly_panel"]

TERMINAL_STATES = {"cured", "charged_off"}
ALLOWED_STATES = {"performing", "delinquent", "credito_em_liquidacao", *TERMINAL_STATES}

TABLE_DOCS = {
    "borrowers": "One row per borrower — static cadastral, demographic and portfolio attributes.",
    "loans": "One row per loan — product, balances and the delinquency state at the panel start.",
    "actions": "One row per loan-month — the collection action taken (channel + settlement offer).",
    "monthly_panel": "One row per loan-month — the state, the outcome, and the evolving balance.",
}

FIELD_DOCS = {
    "borrower_id": "Borrower identifier (B######).",
    "loan_id": "Loan identifier (L######).",
    "action_id": "Action identifier (A########).",
    "month": "Month index within the panel window (0-based).",
    "uf": "Region code of the borrower.",
    "city": "City of the borrower.",
    "lat": "City-centroid latitude.",
    "lon": "City-centroid longitude.",
    "age": "Borrower age in years.",
    "gender": "Self-declared gender {female, male, non_binary, not_informed}.",
    "income_band": "Income bracket in minimum-wage multiples (e.g. ate_2_sm).",
    "monthly_income": "Monthly income (currency units), sampled within the income band.",
    "dependents": "Number of declared dependents.",
    "product_type": "Credit product (e.g. cartao_credito, credito_pessoal).",
    "origination_month": "Origination month relative to the window start (negative = prior).",
    "original_principal": "Original loan principal (currency units).",
    "outstanding_balance": "Outstanding balance (currency units); evolves month to month.",
    "interest_rate_monthly": "Contractual monthly interest rate.",
    "term_months": "Loan term in months.",
    "loan_age_months": "Months since origination.",
    "remaining_maturity_months": "Months remaining to maturity.",
    "days_past_due": "Days past due. In monthly_panel this is the value AFTER the month's "
                     "transition (0 if the loan cured that month).",
    "days_in_credito_em_liquidacao": "Days spent in credito_em_liquidacao (90+ dpd).",
    "initial_status": "Loan status at the panel start "
                      "{performing, delinquent, credito_em_liquidacao}.",
    "channel": "Contact channel used this month (in the released members: none, sms, whatsapp, phone).",
    "discount_offer": "Settlement discount offered this month, in [0, max_discount].",
    "settlement_offer_amount": "Settlement offered = outstanding_balance x (1 - discount_offer).",
    "is_exploration": "True if this action was a random ε-floor exploration draw (known propensity).",
    "exploration_epsilon": "Exploration probability in force for this row (0 when not explorable).",
    "n_allowed_channels": "Contact channels allowed for this loan-month (eligibility mask size); "
                          "exploration propensity = ε/n_allowed × 1/(range width).",
    "contact_cost": "Operational cost of the contact channel used (scenario-defined; the soft "
                    "cost mechanism — human agents cost far more than digital).",
    "status": "Loan status entering the month (before the transition).",
    "next_status": "Loan status after the month's transition.",
    "contacts_last_3m": "Number of contacts in the trailing three months.",
    "recovered": "True if the borrower paid/cured this month.",
    "charged_off": "True if the loan charged off this month.",
    "recovered_amount": "Cash actually collected this month (capped at the settlement offered).",
}
