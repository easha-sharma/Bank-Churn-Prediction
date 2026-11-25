import streamlit as st
import pandas as pd
import joblib
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import os

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏦 Bank Churn Predictor",
    page_icon="🏦",
    layout="wide",
)

# ─── MongoDB Connection ──────────────────────────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")

@st.cache_resource
def get_db():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    client.admin.command("ping")          # raises if not reachable
    db = client["bank_churn_db"]
    return db["prediction_history"]

@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)
    return joblib.load(os.path.join(base, "model.pkl"))

# ─── Helpers ─────────────────────────────────────────────────────────────────
def safe_object_id(raw: str):
    try:
        return ObjectId(raw.strip())
    except Exception:
        return None

# ─── Bootstrap ───────────────────────────────────────────────────────────────
try:
    records = get_db()
    DB_OK = True
except Exception as exc:
    DB_OK = False
    DB_ERR = str(exc)

try:
    model = load_model()
    MODEL_OK = True
except Exception as exc:
    MODEL_OK = False
    MODEL_ERR = str(exc)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("🏦 Bank Churn App")
st.sidebar.markdown("---")

if DB_OK:
    st.sidebar.success("✅ MongoDB connected")
else:
    st.sidebar.error("❌ MongoDB offline")

if MODEL_OK:
    st.sidebar.success("✅ Model loaded")
else:
    st.sidebar.error("❌ Model not found")

page = st.sidebar.radio(
    "Navigate",
    [
        "🔮 Predict & Save",
        "📋 All Records",
        "🔍 Find by ID",
        "✏️ Update Note",
        "🗑️ Delete Record",
        "📊 Count Records",
        "🎯 Filter by Age",
        "🔃 Sort by Age",
        "📈 Aggregation Summary",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("Bank Churn Prediction · MongoDB + Streamlit")

# ─── Guard ───────────────────────────────────────────────────────────────────
if not DB_OK or not MODEL_OK:
    st.error("### App cannot start")
    if not DB_OK:
        st.warning(f"**MongoDB:** {DB_ERR}")
        st.info("Start MongoDB with `brew services start mongodb-community` and refresh.")
    if not MODEL_OK:
        st.warning(f"**Model:** {MODEL_ERR}")
        st.info("Run `python train_model.py` to generate `model.pkl`, then refresh.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Predict & Save
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Predict & Save":
    st.title("🔮 Predict Customer Churn")
    st.markdown("Fill in the customer details and click **Predict**.")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            name        = st.text_input("Customer Name", placeholder="Alice")
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
            geography   = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender      = st.selectbox("Gender", ["Male", "Female"])

        with col2:
            age         = st.number_input("Age", min_value=18, max_value=100, value=35)
            tenure      = st.number_input("Tenure (years)", min_value=0, max_value=10, value=5)
            balance     = st.number_input("Balance ($)", min_value=0.0, value=50000.0, step=1000.0)

        with col3:
            num_products = st.number_input("Num of Products", min_value=1, max_value=4, value=1)
            has_cr_card  = st.selectbox("Has Credit Card?", ["Yes", "No"])
            is_active    = st.selectbox("Is Active Member?", ["Yes", "No"])
            salary       = st.number_input("Estimated Salary ($)", min_value=0.0, value=80000.0, step=1000.0)

        submitted = st.form_submit_button("⚡ Predict", use_container_width=True)

    if submitted:
        if not name.strip():
            st.warning("Please enter a customer name.")
        else:
            customer_df = pd.DataFrame([{
                "CreditScore":     int(credit_score),
                "Geography":       geography,
                "Gender":          gender,
                "Age":             int(age),
                "Tenure":          int(tenure),
                "Balance":         float(balance),
                "NumOfProducts":   int(num_products),
                "HasCrCard":       1 if has_cr_card == "Yes" else 0,
                "IsActiveMember":  1 if is_active   == "Yes" else 0,
                "EstimatedSalary": float(salary),
            }])

            prediction  = int(model.predict(customer_df)[0])
            probability = float(model.predict_proba(customer_df)[0][1])
            label       = "Likely to Churn" if prediction == 1 else "Likely to Stay"

            # Save to MongoDB
            doc = {
                "name":          name.strip(),
                "credit_score":  int(credit_score),
                "country":       geography,
                "gender":        gender,
                "age":           int(age),
                "tenure":        int(tenure),
                "balance":       float(balance),
                "products":      int(num_products),
                "has_card":      1 if has_cr_card == "Yes" else 0,
                "active_member": 1 if is_active   == "Yes" else 0,
                "salary":        float(salary),
                "prediction":    prediction,
                "result":        label,
                "probability":   round(probability, 4),
                "time":          datetime.now(),
            }
            saved = records.insert_one(doc)

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Prediction", label)
            c2.metric("Churn Probability", f"{probability:.2%}")
            c3.metric("MongoDB ID", str(saved.inserted_id)[:10] + "…")

            if prediction == 1:
                st.error(f"⚠️ **{name}** is **Likely to Churn** with probability **{probability:.2%}**.")
            else:
                st.success(f"✅ **{name}** is **Likely to Stay** with churn probability **{probability:.2%}**.")

            st.info(f"Record saved to MongoDB with ID: `{saved.inserted_id}`")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: All Records
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 All Records":
    st.title("📋 All Recent Records")

    total = records.count_documents({})
    limit = st.slider("Max records to display", 5, 100, 10)

    if total == 0:
        st.info("No records yet. Use **Predict & Save** to add some.")
    else:
        docs = list(records.find().sort("_id", -1).limit(limit))
        rows = []
        for d in docs:
            rows.append({
                "ID":          str(d["_id"]),
                "Name":        d.get("name", "—"),
                "Country":     d.get("country", "—"),
                "Gender":      d.get("gender", "—"),
                "Age":         d.get("age", "—"),
                "Result":      d.get("result", "—"),
                "Probability": f"{d.get('probability', 0):.2%}",
                "Note":        d.get("note", ""),
                "Time":        str(d.get("time", ""))[:19],
            })
        df = pd.DataFrame(rows)

        def colour_result(val):
            return "background-color: #ffd6d6" if val == "Likely to Churn" else "background-color: #d6ffd6"

        st.dataframe(
            df.style.map(colour_result, subset=["Result"]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"Showing {len(docs)} of {total} total records (newest first).")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Find by ID
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Find by ID":
    st.title("🔍 Find Record by ID")
    record_id = st.text_input("MongoDB Record ID", placeholder="e.g. 665f1a2b3c4d5e6f78901234")

    if st.button("🔍 Find", use_container_width=True):
        oid = safe_object_id(record_id)
        if oid is None:
            st.error("Invalid ID format. It must be a 24-character hex string.")
        else:
            doc = records.find_one({"_id": oid})
            if doc:
                doc["_id"] = str(doc["_id"])
                st.json(doc)
            else:
                st.warning("No record found with that ID.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Update Note
# ══════════════════════════════════════════════════════════════════════════════
elif page == "✏️ Update Note":
    st.title("✏️ Update a Record Note")
    record_id = st.text_input("Record ID", placeholder="24-char hex ID")
    note      = st.text_area("Note", placeholder="e.g. Needs retention offer")

    if st.button("💾 Save Note", use_container_width=True):
        if not record_id.strip():
            st.warning("Please enter a record ID.")
        elif not note.strip():
            st.warning("Please enter a note.")
        else:
            oid = safe_object_id(record_id)
            if oid is None:
                st.error("Invalid ID format.")
            else:
                result = records.update_one(
                    {"_id": oid},
                    {"$set": {"note": note.strip()}}
                )
                if result.matched_count:
                    st.success("✅ Note updated successfully.")
                else:
                    st.warning("No record found with that ID.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Delete Record
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗑️ Delete Record":
    st.title("🗑️ Delete a Record")
    record_id = st.text_input("Record ID to delete", placeholder="24-char hex ID")

    st.warning("⚠️ This action is **irreversible**.")
    confirm = st.checkbox("I understand this record will be permanently deleted.")

    if st.button("🗑️ Delete", disabled=not confirm, use_container_width=True):
        if not record_id.strip():
            st.warning("Please enter a record ID.")
        else:
            oid = safe_object_id(record_id)
            if oid is None:
                st.error("Invalid ID format.")
            else:
                result = records.delete_one({"_id": oid})
                if result.deleted_count:
                    st.success("✅ Record deleted successfully.")
                else:
                    st.warning("No record found with that ID.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Count Records
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Count Records":
    st.title("📊 Record Counts")

    total = records.count_documents({})
    churn = records.count_documents({"prediction": 1})
    stay  = records.count_documents({"prediction": 0})

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", total)
    c2.metric("Likely to Churn 🔴", churn)
    c3.metric("Likely to Stay 🟢", stay)

    if total > 0:
        st.markdown("---")
        st.subheader("Distribution")
        dist_df = pd.DataFrame({
            "Label": ["Likely to Stay", "Likely to Churn"],
            "Count": [stay, churn],
        })
        st.bar_chart(dist_df.set_index("Label"))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Filter by Age
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Filter by Age":
    st.title("🎯 Filter Customers by Age")
    age_threshold = st.number_input("Show customers **older than** age:", min_value=0, max_value=99, value=30)

    if st.button("🔎 Filter", use_container_width=True):
        docs = list(records.find({"age": {"$gt": int(age_threshold)}}).sort("age", -1))
        if not docs:
            st.info(f"No customers older than {age_threshold}.")
        else:
            rows = []
            for d in docs:
                rows.append({
                    "ID":          str(d["_id"]),
                    "Name":        d.get("name", "—"),
                    "Age":         d.get("age", "—"),
                    "Country":     d.get("country", "—"),
                    "Result":      d.get("result", "—"),
                    "Probability": f"{d.get('probability', 0):.2%}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(f"{len(docs)} record(s) found.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Sort by Age
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔃 Sort by Age":
    st.title("🔃 Sort Records by Age")
    order   = st.radio("Sort order", ["Ascending ↑", "Descending ↓"], horizontal=True)
    top_n   = st.slider("Number of records", 5, 50, 10)

    if st.button("📂 Load", use_container_width=True):
        direction = 1 if order.startswith("Ascending") else -1
        docs = list(records.find().sort("age", direction).limit(top_n))
        if not docs:
            st.info("No records found.")
        else:
            rows = []
            for d in docs:
                rows.append({
                    "ID":          str(d["_id"]),
                    "Name":        d.get("name", "—"),
                    "Age":         d.get("age", "—"),
                    "Country":     d.get("country", "—"),
                    "Gender":      d.get("gender", "—"),
                    "Result":      d.get("result", "—"),
                    "Probability": f"{d.get('probability', 0):.2%}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Aggregation Summary
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Aggregation Summary":
    st.title("📈 Aggregation Summary")

    docs = list(records.aggregate([
        {"$group": {
            "_id":                "$result",
            "total":              {"$sum": 1},
            "average_probability":{"$avg": "$probability"},
        }},
        {"$sort": {"total": -1}},
    ]))

    if not docs:
        st.info("No records to aggregate. Add predictions first.")
    else:
        st.subheader("By Prediction Label")
        label_df = pd.DataFrame([{
            "Label":               d["_id"],
            "Count":               d["total"],
            "Avg Churn Probability": f"{d['average_probability']:.2%}",
        } for d in docs])
        st.dataframe(label_df, use_container_width=True, hide_index=True)
        st.bar_chart(label_df.set_index("Label")["Count"])

    st.markdown("---")

    # By geography
    geo_docs = list(records.aggregate([
        {"$group": {
            "_id":         "$country",
            "total":       {"$sum": 1},
            "avg_balance": {"$avg": "$balance"},
            "churn_count": {"$sum": "$prediction"},
        }},
        {"$sort": {"total": -1}},
    ]))

    if geo_docs:
        st.subheader("By Geography")
        geo_df = pd.DataFrame([{
            "Country":     d["_id"],
            "Total":       d["total"],
            "Churn Count": d["churn_count"],
            "Avg Balance": f"${d['avg_balance']:,.0f}",
        } for d in geo_docs])
        st.dataframe(geo_df, use_container_width=True, hide_index=True)
        st.bar_chart(geo_df.set_index("Country")[["Total", "Churn Count"]])
