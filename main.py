import json
from datetime import datetime
import joblib
import pandas as pd
from pymongo import MongoClient
from bson import ObjectId


client = MongoClient("mongodb://localhost:27017/")
db = client["bank_churn_db"]
records = db["prediction_history"]

model = joblib.load("model.pkl")


def predict_customer():
    print("\nEnter customer details:")
    raw = input("Name, CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary: ")

    try:
        values = [x.strip() for x in raw.split(",")]

        if len(values) != 11:
            print("Enter exactly 11 values.")
            return

        name, score, country, gender, age, tenure, balance, products, card, active, salary = values

        customer = pd.DataFrame([{
            "CreditScore": int(score),
            "Geography": country,
            "Gender": gender,
            "Age": int(age),
            "Tenure": int(tenure),
            "Balance": float(balance),
            "NumOfProducts": int(products),
            "HasCrCard": int(card),
            "IsActiveMember": int(active),
            "EstimatedSalary": float(salary)
        }])


        prediction = int(model.predict(customer)[0])
        probability = float(model.predict_proba(customer)[0][1])

        label = "Likely to Churn" if prediction == 1 else "Likely to Stay"

        record = {
            "name": name,
            "credit_score": int(score),
            "country": country,
            "gender": gender,
            "age": int(age),
            "tenure": int(tenure),
            "balance": float(balance),
            "products": int(products),
            "has_card": int(card),
            "active_member": int(active),
            "salary": float(salary),
            "prediction": prediction,
            "result": label,
            "probability": round(probability, 4),
            "time": datetime.now()
        }

        print("Saving record to MongoDB...")

        saved = records.insert_one(record)

        print("\nPrediction:", label)
        print("Churn Probability:", f"{probability:.2%}")
        print("MongoDB ID:", saved.inserted_id)

    except Exception as e:
        print("Error:", e)
   

def show_all():
    docs = records.find().sort("_id", -1).limit(10)

    if records.count_documents({}) == 0:
        print("No records found.")
        return

    for doc in docs:
        print(json.dumps(doc, default=str, indent=2))


def find_one():
    print("\nExample ID: 665f1a2b3c4d5e6f78901234")
    record_id = input("Enter MongoDB record ID: ").strip()
    try:
        doc = records.find_one({"_id": ObjectId(record_id)})
        print(json.dumps(doc, default=str, indent=2) if doc else "No record found.")
    except Exception:
        print("Invalid ID. Copy the ID printed after a prediction is saved.")


def update_note():
    print("\nExample input: <record_id>, Needs follow-up call")
    data = input("Enter record ID and note, separated by a comma: ").split(",", 1)
    if len(data) != 2:
        print("Please enter both ID and note separated by a comma.")
        return
    try:
        result = records.update_one(
            {"_id": ObjectId(data[0].strip())},
            {"$set": {"note": data[1].strip()}}
        )
        
    except Exception:
        print("Invalid MongoDB ID.")


def delete_one():
    print("\nExample input: <record_id> (use a real ID from your records)")
    record_id = input("Enter MongoDB record ID to delete: ").strip()
    if input("Type YES to confirm deletion: ").strip() != "YES":
        print("Delete cancelled.")
        return
    try:
        result = records.delete_one({"_id": ObjectId(record_id)})
        if result.deleted_count:
            print("Record deleted successfully.")
        else:
            print("No record found with that ID.")
    except Exception:
        print("Invalid MongoDB ID.")


def count_records():
    print("\nRecord Counts:")
    print("Total: ", records.count_documents({}))
    print("Churn: ", records.count_documents({"prediction": 1}))
    print("Stay:  ", records.count_documents({"prediction": 0}))


def filter_by_age():
    print("\nExample input: 30 (shows customers older than 30)")
    try:
        age = int(input("Show customers older than age: "))
        docs = list(records.find({"age": {"$gt": age}}).sort("age", -1))
        print(json.dumps(docs, default=str, indent=2) if docs else "No matching records.")
    except ValueError:
        print("Enter age as a whole number.")


def sort_by_age():
    print("\nExample input: 1 for ascending, 2 for descending")
    choice = input("Choose sort order: ").strip()
    if choice not in ["1", "2"]:
        print("Enter 1 or 2.")
        return
    direction = 1 if choice == "1" else -1
    docs = list(records.find().sort("age", direction).limit(10))
    print(json.dumps(docs, default=str, indent=2) if docs else "No records found.")


def summary():
    print("\n--- Aggregation Summary ---")
    docs = list(records.aggregate([
        {"$group": {
            "_id": "$result",
            "total": {"$sum": 1},
            "average_probability": {"$avg": "$probability"}
        }},
        {"$sort": {"total": -1}}
    ]))
    print(json.dumps(docs, default=str, indent=2) if docs else "No records to summarize.")


def menu():
    while True:
        try:
            print("\n====================================")
            print("   Bank Churn MongoDB Tester")
            print("====================================")
            print("1. Predict customer and save to MongoDB")
            print("2. Show all recent records")
            print("3. Find one record by ID")
            print("4. Update a record note")
            print("5. Delete one record")
            print("6. Count records")
            print("7. Filter customers by age")
            print("8. Sort records by age")
            print("9. Aggregation summary")
            print("0. Exit")

            choice = input("Choose an option: ").strip()

            if choice == "1":
                predict_customer()
            elif choice == "2":
                show_all()
            elif choice == "3":
                find_one()
            elif choice == "4":
                update_note()
            elif choice == "5":
                delete_one()
            elif choice == "6":
                count_records()
            elif choice == "7":
                filter_by_age()
            elif choice == "8":
                sort_by_age()
            elif choice == "9":
                summary()
            elif choice == "0":
                print("Goodbye!")
                break
            else:
                print("Invalid choice.")

        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            break


if __name__ == "__main__":
    try:
        client.admin.command("ping")
        print("✓ MongoDB connected successfully.")
        menu()
    except Exception as error:
        print("Could not connect to MongoDB. Check that it is running.")
        print("Details:", error)
    finally:
        client.close()
