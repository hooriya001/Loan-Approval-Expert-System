"""
Loan Approval Expert System
============================
A rule-based expert system that predicts loan approval using
decision rules derived from common lending criteria.

Usage:
    python loan_expert_system.py                # Interactive mode
    python loan_expert_system.py --batch data.csv  # Batch mode on CSV
"""

import sys
import csv
import os
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────
# Knowledge Base: Rules & Weights
# ──────────────────────────────────────────────

RULES = [
    {
        "id": "R1",
        "desc": "Credit history is positive",
        "condition": lambda f: f["credit_history"] == 1,
        "weight": 40,
    },
    {
        "id": "R2",
        "desc": "Applicant is a graduate",
        "condition": lambda f: f["education"] == "graduate",
        "weight": 10,
    },
    {
        "id": "R3",
        "desc": "Total income ≥ 5000",
        "condition": lambda f: f["total_income"] >= 5000,
        "weight": 15,
    },
    {
        "id": "R4",
        "desc": "Loan-to-income ratio ≤ 4",
        "condition": lambda f: f["loan_to_income"] <= 4,
        "weight": 15,
    },
    {
        "id": "R5",
        "desc": "Applicant is married",
        "condition": lambda f: f["married"] == "yes",
        "weight": 5,
    },
    {
        "id": "R6",
        "desc": "Property area is semiurban or urban",
        "condition": lambda f: f["property_area"] in ("semiurban", "urban"),
        "weight": 5,
    },
    {
        "id": "R7",
        "desc": "No excessive dependents (≤ 2)",
        "condition": lambda f: f["dependents"] <= 2,
        "weight": 5,
    },
    {
        "id": "R8",
        "desc": "Not self-employed",
        "condition": lambda f: f["self_employed"] == "no",
        "weight": 5,
    },
]

APPROVAL_THRESHOLD = 55  # out of 100


# ──────────────────────────────────────────────
# Inference Engine
# ──────────────────────────────────────────────

def derive_features(raw: dict) -> dict:
    """Compute derived facts from raw inputs."""
    applicant_income = float(raw.get("applicant_income", 0))
    coapplicant_income = float(raw.get("coapplicant_income", 0))
    loan_amount = float(raw.get("loan_amount", 1))
    total_income = applicant_income + coapplicant_income
    return {
        **raw,
        "total_income": total_income,
        "loan_to_income": (loan_amount * 1000) / total_income if total_income > 0 else 999,
    }


def infer(raw: dict) -> dict:
    """Run all rules and return verdict + explanation."""
    facts = derive_features(raw)
    fired, not_fired = [], []
    score = 0

    for rule in RULES:
        try:
            if rule["condition"](facts):
                fired.append(rule)
                score += rule["weight"]
            else:
                not_fired.append(rule)
        except Exception:
            not_fired.append(rule)

    approved = score >= APPROVAL_THRESHOLD
    return {
        "score": score,
        "threshold": APPROVAL_THRESHOLD,
        "approved": approved,
        "decision": "APPROVED ✅" if approved else "REJECTED ❌",
        "rules_fired": fired,
        "rules_not_fired": not_fired,
    }


# ──────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────

def normalize_raw(row: dict) -> dict:
    """Normalise column names from various CSV formats."""
    def g(keys, default=""):
        for k in keys:
            if k in row:
                return str(row[k]).strip().lower()
        return default

    dep = g(["Dependents", "dependents"], "0").replace("+", "")
    try:
        dep = int(dep)
    except ValueError:
        dep = 0

    cr = g(["Credit_History", "credit_history"], "0")
    try:
        cr = int(float(cr))
    except ValueError:
        cr = 0

    return {
        "gender": g(["Gender", "gender"], "male"),
        "married": g(["Married", "married"], "no"),
        "dependents": dep,
        "education": g(["Education", "education"], "graduate"),
        "self_employed": g(["Self_Employed", "self_employed"], "no"),
        "applicant_income": float(g(["ApplicantIncome", "applicant_income"], "0") or 0),
        "coapplicant_income": float(g(["CoapplicantIncome", "coapplicant_income"], "0") or 0),
        "loan_amount": float(g(["LoanAmount", "loan_amount"], "1") or 1),
        "loan_amount_term": float(g(["Loan_Amount_Term", "loan_amount_term"], "360") or 360),
        "credit_history": cr,
        "property_area": g(["Property_Area", "property_area"], "urban"),
    }


def print_result(result: dict, label: str = ""):
    if label:
        print(f"\n{'='*50}")
        print(f"  {label}")
        print(f"{'='*50}")
    print(f"\n  Decision : {result['decision']}")
    print(f"  Score    : {result['score']} / 100  (threshold: {result['threshold']})")
    print(f"\n  Rules FIRED:")
    for r in result["rules_fired"]:
        print(f"    ✔ [{r['id']}] {r['desc']}  (+{r['weight']})")
    print(f"\n  Rules NOT fired:")
    for r in result["rules_not_fired"]:
        print(f"    ✘ [{r['id']}] {r['desc']}  (+{r['weight']})")
    print()


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────

def predict(data: dict):
    """Wrapper around infer() that returns tuple format."""
    result = infer(data)
    return (
        result["score"],
        result["decision"],
        result["rules_fired"],
        result["rules_not_fired"],
    )


def get_input() -> dict:
    """Get user input interactively."""
    print("\n🏦  LOAN APPROVAL EXPERT SYSTEM")
    print("=" * 40)

    def ask(prompt, options=None, cast=str):
        while True:
            val = input(f"  {prompt}: ").strip().lower()
            if options and val not in options:
                print(f"    ⚠ Choose from: {', '.join(options)}")
                continue
            try:
                return cast(val)
            except ValueError:
                print("    ⚠ Invalid input, try again.")

    return {
        "gender": ask("Gender (male/female)", ["male", "female"]),
        "married": ask("Married (yes/no)", ["yes", "no"]),
        "dependents": ask("Dependents (0/1/2/3+)", cast=lambda x: int(x.replace("+", ""))),
        "education": ask("Education (graduate/not graduate)", ["graduate", "not graduate"]),
        "self_employed": ask("Self-employed (yes/no)", ["yes", "no"]),
        "applicant_income": ask("Applicant monthly income", cast=float),
        "coapplicant_income": ask("Co-applicant monthly income", cast=float),
        "loan_amount": ask("Loan amount (in thousands)", cast=float),
        "loan_amount_term": ask("Loan term in months (e.g. 360)", cast=float),
        "credit_history": ask("Credit history (1=good / 0=bad)", ["0", "1"], cast=int),
        "property_area": ask("Property area (urban/semiurban/rural)", ["urban", "semiurban", "rural"]),
    }


def show_graphs(score: int, decision: str, fired: list, not_fired: list):
    # ─────────────── 1. Score vs Threshold ───────────────
    plt.figure()
    plt.bar(["Score", "Threshold"], [score, APPROVAL_THRESHOLD])
    plt.title("Loan Score vs Threshold")
    plt.ylim(0, 100)
    plt.show()

    # ─────────────── 2. Fired vs Not Fired ───────────────
    plt.figure()
    plt.pie(
        [len(fired), len(not_fired)],
        labels=["Fired", "Not Fired"],
        autopct="%1.1f%%"
    )
    plt.title("Rule Evaluation")
    plt.show()

    # ─────────────── 3. Fired Rules Contribution ───────────────
    if fired:
        names = [r["id"] for r in fired]
        weights = [r["weight"] for r in fired]

        plt.figure()
        plt.bar(names, weights)
        plt.title("Fired Rules Contribution")
        plt.xlabel("Rules")
        plt.ylabel("Weight")
        plt.show()

    # ─────────────── 4. Not Fired Rules ───────────────
    if not_fired:
        names = [r["id"] for r in not_fired]
        weights = [r["weight"] for r in not_fired]

        plt.figure()
        plt.bar(names, weights)
        plt.title("Not Fired Rules")
        plt.xlabel("Rules")
        plt.ylabel("Weight")
        plt.show()

    # ─────────────── 5. Final Decision ───────────────
    plt.figure()
    plt.bar(
        ["Approved", "Rejected"],
        [1 if "APPROVED" in decision else 0,
         1 if "REJECTED" in decision else 0]
    )
    plt.title("Final Decision")
    plt.ylim(0, 1)
    plt.show()


# ──────────────────────────────────────────────
# GUI Mode
# ──────────────────────────────────────────────

def gui_app():
    window = tk.Tk()
    window.title("Loan Approval Expert System")
    window.geometry("400x600")

    entries = {}

    fields = [
        ("Gender (male/female)", "gender"),
        ("Married (yes/no)", "married"),
        ("Dependents", "dependents"),
        ("Education (graduate/not graduate)", "education"),
        ("Self Employed (yes/no)", "self_employed"),
        ("Applicant Income", "applicant_income"),
        ("Coapplicant Income", "coapplicant_income"),
        ("Loan Amount (in thousands)", "loan_amount"),
        ("Loan Term", "loan_amount_term"),
        ("Credit History (1/0)", "credit_history"),
        ("Property Area (urban/semiurban/rural)", "property_area"),
    ]

    for label, key in fields:
        tk.Label(window, text=label).pack()
        entry = tk.Entry(window)
        entry.pack()
        entries[key] = entry

    def submit():
        try:
            data = {
                "gender": entries["gender"].get().strip().lower(),
                "married": entries["married"].get().strip().lower(),
                "dependents": int(entries["dependents"].get()),
                "education": entries["education"].get().strip().lower(),
                "self_employed": entries["self_employed"].get().strip().lower(),
                "applicant_income": float(entries["applicant_income"].get()),
                "coapplicant_income": float(entries["coapplicant_income"].get()),
                "loan_amount": float(entries["loan_amount"].get()),
                "loan_amount_term": float(entries["loan_amount_term"].get()),
                "credit_history": int(entries["credit_history"].get()),
                "property_area": entries["property_area"].get().strip().lower(),
            }

            score, decision, fired, not_fired = predict(data)

            result_text = f"Score: {score}\nDecision: {decision}\n\nRules Fired: {len(fired)}"

            messagebox.showinfo("Result", result_text)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    tk.Button(window, text="Predict", command=submit).pack(pady=20)

    window.mainloop()


# ──────────────────────────────────────────────
# Interactive Mode
# ──────────────────────────────────────────────

def interactive():
    data = get_input()
    score, decision, fired, not_fired = predict(data)

    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print(" RESULT")
    print("━━━━━━━━━━━━━━━━━━━━━━")
    print("Score:", score)
    print("Decision:", decision)
    print("\nRules Fired:", len(fired))
    print("Rules Not Fired:", len(not_fired))

    show_graphs(score, decision, fired, not_fired)


# ──────────────────────────────────────────────
# Batch Mode
# ──────────────────────────────────────────────

def batch(file_name):
    try:
        with open(file_name, newline='') as file:
            reader = csv.DictReader(file)

            print("\n📂 PROCESSING CSV...\n")

            for i, row in enumerate(reader, start=1):
                try:
                    # ✅ USE THIS (fixes your error)
                    data = normalize_raw(row)

                    score, decision, fired, not_fired = predict(data)

                    print(f"\n--- Applicant #{i} ---")
                    print("Score:", score)
                    print("Decision:", decision)

                    # ✅ keep your graphs / rule display
                    show_graphs(score, decision, fired, not_fired)

                except Exception as e:
                    print(f"⚠️ Error in row {i}: {e}")

    except FileNotFoundError:
        print("❌ File not found!")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("\nSelect Mode:")
    print("1. Manual Input (Terminal)")
    print("2. CSV Batch Mode")
    print("3. GUI Mode (Tkinter)")

    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "2":
        file_name = input("Enter CSV file name: ").strip()
        batch(file_name)
    elif choice == "3":
        gui_app()
    else:
        interactive()