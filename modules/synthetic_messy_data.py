"""
=============================================================================
Thiranex Solutions — Synthetic Messy Dataset Generator
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_messy_enterprise_dataset(domain: str = "Customer CRM & Sales", row_count: int = 250, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic enterprise dataset injected with intentional quality issues:
    - Missing values (NaN, null, empty strings)
    - Exact & Fuzzy duplicates
    - Mixed date formats
    - Inconsistent text casing & spaces
    - Unformatted phone numbers & invalid emails
    - Numeric outliers & extreme values
    - Mixed unit values (e.g. kg/lbs)
    """
    np.random.seed(seed)
    random.seed(seed)
    
    if domain == "Customer CRM & Sales":
        return _generate_crm_dataset(row_count)
    elif domain == "Financial Transactions":
        return _generate_financial_dataset(row_count)
    elif domain == "HR & Employee Records":
        return _generate_hr_dataset(row_count)
    else:
        return _generate_crm_dataset(row_count)

def _generate_crm_dataset(rows: int) -> pd.DataFrame:
    first_names = ["John", "john", "JOHN", "Jane", "Jane ", "JANE", "Michael", "Michal", "Sarah", "Sara", "Robert", "Rob", "Emily", "Emlee", "David", "Dave", "Lisa", "Lisa M."]
    last_names = ["Smith", "smith", "SMITH", "Johnson", "Johnston", "Williams", "Brown", "Jones", "Miller", "Davis", "G Garcia", "Rodriguez", "Wilson"]
    cities = ["New York", "new york", "NEW YORK", "Los Angeles", "L.A.", "Chicago", "chicago", "Houston", "houston", "Phoenix", "Philadelphia", "San Antonio"]
    states = ["NY", "ny", "N.Y.", "CA", "ca", "IL", "TX", "tx", "AZ", "PA"]
    
    data = []
    base_date = datetime(2025, 1, 1)
    
    for i in range(1, rows + 1):
        fn = random.choice(first_names)
        ln = random.choice(last_names)
        
        # Email generation with some invalid formats
        if random.random() < 0.08:
            email = f"{fn.lower()}.{ln.lower()}at_company.com" # Missing @
        elif random.random() < 0.08:
            email = f"{fn.lower()}{ln.lower()}@domain" # Missing TLD
        else:
            email = f"{fn.strip().lower()}.{ln.strip().lower()}@thiranex-client.org"
            
        # Phone number with mixed formatting
        p_raw = "".join([str(random.randint(0, 9)) for _ in range(10)])
        p_fmt_choices = [
            f"({p_raw[:3]}) {p_raw[3:6]}-{p_raw[6:]}",
            f"{p_raw[:3]}-{p_raw[3:6]}-{p_raw[6:]}",
            f"+1 {p_raw}",
            f"{p_raw}",
            f"{p_raw[:3]}.{p_raw[3:6]}.{p_raw[6:]}"
        ]
        phone = random.choice(p_fmt_choices)
        
        # Mixed Date Formats
        dt_val = base_date + timedelta(days=random.randint(0, 365), hours=random.randint(0, 23))
        dt_choices = [
            dt_val.strftime("%Y-%m-%d"),
            dt_val.strftime("%m/%d/%Y"),
            dt_val.strftime("%d-%b-%Y"),
            dt_val.strftime("%Y/%m/%d %H:%M:%S"),
            dt_val.strftime("%B %d, %Y")
        ]
        signup_date = random.choice(dt_choices)
        
        # Numeric values with outliers
        if random.random() < 0.05:
            annual_spend = random.choice([999999.00, -5000.00, 1500000.50]) # Extreme outlier
        else:
            annual_spend = round(random.uniform(500.0, 25000.0), 2)
            
        credit_score = random.choice([300, 650, 720, 810, 850, 9999, -50]) if random.random() < 0.05 else random.randint(580, 820)
        
        # Age
        age = random.choice([150, -5, 200]) if random.random() < 0.04 else random.randint(18, 75)
        
        # Customer status
        status = random.choice(["Active", "active", "ACTIVE", "Pending", "pending", "Inactive", "INACTIVE", "Churned"])
        
        # Weight with mixed units
        w_val = round(random.uniform(50.0, 100.0), 1)
        unit = random.choice(["kg", "lbs", "kg"])
        weight_str = f"{w_val} {unit}"
        
        data.append({
            "Customer_ID": f"CUST-{1000 + i}",
            "First_Name": fn,
            "Last_Name": ln,
            "Email": email,
            "Phone_Number": phone,
            "City": random.choice(cities),
            "State": random.choice(states),
            "Signup_Date": signup_date,
            "Age": age,
            "Credit_Score": credit_score,
            "Annual_Spend_USD": annual_spend,
            "Weight_Recorded": weight_str,
            "Account_Status": status
        })
        
    df = pd.DataFrame(data)
    
    # Inject missing values (NaNs) into random positions
    for col in ["Email", "Phone_Number", "City", "Age", "Credit_Score", "Annual_Spend_USD", "Account_Status"]:
        mask = np.random.rand(len(df)) < 0.12
        df.loc[mask, col] = np.nan
        
    # Inject exact duplicate rows
    dupes = df.iloc[:15].copy()
    df = pd.concat([df, dupes], ignore_index=True)
    
    # Inject fuzzy duplicates (slightly altered names/emails)
    fuzzy_dupes = df.iloc[20:30].copy()
    fuzzy_dupes["First_Name"] = fuzzy_dupes["First_Name"].astype(str) + " "
    fuzzy_dupes["Annual_Spend_USD"] = fuzzy_dupes["Annual_Spend_USD"] + random.uniform(0.1, 1.0)
    df = pd.concat([df, fuzzy_dupes], ignore_index=True)
    
    # Shuffle dataframe
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df

def _generate_financial_dataset(rows: int) -> pd.DataFrame:
    data = []
    base_date = datetime(2025, 1, 1)
    types = ["WIRE", "wire", "ACH", "ach", "CREDIT", "DEBIT", "CRYPTO", "UNKNOWN"]
    currencies = ["USD", "usd", "EUR", "eur", "GBP", "CAD", "JPY"]
    
    for i in range(1, rows + 1):
        dt = base_date + timedelta(days=random.randint(0, 180))
        amt = random.choice([5000000.0, -100000.0]) if random.random() < 0.05 else round(random.uniform(10.0, 15000.0), 2)
        fee = round(amt * random.uniform(0.001, 0.02), 2) if amt > 0 else 0.0
        
        data.append({
            "Txn_ID": f"TXN-{20000 + i}",
            "Txn_Date": dt.strftime("%Y-%m-%d") if random.random() > 0.3 else dt.strftime("%d/%m/%Y"),
            "Sender_Account": f"ACC-{random.randint(100, 150)}",
            "Receiver_Account": f"ACC-{random.randint(200, 250)}",
            "Txn_Type": random.choice(types),
            "Amount": amt,
            "Fee": fee,
            "Currency": random.choice(currencies),
            "Risk_Score": round(random.uniform(0.0, 1.0), 3) if random.random() > 0.1 else np.nan,
            "Is_Flagged": random.choice(["TRUE", "true", "FALSE", "false", "Yes", "No", np.nan])
        })
    df = pd.DataFrame(data)
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)

def _generate_hr_dataset(rows: int) -> pd.DataFrame:
    data = []
    departments = ["Engineering", "engineering", "ENG", "Sales", "sales", "Marketing", "HR", "hr", "Finance", "Legal"]
    for i in range(1, rows + 1):
        salary = random.choice([2500000, 10000]) if random.random() < 0.04 else random.randint(45000, 180000)
        exp = random.choice([50, -2]) if random.random() < 0.03 else random.randint(0, 35)
        data.append({
            "Emp_ID": f"EMP-{500 + i}",
            "Full_Name": f"Employee {i}",
            "Department": random.choice(departments),
            "Job_Title": random.choice(["Software Engineer", "Sr. Developer", "Account Rep", "HR Specialist", "Manager", "Director"]),
            "Hire_Date": f"2018-0{random.randint(1,9)}-15",
            "Years_Experience": exp,
            "Salary_USD": salary,
            "Performance_Rating": random.choice([1, 2, 3, 4, 5, np.nan]),
            "Remote_Work": random.choice(["Yes", "YES", "yes", "No", "NO", "no", np.nan])
        })
    df = pd.DataFrame(data)
    df = pd.concat([df, df.iloc[:8]], ignore_index=True)
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)
