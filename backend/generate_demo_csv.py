from ml_pipeline import make_demo_dataset
make_demo_dataset().to_csv("demo_credit_fraud.csv", index=False)
print("Created demo_credit_fraud.csv (synthetic data; not a real-world benchmark).")
