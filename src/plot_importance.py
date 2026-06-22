import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("xgboost_feature_importance_fixed_ura.csv")

top = df.head(15)

plt.figure(figsize=(10,6))
plt.barh(top["feature"], top["importance"])
plt.gca().invert_yaxis()
plt.title("Top 15 XGBoost Features")
plt.tight_layout()
plt.savefig("xgboost_importance.png", dpi=300)
plt.show()

