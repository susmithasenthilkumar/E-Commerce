
# E-COMMERCE DATA SCIENCE PROJECT


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
from scipy.stats import norm

np.random.seed(42)

print("\nEXP 1: Customer Behavior Correlation")

data = np.random.randint(1, 100, (50, 5))
features = ["Time_Spent", "Pages_Viewed", "Clicks", "Cart_Items", "Purchases"]

corr = np.corrcoef(data, rowvar=False)

plt.imshow(corr)
plt.colorbar()
plt.xticks(range(5), features)
plt.yticks(range(5), features)
plt.title("Customer Behavior Heatmap")
plt.show()

print("\nEXP 2: Purchase Prediction")

df = pd.DataFrame(np.random.rand(200, 5))
df["Purchased"] = np.random.randint(0, 2, 200)

X = df.iloc[:, :-1]
y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

print("\nEXP 3: Sales Prediction")

days = np.arange(1, 31)
sales = days * 500 + np.random.randint(-2000, 2000, 30)

model = LinearRegression()
model.fit(days.reshape(-1,1), sales)

plt.scatter(days, sales)
plt.plot(days, model.predict(days.reshape(-1,1)))
plt.title("Sales Trend")
plt.show()

print("\nEXP 4: Customer Sampling")

df = pd.DataFrame({
    "CustomerID": np.arange(1,101),
    "Category": np.random.choice(["Electronics","Clothing","Grocery"],100)
})

print(df.sample(frac=0.25))

print("\nEXP 5: Z-Test on Spending")

spending = np.random.normal(500, 50, 40)
z = (np.mean(spending) - 450) / (np.std(spending)/np.sqrt(40))

print("Z-score:", z)
print("Decision:", "Reject H0" if z > norm.ppf(0.95) else "Accept H0")

print("\nEXP 6: Sales Operations")

sales = np.random.randint(1000, 5000, 12)
print("Mean:", np.mean(sales))
print("Sum:", np.sum(sales))

print("\nEXP 7: Cleaning Data")

data = np.array([[100,200,np.nan],[300,400,500]])
data[np.isnan(data)] = np.nanmean(data)

print(data)

print("\nEXP 8: Financial Analysis")

returns = np.random.normal(0.002, 0.01, (100, 3))
print("Mean Returns:", returns.mean(axis=0))

print("\nEXP 9: Customer Analysis")

data = np.random.randint(1,100,(10,5))
print("Avg:", data.mean(axis=0))

plt.imshow(np.corrcoef(data, rowvar=False))
plt.colorbar()
plt.show()
print("\nEXP 10: Revenue Analysis")

df = pd.DataFrame({
    "Qty": np.random.randint(1,10,50),
    "Price": np.random.randint(100,500,50)
})

df["Revenue"] = df["Qty"] * df["Price"]

print("Total Revenue:", df["Revenue"].sum())

plt.plot(df["Revenue"])
plt.title("Revenue Trend")
plt.show()

print("\n✅ PROJECT COMPLETED SUCCESSFULLY")
