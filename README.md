🛒 E-Commerce Customer Behavior Analysis & Sales Prediction
📌 Project Overview

This project analyzes customer behavior in an e-commerce system using Python.
It helps to understand how customers interact with products and predicts future sales and purchase patterns.

⚙️ Technologies Used

Python 🐍
Pandas
NumPy
Matplotlib
Scikit-learn

📂 Dataset

The dataset simulates real-world e-commerce data.

Columns include:
Customer_ID
Product_Category
Product_Price
Quantity
Purchase_Status
Date

🔍 Features & Analysis
✅ Data Preprocessing
Handles missing values
Cleans and formats data
Converts date into proper format
✅ Revenue Calculation

Revenue is calculated using:
Revenue = Quantity × Product Price

✅ Customer Behavior Analysis
Analyzes customer activity
Identifies popular product categories
Studies purchasing patterns
✅ Correlation Analysis

Finds relationship between:

Product Price
Quantity
Revenue
✅ Sales Prediction

📌 Logistic Regression

Predicts whether a customer will make a purchase

📌 Linear Regression

Predicts future sales trends
✅ Statistical Testing
Uses Z-test to analyze customer spending behavior
Validates assumptions using statistics
✅ Customer Segmentation
Groups customers based on behavior
Helps in targeted marketing
📊 Visualizations

The project generates the following charts:

📌 1. Bar Chart – Category Sales
Shows revenue for each product category

📌 2. Line Chart – Sales Trend
Displays sales over time

📌 3. Scatter Plot – Price vs Quantity
Shows relationship between price and purchase

📌 4. Heatmap – Correlation Matrix
Shows relationships between variables

▶️ How to Run
1. Install Dependencies
pip install pandas numpy matplotlib scikit-learn
2. Load Dataset
df = pd.read_csv("your_file_path.csv")
3. Run the Script
python your_script_name.py
📈 Output
Customer behavior insights
Sales prediction results
Product performance analysis
Visual charts for understanding data
🚀 Future Enhancements
Add interactive dashboard (Streamlit / Plotly)
Build web application (Flask)
Real-time data integration
Advanced AI-based recommendation system
