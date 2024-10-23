import pandas as pd
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Title of the app
st.title("Medicine Demand Forecasting")

# Load datasets
df = pd.read_csv("1medicine_prices.csv")
df1 = pd.read_csv("1medicine_prices.csv")

# Convert 'date' to datetime
df["date"] = pd.to_datetime(df["date"], dayfirst=True)


# Feature engineering function
def create_date_features(df):
    df["month"] = df.date.dt.month
    df["day_of_month"] = df.date.dt.day
    df["day_of_year"] = df.date.dt.dayofyear
    df["day_of_week"] = df.date.dt.dayofweek
    df["week_of_year"] = df.date.dt.isocalendar().week
    df["year"] = df.date.dt.year
    df["is_weekend"] = df.date.dt.weekday > 5
    df["is_month_beginning"] = df.date.dt.is_month_start.astype(int)
    df["is_month_last"] = df.date.dt.is_month_end.astype(int)
    return df


# Apply feature engineering
df = create_date_features(df)

# Get dummies for categorical features
df = pd.get_dummies(df, columns=["medicine_name", "day_of_week", "month"])

# Log transformation of target variable
df["quantity_sold"] = np.log1p(df["quantity_sold"].values)

# Train-test split
train = df[:1323]
test = df[1323:]

# Select features and target variables
cols = [
    col
    for col in train.columns
    if col not in ["date", "medicine_name", "quantity_sold", "year"]
]
X_train, Y_train = train[cols], train["quantity_sold"]
X_test, Y_test = test[cols], test["quantity_sold"]

# LightGBM parameters
lgb_params = {
    "num_leaves": 10,
    "learning_rate": 0.02,
    "feature_fraction": 0.8,
    "max_depth": 5,
    "verbose": 0,
    "num_boost_round": 1000,
    "early_stopping_rounds": 200,
    "nthread": -1,
}

# LightGBM Dataset and model training
lgbtrain = lgb.Dataset(X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(X_test, label=Y_test, reference=lgbtrain, feature_name=cols)
model = lgb.train(
    lgb_params,
    lgbtrain,
    valid_sets=[lgbtrain, lgbval],
    callbacks=[lgb.early_stopping(lgb_params["early_stopping_rounds"])],
)

# Generate future dates
future_dates = pd.date_range(start="2022-01-01", end="2024-10-31", freq="D")
future_df = pd.DataFrame(future_dates, columns=["date"])
future_df = create_date_features(future_df)

# Extract top selling products from df1
top_medicines = (
    df1.groupby("medicine_name")["quantity_sold"].sum().nlargest(5).index.tolist()
)

# Sidebar for selecting medicine
st.sidebar.title("Medicine Selection")
selected_medicine = st.sidebar.selectbox("Choose a Medicine to Forecast", top_medicines)

# Dropdown for selecting any medicine from the dataset
all_medicines = df1["medicine_name"].unique().tolist()
selected_medicine_custom = st.sidebar.selectbox(
    "Choose a Medicine (Custom)", all_medicines
)


# Function to align future_df with training features
def align_features(df, reference_columns):
    for col in reference_columns:
        if col not in df.columns:
            df[col] = 0
    return df[reference_columns]


# Function to forecast for selected medicine
def forecast_medicine(medicine_name):
    future_df_copy = future_df.copy()

    # Ensure all medicine_name columns exist, even if they are not in the future_df
    # Add columns if missing and set them to 0 by default
    medicine_cols = [col for col in df.columns if col.startswith("medicine_name_")]
    for col in medicine_cols:
        if col not in future_df_copy.columns:
            future_df_copy[col] = 0

    # Set the selected medicine column to 1, others to 0
    future_df_copy[f"medicine_name_{medicine_name}"] = 1

    # Align features and make predictions
    future_df_aligned = align_features(future_df_copy, X_train.columns)
    predictions = model.predict(future_df_aligned) + np.random.normal(
        0, 0.05, size=future_df_aligned.shape[0]
    )

    # Create forecast DataFrame and plot
    forecast_df = pd.DataFrame(
        {"date": future_dates, "predicted_quantity": predictions}
    )

    # Apply a moving average to smooth the predicted_quantity
    window_size = 5  # You can adjust this to smooth more or less
    forecast_df['smoothed_quantity'] = forecast_df['predicted_quantity'].rolling(window=window_size).mean()

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    # Original plot with smoothing applied
    ax.plot(
        forecast_df["date"],
        forecast_df["smoothed_quantity"],
        marker="o",
        linestyle="-",
        color="blue",
        label="Smoothed Prediction"
    )

    # Optional: add the original line for comparison
    ax.plot(
        forecast_df["date"],
        forecast_df["predicted_quantity"],
        linestyle="--",
        color="gray",
        alpha=0.5,
        label="Original Prediction"
    )

    # Customize the plot
    ax.set_title(f"Smoothed Future Demand Forecast for {medicine_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Quantity Sold")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    # Add a legend to differentiate between smoothed and original
    ax.legend()

    # Show plot in Streamlit
    st.pyplot(fig)



# Forecast for the selected medicine
if selected_medicine_custom:
    st.subheader(f"Forecasting for: {selected_medicine_custom}")
    forecast_medicine(selected_medicine_custom)
elif selected_medicine:
    st.subheader(f"Forecasting for: {selected_medicine}")
    forecast_medicine(selected_medicine)

# Forecast for all top 5 medicines
st.subheader("Forecasting for All Top 5 Medicines")
for medicine in top_medicines:
    forecast_medicine(medicine)
