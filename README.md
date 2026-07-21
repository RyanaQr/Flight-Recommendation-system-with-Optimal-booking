# ✈️ Optimal Flight Booking Prediction

A big data analytics project that leverages **Apache PySpark**, **machine learning**, and **time-series forecasting** to identify the optimal window for booking flights at the lowest price.

---

## 📌 Overview

Airline ticket pricing is dynamic and complex — fares fluctuate based on departure day, booking lead time, seat availability, and route. This project processes a large-scale flight itinerary dataset to:

- Analyze fare markups and pricing patterns across routes and days
- Predict the optimal number of days in advance to book a flight
- Forecast future prices using Holt-Winters and Facebook Prophet
- Score and rank booking recommendations per route using a composite model
- Identify deal fares using statistical thresholds (route mean ± std dev)

---

## 🗂️ Dataset

- **Source:** `Itineraries.csv` (~1M+ records)
- **Key fields used:**

| Column | Description |
|---|---|
| `flightDate` | Scheduled date of the flight |
| `searchDate` | Date the fare was searched/observed |
| `startingAirport` | Origin airport code |
| `destinationAirport` | Destination airport code |
| `baseFare` | Base ticket price (USD) |
| `totalFare` | Total price including fees (USD) |
| `seatsRemaining` | Remaining seat count at time of search |
| `isBasicEconomy` | Boolean flag for basic economy class |
| `isRefundable` | Boolean flag for refundable tickets |
| `segmentsAirlineCode` | Airline code(s) for each flight segment |

> A stratified sample of 200 records was used for exploratory analysis before scaling up to full-dataset Spark aggregations.
> A random 100 records was pushed into this repository to make it understandable.
---

## 🔧 Tech Stack

| Category | Tools |
|---|---|
| **Big Data Processing** | Apache PySpark, SparkSQL, Window Functions |
| **Data Manipulation** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (Linear Regression, Random Forest), XGBoost |
| **Time-Series Forecasting** | Statsmodels (Holt-Winters / Exponential Smoothing), Facebook Prophet |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Google Colab, Google Drive |

---

## 🧱 Project Structure

```
Optimal_flight_booking_prediction.ipynb
│
├── 1. Setup & Data Ingestion
│   ├── Mount Google Drive
│   ├── Initialize SparkSession
│   └── Load Itineraries.csv into Spark DataFrame
│
├── 2. Exploratory Analysis
│   ├── Average Price Analysis (base fare, total fare, markup %)
│   ├── Grouped & sorted fare data by route and date
│   └── Weekday vs. Weekend fare comparison
│
├── 3. Optimal Booking Window
│   ├── Compute days_before_departure (flightDate − searchDate)
│   ├── Scatter plot: Fare vs. Days Before Departure
│   └── Booking window bucketing (0–7, 8–14, ..., 90+ days)
│
├── 4. Predictive Modeling
│   ├── Linear Regression: fare → days before departure
│   ├── Holt-Winters Exponential Smoothing forecast
│   └── Facebook Prophet (yearly + weekly seasonality, 90-day horizon)
│
├── 5. Airline & Route Intelligence
│   ├── Airline fare comparison (avg fare, markup, flight count)
│   ├── Best day-of-week to fly vs. best day to book
│   └── Route-specific forecasting loop (per origin–destination pair)
│
├── 6. Advanced Features
│   ├── Random Forest feature importance analysis
│   ├── Price alert / deal detection (below route mean − 1 std dev)
│   ├── Seat urgency tier classification (Critical / High / Medium / Low)
│   └── Composite booking recommender function (price + refundability + class)
│
└── 7. Scalability Layer
    └── Full-dataset PySpark aggregations (no early .toPandas() calls)
```

---

## 🔍 Key Analyses & Results

### 📊 Fare Markup Analysis
Computed average base fare, total fare, markup (absolute and percentage) across the dataset to understand airline pricing behavior.

### 📅 Booking Window Optimization
Derived `days_before_departure` for each record and bucketed into meaningful intervals. Identified which booking windows (e.g., 22–45 days) correlate with lower average fares.

### 📈 Price Forecasting
- **Holt-Winters:** Applied additive exponential smoothing on sorted fare series to forecast prices over the next 30 days.
- **Prophet:** Modeled fare trends with yearly and weekly seasonality components, forecasting 90 days ahead with a configurable changepoint sensitivity.

### 🤖 ML Feature Importance (Random Forest)
Trained a Random Forest Regressor to predict `totalFare` using features:
- `AvgAgg_BaseFare`, `seatsRemaining`, `isBasicEconomy`, `isRefundable`, `day_of_week_num`, `days_before_departure_int`

Feature importances reveal the strongest price drivers.

### 🏷️ Deal Detection
Flagged fares falling below `route_avg − route_std` using PySpark Window functions, enabling automatic identification of below-market deals per route.

### 🎯 Composite Booking Recommender
```python
def recommend_booking(origin, dest, budget, days_available, pandas_df):
    """Returns top 3 booking recommendations for a route."""
    # Scores flights on: price (50%), refundability (30%), cabin class (20%)
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pyspark pandas numpy matplotlib seaborn scikit-learn statsmodels prophet xgboost
```

### Running the Notebook
1. Upload `Itineraries.csv` to your Google Drive
2. Open `Optimal_flight_booking_prediction.ipynb` in Google Colab
3. Update `GOOGLE_DRIVE_PATH_AFTER_MYDRIVE` to your Drive path
4. Run all cells sequentially

---

## 📌 Future Improvements

- [ ] Expand to the full 1M+ record dataset using distributed Spark jobs
- [ ] Deploy Prophet models per route for personalized forecasting
- [ ] Build a real-time fare alert system using Kafka + streaming predictions
- [ ] Add a Streamlit or Dash web UI for interactive booking recommendations
- [ ] Integrate external signals (holidays, events, fuel prices) as features

---

## 👤 Author

**RyanaQuadir**
Data Scientist
Python | Big Data | AI/ML Systems  

---

## 📄 License

This project is for a project purpose.
