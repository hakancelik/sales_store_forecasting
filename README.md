Here is an English version of the `README.md` file for your GitHub project:

---

# Store Sales Forecasting

[![GitHub repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/hakancelik/sales_store_forecasting)

## Project Overview

This project aims to analyze and forecast store sales and profitability trends based on historical data. By leveraging various machine learning models and data analysis techniques, the project provides insights into how factors such as sales, quantity, and discounts affect profitability. The ultimate goal is to build a predictive model that accurately forecasts future profits.

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling](#modeling)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributors](#contributors)
- [License](#license)

## Data

The dataset used for this project includes various features such as:

- **Sales**: Total sales of the store.
- **Quantity**: Number of units sold.
- **Discount**: Discount applied during the sales transaction.
- **Profit**: Profit made from each transaction.
- **Order Date**: The date when the order was placed.
- **Ship Date**: The date when the order was shipped.

## Data Preprocessing

Several preprocessing steps were applied to clean and prepare the data:

- Handling missing values.
- Converting date columns (`Order Date` and `Ship Date`) into `datetime` format.
- Calculating additional features such as **Shipping Duration** (the number of days between the order and shipment).
- Categorizing key features like **Ship Mode**, **Segment**, **Sub-Category**, and **Region**.

## Exploratory Data Analysis

Key insights and visualizations were derived through:

- Distribution plots for numerical features (e.g., **Sales**, **Profit**).
- Bar plots to show the impact of **Region**, **Segment**, and **Category** on sales and profit.
- Heatmaps to illustrate the correlation between numerical features.
- Trend analysis of monthly sales and profit.


## Modeling

A linear regression model was implemented to predict profitability based on sales, quantity, and discounts. The model was trained and tested using a portion of the dataset:

- **Model**: Linear Regression
- **Training-Test Split**: 80-20 ratio

The following metrics were used to evaluate model performance:

- **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted values.
- **R^2 Score**: The proportion of variance in the dependent variable that is predictable from the independent variables.

## Results

| Metric                  | Value           |
|-------------------------|-----------------|
| **Mean Squared Error**  | 12780.86223084061     |
| **R^2 Score**           | 0.28550818881264695   |


Visualizations of actual vs. predicted profits provide insight into model accuracy.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hakancelik/sales_store_forecasting.git
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main analysis script:

```bash
python main.py
```

The script will perform data loading, preprocessing, and model training, and display key visualizations.

## Contributors

- Hakan Çelik - [GitHub Profile](https://github.com/hakancelik)

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---

This provides a clear overview of your project and includes sections for installation and usage instructions. Let me know if you'd like any further customizations!
