# Interactive A/B Testing Simulation for Meta Quest Store

## Introduction

This Jupyter Notebook provides an interactive simulation of an A/B test for the Meta Quest Store. 
The scenario is based on a hypothetical decline in Average Revenue Per User (ARPU), and we are 
testing whether a new recommendation algorithm can help improve this metric.

**What you can do with this notebook:**

*   **Explore A/B testing concepts:** Learn about hypothesis testing, statistical significance, p-values, and power analysis in a practical context.
*   **Run simulations:** Generate synthetic data and perform A/B tests with different parameters.
*   **Adjust parameters:** Use interactive sliders and dropdowns to change:
    *   Sample size
    *   Baseline values for metrics (e.g., control group ARPU, conversion rate)
    *   Effect sizes (e.g., the lift in ARPU you expect from the new algorithm)
    *   Significance level (alpha)
    *   Statistical power
    *   The type of statistical test (t-test or Mann-Whitney U test)
    *   The metric being analyzed (e.g. ARPU, conversion rate, etc.)
*   **Visualize results:** See the impact of your parameter choices on the distributions of the metrics and the outcome of the statistical tests.
*   **Learn about the Meta Quest Store scenario:** Gain insights into the types of metrics and analyses that might be relevant for a VR app store.

**Scenario:**

The Meta Quest Store has seen a 7% decline in ARPU over the past quarter. We hypothesize that a new 
recommendation algorithm (treatment group) can improve user discovery of relevant games and apps, 
leading to increased purchases and a higher ARPU compared to the existing algorithm (control group).

**Key Metrics:**

*   **ARPU (Average Revenue Per User):** Total revenue divided by the number of users.
*   **Conversion Rate:** Percentage of users who make at least one purchase.
*   **Total Revenue:**  Total revenue generated.
*   **Total Purchases:** Total number of purchases made.
*   **Average Session Duration:** Average time spent per session.
*   **Average Daily Sessions:** Average number of sessions per day.
*   **Days Engaged:** Number of days a user is active.

**Statistical Tests:**

*   **T-test:** Used to compare the means of two groups when data is normally distributed.
*   **Mann-Whitney U Test:** A non-parametric test used to compare two groups when the data may not be normally distributed.

**Let's get started!**


```python
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.power as smp
from ipywidgets import interact, interactive, fixed, interact_manual, Layout, HBox, VBox, Label, Button, Output
import ipywidgets as widgets

```


```python

# --- Configuration (Defaults) ---
config = {
    "n_users": 50000,  # Total number of users
    "control_proportion": 0.5,  # Proportion of users in the control group
    "avg_arpu_control": 30,
    "std_dev_arpu_control": 5,
    "avg_arpu_treatment_lift": 2,  # Lift in ARPU for the treatment group
    "std_dev_arpu_treatment": 6,
    "avg_session_duration_pre": 30,  # in minutes
    "std_dev_session_duration_pre": 10,
    "avg_daily_sessions_pre": 2,
    "std_dev_daily_sessions_pre": 1,
    "avg_conversion_rate_control": 0.2, # 20%
    "avg_conversion_rate_treatment": 0.25, # 25%
    "avg_total_purchases_pre": 5,
    "std_dev_total_purchases_pre": 2,
    "avg_days_engaged":14,
    "std_dev_days_engaged": 5,
    "alpha": 0.05,
    "power": 0.8,
    "alternative": "greater",  # One-tailed test
    "test_type": "t-test",  # t-test or mann-whitney
}
```


```python
# --- Data Generation Function ---
def generate_synthetic_data(config):
    """
    Generates synthetic data for the Meta Quest Store A/B test scenario.

    This function creates a pandas DataFrame with user-level data, including both
    pre-experiment metrics and metrics that would be collected during the experiment.
    The data is generated based on the parameters specified in the `config` dictionary.

    Args:
        config (dict): A dictionary containing configuration parameters for the simulation.
                       See the "Configuration (Defaults)" section for details.

    Returns:
        pandas.DataFrame: A DataFrame with synthetic user data.
    """

    n_control = int(config["n_users"] * config["control_proportion"])
    n_treatment = config["n_users"] - n_control

    # Generate User Features
    user_ids = np.arange(config["n_users"])
    countries = np.random.choice(["US", "CA", "UK", "DE", "FR", "JP", "KR"], config["n_users"])
    age_groups = np.random.choice(["18-24", "25-34", "35-44", "45+"], config["n_users"])
    days_since_signup = np.random.randint(30, 365, config["n_users"])
    device_types = np.random.choice(["Quest 1", "Quest 2", "Quest Pro"], config["n_users"])

    # Generate Pre-Experiment Metrics
    avg_session_duration_pre = np.random.normal(
        config["avg_session_duration_pre"], config["std_dev_session_duration_pre"], config["n_users"]
    )
    avg_daily_sessions_pre = np.random.normal(
        config["avg_daily_sessions_pre"], config["std_dev_daily_sessions_pre"], config["n_users"]
    )
    total_purchases_pre = np.random.normal(
        config["avg_total_purchases_pre"], config["std_dev_total_purchases_pre"], config["n_users"]
    )
    total_purchases_pre = np.clip(total_purchases_pre, 0, None).astype(int) # Make sure no negative numbers
   # Assign Groups
    groups = ["control"] * n_control + ["treatment"] * n_treatment
    np.random.shuffle(groups)  # Shuffle to ensure random assignment

    # Generate Experiment Metrics
    arpu = np.where(
        np.array(groups) == "control",
        np.random.normal(config["avg_arpu_control"], config["std_dev_arpu_control"], config["n_users"]),
        np.random.normal(
            config["avg_arpu_control"] + config["avg_arpu_treatment_lift"],
            config["std_dev_arpu_treatment"],
            config["n_users"],
        ),
    )
    
    conversion_rate = np.where(
        np.array(groups) == "control",
        np.random.binomial(1, config["avg_conversion_rate_control"], config["n_users"]),
        np.random.binomial(1, config["avg_conversion_rate_treatment"], config["n_users"])
    )

    total_revenue = np.where(
        conversion_rate == 1,
        arpu, 0
    )

    total_purchases = np.where(
        np.array(groups) == "control",
        np.random.poisson(config["avg_total_purchases_pre"], config["n_users"]),
        np.random.poisson(config["avg_total_purchases_pre"] + config["avg_arpu_treatment_lift"]/5, config["n_users"]) #+ treatment_effect_purchases, config["n_users"])
    )

    avg_session_duration_post = np.where(
        np.array(groups) == "control",
        np.random.normal(config["avg_session_duration_pre"], config["std_dev_session_duration_pre"], config["n_users"]),
        np.random.normal(
            config["avg_session_duration_pre"] + config["avg_arpu_treatment_lift"],
            config["std_dev_session_duration_pre"],
            config["n_users"],
        ),
    )

    avg_daily_sessions_post = np.where(
        np.array(groups) == "control",
        np.random.normal(config["avg_daily_sessions_pre"], config["std_dev_daily_sessions_pre"], config["n_users"]),
        np.random.normal(
            config["avg_daily_sessions_pre"] + config["avg_arpu_treatment_lift"]/10,
            config["std_dev_daily_sessions_pre"],
            config["n_users"],
        ),
    )
    
    days_engaged = np.random.normal(
        config["avg_days_engaged"], config["std_dev_days_engaged"], config["n_users"]
    )
    days_engaged = np.clip(days_engaged, 0, config["avg_days_engaged"]*2).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame(
        {
            "user_id": user_ids,
            "country": countries,
            "age_group": age_groups,
            "days_since_signup": days_since_signup,
            "device_type": device_types,
            "avg_session_duration_pre": avg_session_duration_pre,
            "avg_daily_sessions_pre": avg_daily_sessions_pre,
            "total_purchases_pre": total_purchases_pre,
            "group": groups,
            "recommendation_algorithm_used": np.where(np.array(groups) == "control", "default", "new"),
            "date":  pd.to_datetime('today').strftime('%Y-%m-%d'),
            "arpu": arpu,
            "conversion_rate": conversion_rate,
            "total_revenue": total_revenue,
            "total_purchases": total_purchases,
            "avg_session_duration_post": avg_session_duration_post,
            "avg_daily_sessions_post": avg_daily_sessions_post,
            "days_engaged": days_engaged
        }
    )
    
    # Ensuring no negative values for relevant metrics
    for col in ["arpu", "total_revenue", "total_purchases", "avg_session_duration_post", "avg_daily_sessions_post", "days_engaged"]:
        df[col] = df[col].apply(lambda x: max(0, x))

    return df

```


```python
 
# --- Helper Functions ---
def calculate_sample_size(config):
    """
    Calculates the sample size needed for each group based on desired power and effect size.

    Args:
        config (dict): A dictionary containing configuration parameters.

    Returns:
        int: The calculated sample size required for each group.
    """
    effect_size = config["avg_arpu_treatment_lift"] / config["std_dev_arpu_control"]
    
    nobs = smp.tt_ind_solve_power(
        effect_size=effect_size,
        alpha=config["alpha"],
        power=config["power"],
        ratio=1, # Assuming equal sample sizes for control and treatment
        alternative=config["alternative"],
    )
    return int(nobs)
```


```python
# --- Visualization Function ---
def visualize_data(df, group_col, metric_col):
    """
    Creates visualizations (histogram and box plot) to compare the distributions of the two groups.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        group_col (str): The name of the column representing the groups ('control', 'treatment').
        metric_col (str): The name of the column representing the metric being analyzed.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x=metric_col, hue=group_col, kde=True, element="step")
    plt.title(f"Distribution of {metric_col} by Group")
    plt.xlabel(metric_col)
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.boxplot(x=group_col, y=metric_col, data=df)
    plt.title(f"Box Plot of {metric_col} by Group")
    plt.xlabel("Group")
    plt.ylabel(metric_col)
    plt.show()
```


```python
# --- Statistical Test Functions ---
def perform_t_test(df, group_col, metric_col, alpha, alternative="two-sided"):
    """
    Performs an independent two-sample t-test.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        group_col (str): The name of the column representing the groups ('control', 'treatment').
        metric_col (str): The name of the column representing the metric being analyzed.
        alpha (float): The significance level (e.g., 0.05).
        alternative (str): The alternative hypothesis ('two-sided', 'greater', or 'less').

    Returns:
        tuple: The t-statistic, the p-value, and a conclusion string.
    """
    control_data = df[df[group_col] == "control"][metric_col]
    treatment_data = df[df[group_col] == "treatment"][metric_col]

    t_statistic, p_value = stats.ttest_ind(
        treatment_data, control_data, equal_var=False, alternative=alternative
    )

    print(f"T-statistic: {t_statistic:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value <= alpha:
        conclusion = f"Reject the null hypothesis. There is statistically significant evidence of a difference in {metric_col} between the two groups."
    else:
        conclusion = f"Fail to reject the null hypothesis. There is not enough statistically significant evidence of a difference in {metric_col} between the two groups."

    return t_statistic, p_value, conclusion

def perform_mann_whitney_u_test(df, group_col, metric_col, alpha, alternative="two-sided"):
    """
    Performs a Mann-Whitney U test (non-parametric alternative to the t-test).

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        group_col (str): The name of the column representing the groups ('control', 'treatment').
        metric_col (str): The name of the column representing the metric being analyzed.
        alpha (float): The significance level (e.g., 0.05).
        alternative (str): The alternative hypothesis ('two-sided', 'greater', or 'less').

    Returns:
        tuple: The U-statistic, the p-value, and a conclusion string.
    """
    control_data = df[df[group_col] == "control"][metric_col]
    treatment_data = df[df[group_col] == "treatment"][metric_col]

    u_statistic, p_value = stats.mannwhitneyu(
        treatment_data, control_data, alternative=alternative
    )

    print(f"U-statistic: {u_statistic:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value <= alpha:
        conclusion = f"Reject the null hypothesis. There is statistically significant evidence of a difference in the distribution of {metric_col} between the two groups."
    else:
        conclusion = f"Fail to reject the null hypothesis. There is not enough statistically significant evidence of a difference in the distribution of {metric_col} between the two groups."

    return u_statistic, p_value, conclusion

# --- Main Simulation Function ---
def run_simulation(config):
    """
    Runs the A/B testing simulation based on the provided configuration.

    Args:
        config (dict): A dictionary containing configuration parameters for the simulation.
    """
    print("Generating synthetic data...")
    df = generate_synthetic_data(config)

    print("\nSample Data:")
    print(df.head())

    # Calculate and display required sample size
    required_sample_size = calculate_sample_size(config)
    print(f"\nRequired sample size per group: {required_sample_size}")

    # Proceed with the simulation using the generated samples
    print(f"\nRunning simulation with {config['test_type']}...")
    if config["test_type"] == "t-test":
        test_func = perform_t_test
    elif config["test_type"] == "mann-whitney":
        test_func = perform_mann_whitney_u_test
    else:
        raise ValueError("Invalid test_type. Choose 't-test' or 'mann-whitney'.")
    
    metric_col = config["metric_col"]
    if metric_col == 'conversion_rate':
        df['conversion_rate'] = df['conversion_rate'].astype(float)

    visualize_data(df, "group", metric_col)
    _, p_value, conclusion = test_func(
        df, "group", metric_col, config["alpha"], config["alternative"]
    )
    print(f"\nTest Results (Metric: {metric_col}):")
    print(conclusion)


```


```python
# --- Interactive Widget Interface ---
def interactive_simulation():
    """
    Creates an interactive interface for the A/B testing simulation using ipywidgets.

    This function sets up sliders and dropdowns for each parameter in the `config` dictionary,
    allowing users to dynamically adjust the parameters and see the results of the simulation
    in real-time.
    """
    style = {'description_width': 'initial'}

    # Create widgets for each parameter
    n_users_slider = widgets.IntSlider(value=config["n_users"], min=1000, max=100000, step=1000, description="Total Users:", style=style)
    control_proportion_slider = widgets.FloatSlider(value=config["control_proportion"], min=0.1, max=0.9, step=0.1, description="Control Proportion:", style=style)
    avg_arpu_control_slider = widgets.FloatSlider(value=config["avg_arpu_control"], min=10, max=100, step=1, description="Avg ARPU (Control):", style=style)
    std_dev_arpu_control_slider = widgets.FloatSlider(value=config["std_dev_arpu_control"], min=1, max=20, step=1, description="Std Dev ARPU (Control):", style=style)
    avg_arpu_treatment_lift_slider = widgets.FloatSlider(value=config["avg_arpu_treatment_lift"], min=0, max=10, step=0.5, description="Avg ARPU Lift (Treatment):", style=style)
    std_dev_arpu_treatment_slider = widgets.FloatSlider(value=config["std_dev_arpu_treatment"], min=1, max=20, step=1, description="Std Dev ARPU (Treatment):", style=style)
    avg_conversion_rate_control_slider = widgets.FloatSlider(value=config["avg_conversion_rate_control"], min=0, max=1, step=0.05, description="Avg Conversion Rate (Control):", style=style)
    avg_conversion_rate_treatment_slider = widgets.FloatSlider(value=config["avg_conversion_rate_treatment"], min=0, max=1, step=0.05, description="Avg Conversion Rate (Treatment):", style=style)
    alpha_slider = widgets.FloatSlider(value=config["alpha"], min=0.01, max=0.1, step=0.01, description="Alpha:", style=style)
    power_slider = widgets.FloatSlider(value=config["power"], min=0.5, max=0.95, step=0.05, description="Power:", style=style)
    alternative_dropdown = widgets.Dropdown(options=["two-sided", "greater", "less"], value=config["alternative"], description="Alternative:", style=style)
    test_type_dropdown = widgets.Dropdown(options=["t-test", "mann-whitney"], value=config["test_type"], description="Test Type:", style=style)
    metric_col_dropdown = widgets.Dropdown(options=["arpu", "conversion_rate", "total_revenue", "total_purchases","avg_session_duration_post", "avg_daily_sessions_post", "days_engaged"], value="arpu", description="Metric:", style=style)

    # Button to trigger simulation
    simulation_runner = widgets.Button(description="Run Simulation", button_style='success')
    output = widgets.Output()

    # Function to update config and run simulation
    def update_and_run(*args):
        with output:
            output.clear_output()
            config.update({
                "n_users": n_users_slider.value,
                "control_proportion": control_proportion_slider.value,
                "avg_arpu_control": avg_arpu_control_slider.value,
                "std_dev_arpu_control": std_dev_arpu_control_slider.value,
                "avg_arpu_treatment_lift": avg_arpu_treatment_lift_slider.value,
                "std_dev_arpu_treatment": std_dev_arpu_treatment_slider.value,
                "avg_conversion_rate_control": avg_conversion_rate_control_slider.value,
                "avg_conversion_rate_treatment": avg_conversion_rate_treatment_slider.value,
                "alpha": alpha_slider.value,
                "power": power_slider.value,
                "alternative": alternative_dropdown.value,
                "test_type": test_type_dropdown.value,
                "metric_col": metric_col_dropdown.value
            })
            run_simulation(config)

    # Attach the function to button click event
    simulation_runner.on_click(update_and_run)

    # Arrange widgets using VBox and HBox for better layout
    sliders_box = VBox([
        n_users_slider,
        control_proportion_slider,
        avg_arpu_control_slider,
        std_dev_arpu_control_slider,
        avg_arpu_treatment_lift_slider,
        std_dev_arpu_treatment_slider,
        avg_conversion_rate_control_slider,
        avg_conversion_rate_treatment_slider,
        alpha_slider,
        power_slider
    ])

    dropdowns_box = VBox([
        alternative_dropdown,
        test_type_dropdown,
        metric_col_dropdown
    ])

    controls_box = HBox([
        sliders_box,
        dropdowns_box
    ])

    # Display the widgets and output
    display(VBox([controls_box, simulation_runner, output]))

# --- Run Interactive Simulation ---
interactive_simulation()
```
