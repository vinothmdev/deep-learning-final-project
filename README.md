The goal of this project is to analyze the household electric power consumption data and build predictive models to forecast future electricity usage. The analysis will involve exploring the data, performing time series analysis, and implementing Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks to predict future power consumption trends. This project aims to provide insights into electricity usage patterns and develop robust models for accurate forecasting, which can be beneficial for efficient energy management and planning.

In addition to that experiment with the implementation of a morden algorith called NBeatX.  NBeatX is a deep learning model designed for multivariate time series forecasting. It's an extension of the N-BEATS model, which focuses on accurate time series prediction using a stack of fully connected layers.  This is to test how well NBeatX helps in this problem compared to RNN and LSTM.

The data for this project is taken from UC Irvine Machine LearningRepository. [Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

## Why I am choosing this problem

Throughout this course, we have explored various advanced topics in machine learning and artificial intelligence, including Neural Networks, Gradient Descent, RNN, CNN, GAN, Transformers, and NLP. As the course approaches its conclusion, I aim to focus on Time Series analysis—a critical and ubiquitous problem in the industry due to its sequential nature.

Time Series data presents unique challenges to deep learning models, often resulting in traditional statistical methods outperforming deep learning approaches. To delve into these challenges and explore potential solutions, I have chosen the household electric power consumption dataset from UC Irvine ML Repository. This dataset provides a real-world example for applying Time Series analysis principles and testing deep learning models.

By tackling this problem, I intend to gain deeper insights into the intricacies of Time Series data, enhance my understanding of how to effectively apply deep learning techniques in this context, and compare their performance against traditional statistical frameworks. This project not only consolidates my learning from the course but also equips me with practical skills highly relevant to industry applications.

## Project Plan:

1. Data Exploration and Preprocessing:
   - Load the dataset and inspect the structure and content.
   - Handle missing values, if any.
   - Convert the date and time information to a suitable datetime format.
   - Resample the data to a suitable frequency (e.g., hourly, daily) based on the analysis requirements.
   - Perform exploratory data analysis (EDA) to understand the distribution and patterns in the data.

2. Time Series Analysis:
   - Decompose the time series to identify trends, seasonality, and residual components.
   - Visualize the time series components to gain insights into the underlying patterns.
   - Perform stationarity tests (e.g., ADF test) and apply differencing if necessary to make the series stationary.

3. Modeling with RNN and LSTM:
   - Split the data into training and testing sets.
   - Normalize the data for better model performance.
   - Build and train RNN and LSTM models using the prepared features.
   - Tune the hyperparameters of the models for optimal performance.
   - Evaluate the models using appropriate metrics (e.g., RMSE, MAE) on the test set.

4. Model Evaluation and Comparison:
   - Compare the performance of RNN and LSTM models.
   - Analyze the forecasted results and compare them with actual values.
   - Visualize the predictions to assess the model's accuracy and reliability.

# Data Description

| **Column**               | **Description**                                                                                                          |
|--------------------------|--------------------------------------------------------------------------------------------------------------------------|
| Date                     | Date in format dd/mm/yyyy                                                                                                |
| Time                     | Time in format hh:mm:ss                                                                                                   |
| Global_active_power      | Household global minute-averaged active power (in kilowatt)                                                              |
| Global_reactive_power    | Household global minute-averaged reactive power (in kilowatt)                                                            |
| Voltage                  | Minute-averaged voltage (in volt)                                                                                        |
| Global_intensity         | Household global minute-averaged current intensity (in ampere)                                                           |
| Sub_metering_1           | Energy sub-metering No. 1 (in watt-hour of active energy). <br>It corresponds to the kitchen, containing mainly a dishwasher, an oven and a microwave (hot plates are not electric but gas powered). |
| Sub_metering_2           | Energy sub-metering No. 2 (in watt-hour of active energy).<br> It corresponds to the laundry room, containing a washing-machine, a tumble-drier, a refrigerator and a light. |
| Sub_metering_3           | Energy sub-metering No. 3 (in watt-hour of active energy).<br> It corresponds to an electric water-heater and an air-conditioner.                                |

# Conclusion

- Both the RNN and LSTM models show a good ability to capture the general patterns in the `Global Active Power` time series data.
- The RNN model achieves a slightly lower Mean Squared Error (MSE) compared to the LSTM model (0.008158 vs. 0.008340), indicating slightly better performance in this particular case.
- Validation loss plots suggest that while both models generalize reasonably well, the LSTM model exhibits more fluctuations, indicating potential overfitting or sensitivity to validation data.

# Future Enhancements

1. **Hyperparameter Tuning**:
   - Further refine hyperparameters using more extensive search methods such as Bayesian Optimization or Grid Search, in addition to Random Search.
   - Consider tuning additional hyperparameters such as dropout rates, batch size, number of layers, and different optimizers.

2. **Model Complexity**:
   - Experiment with deeper architectures by adding more layers to the RNN and LSTM models.
   - Consider using Bidirectional LSTM or GRU (Gated Recurrent Unit) layers, which might provide better performance by capturing dependencies in both directions.

3. **Feature Engineering**:
   - Introduce additional features that could improve the model's performance, such as lagged values, rolling statistics (mean, variance), or external variables like weather data if relevant.
   - Conduct feature selection to identify the most influential features and reduce dimensionality.

4. **Advanced Models**:
   - Experiment with more advanced architectures such as Transformer models, which have shown great success in capturing long-range dependencies in time series data.
   - Consider using hybrid models that combine the strengths of different neural network types, such as CNN-RNN or CNN-LSTM architectures.

5. **Seasonal Decomposition**:
   - Use seasonal decomposition methods to separately model trend, seasonality, and residuals, and then combine the predictions to improve accuracy.