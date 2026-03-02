from scratch.simple_linear_regression import *
import matplotlib.pyplot as plt
import csv
from evans_functions import *
from scratch.statistics import mean

#Opens csv and put x and y into lists and disgards row with a target value of 0
filename = "diabetes.csv"
rows = []

with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader) # Skips the header
    for row in csvreader:
        #Checks if the target value in the row is invalid
        if row and float(row[1]) != 0:
            rows.append(row)

x = []
y = [] 

for row in rows:
    x.append(float(row[5]))
    y.append(float(row[1]))

#replaces all bmi values listing 0 to the mean
x = impute_zeros_with_mean(x)

# Splitting the data in 80% training and 20% testing
split_index = int(0.8 * len(y))

x_train = x[:split_index]
y_train = y[:split_index]

x_test = x[split_index:]
y_test = y[split_index:]

#extract alpha and beta values from x and y
alpha, beta = least_squares_fit(x_train, y_train)

print(f"Model Equation: Glucose = {beta:.2f} * BMI + {alpha:.2f}")

#sum_of_sqerrors
sse = 0
for i in range(len(x_test)):
    actual = y_test[i]
    predicted = predict(alpha, beta, x_test[i])
    error = actual - predicted
    sse += error ** 2
    
print(f"\nsum_of_sqerrors() (SSE) on Test Data: {sse}")

#r_squared
y_bar = mean(y_test)

#calculating the SST
sst = 0
for val in y_test:
    squared_difference = (val - y_bar) ** 2
    sst += squared_difference

if sst == 0:
    r2_val = 1.0 #checking if it is zero
else:
    r2_val = 1 - (sse / sst) 
    
print(f"r_squared() (R2) on Test Data: {r2_val}")

#RMSE
rmse_val = rmse(sse, y_test) 
print(f"rmse() (RMSE) on Test Data: {rmse_val}")

#Plotting a scatter plot
plt.figure()
plt.scatter(x_test, y_test, color='blue', marker='x', alpha=0.6, label='Test Data')

#Using min/max to get a clean line of best-fit
x_min = min(x_test)
x_max = max(x_test)
y_min = predict(alpha, beta, x_min)
y_max = predict(alpha, beta, x_max)

plt.plot([x_min, x_max], [y_min, y_max], color='red', linewidth=2, label='Regression Line')

#Labels for Graph
plt.xlabel('BMI')
plt.ylabel('Glucose')
plt.title('Simple Linear Regression: Glucose vs BMI')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("slr_scatter_plot.png")
plt.show()



#histogram for BMI
plt.hist(x, bins=20, color='green', edgecolor='black')

#Labels for histogram
plt.title('Distribution of BMI (Feature)')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.5)
plt.savefig("histogram_bmi.png")
plt.show()




#histogram for glucose
plt.hist(y, bins=20, color='grey', edgecolor='black')

#Labels for hisogram
plt.title('Distribution of Glucose (Target)')
plt.xlabel('Glucose')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.5)
plt.savefig("histogram_glucose.png")
plt.show()