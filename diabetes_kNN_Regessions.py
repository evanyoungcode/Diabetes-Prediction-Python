import csv
import matplotlib.pyplot as plt


from evans_functions import *
from scratch.linear_algebra import distance
from scratch.statistics import mean


#Opens csv and puts collums into lists and disgards row with a target value of 0
filename = "diabetes.csv"
rows = []


with open(filename, 'r') as csvfile:
   csvreader = csv.reader(csvfile)
   fields = next(csvreader) # Skips the header
   for row in csvreader:
       #Checks if the target value in the row is invalid
       if row and float(row[1]) != 0:
           rows.append(row)




pregnancies = []
glucose = [] #Target Value
bp = []
skin = []
insulin = []
bmi = []
pedigree = []
age = []




for row in rows:
   pregnancies.append(float(row[0]))
   glucose.append(float(row[1]))
   bp.append(float(row[2]))
   skin.append(float(row[3]))
   insulin.append(float(row[4]))
   bmi.append(float(row[5]))
   pedigree.append(float(row[6]))
   age.append(float(row[7]))




#replaces all feature values that are a 0 are replaces with the lists mean
bp = impute_zeros_with_mean(bp)
skin = impute_zeros_with_mean(skin)
insulin = impute_zeros_with_mean(insulin)
bmi = impute_zeros_with_mean(bmi)




#reformatting into list
patients = []
for i in range(len(glucose)):
   patient = [pregnancies[i], bp[i], skin[i], insulin[i], bmi[i], pedigree[i], age[i]]
   patients.append(patient)


#rescaling the list
x_rescaled = rescale(patients)


#splitting the data in 80% training and 20% testing
split_number = int(0.8 * len(glucose))


training_x = x_rescaled[:split_number]
training_y = glucose[:split_number]


training_data = []
for i in range(split_number):
   training_data.append([training_x[i], training_y[i]])


testing_data = x_rescaled[split_number:]
testing_results = glucose[split_number:]




#kNN loop setup for testing best k number
k_values_to_test = range(1, 31, 2)
rmse_results = []
r2_results = []
best_rmse = float('inf')
best_k = 0
best_predictions = []


#testing each k value for best k
for k in k_values_to_test:
   predicitions = []


   #looping through every test patient
   for new_patient in testing_data:
       distances = []


       #Comparing the new patient to all the patients in the training data
       for training_point in training_data:
           training_features = training_point[0]
           training_result = training_point[1]


           #calculating the distance
           dist = distance(training_features, new_patient)
           distances.append((dist, training_result))


       #sorting the distances
       distances.sort()


       #taking the top kN
       kN = distances[:k]


       #Getting the average glucose of the kN
       neighbor_values = []


       # Look for the k
       for neighbor_distance, neighbor_glucose in kN:
      
           neighbor_values.append(neighbor_glucose)
      
       # The prediction is the average of all the glucose scores.
       prediction = mean(neighbor_values)
       predicitions.append(prediction)




#Evaulating for the best K in the loop
   #Calculating the SSE (Sum of Squared Errors)
   sse = 0

   for i in range(len(testing_results)):
       actual = testing_results[i]
       predicted = predicitions[i]
      
       error = actual - predicted
       sse += error ** 2


   # Calculation of RMSE
   rmse_val = rmse(sse, testing_results)
  
   # Calculation of R2
   y_bar = mean(testing_results)


    #calculating the SST
   sst = 0

   for y in testing_results:
       squared_difference = (y - y_bar) ** 2
       sst += squared_difference

   if sst == 0:
       r2_val = 1.0 #checking if is equal to zero
   else:
       r2_val = 1 - (sse / sst) 
  
   rmse_results.append(rmse_val)
   r2_results.append(r2_val)


   print(f"k={k}: RMSE={rmse_val}, R2={r2_val}")


   # Tracking the best k value
   if rmse_val < best_rmse:
       best_rmse = rmse_val
       best_k = k
       best_r2 = r2_val
       best_predictions = predicitions[:] #saving the best prediction




#Plotting/Graphing

#Plotting the elbow method 
plt.figure()
plt.plot(list(k_values_to_test), rmse_results, label='RMSE vs K', color='green', linestyle='-', marker='o')

plt.grid(True)

plt.xlabel('K (Number of Neighbors)')
plt.ylabel('RMSE (Root Mean Squared Error)')
plt.title('K-Value Selection (Elbow Method)')

plt.savefig("Elbow_Method_Graph")
plt.legend()
plt.show()


#actual vs predicted scatter plot
plt.scatter(testing_results, best_predictions, color='blue', marker='x', label='Actual Data')
plt.plot([min(testing_results), max(testing_results)], [min(testing_results), max(testing_results)], color='red', linestyle='--', label='Perfect Prediction')

plt.xlabel('Actual Glucose')
plt.ylabel('Predicted Glucose')
plt.title(f'Actual vs Predicted Glucose (k={best_k})')
plt.legend()
plt.grid(True)
plt.savefig("actual_vs_predicted.png")
plt.show()



#Making Matrix graph
#Grouping the feature listing into single for matrix 
data_columns = [pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age]
column_names = ['Pregnancies', 'Glucose', 'BP', 'Skin', 'Insulin', 'BMI', 'Pedigree', 'Age']

#Calculating the correlation matrix
corr_matrix = correlation_matrix(data_columns)

#creating the heatmap
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')

#Adding a colour bar
plt.colorbar(label='Correlation Coefficient')

#Adding labels
indices = range(len(column_names))
plt.xticks(indices, column_names, rotation=45)
plt.yticks(indices, column_names)


plt.title("Feature Correlation Matrix")
plt.xlabel("Features")
plt.ylabel("Features")
plt.savefig("correlation_matrix.png")
plt.show()