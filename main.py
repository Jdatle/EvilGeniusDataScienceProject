# John Le Evil Geniuses x Genius League: Data Scientist Assessment
# 5/22/2023

from sklearn.metrics import accuracy_score
from fpdf import FPDF
from johnMadeFunctions import FileToList, RemoveColumn, IdMissingDataColumns, ConvertToFloats, ConvertColumnToInt, \
    PartitionTestTrainData, NormalizeData, KMeansClustering, BIRCHClustering, DecisionTreeClassifying, KeepColumn, \
    PlotPieDistribution

# Incoming File
fileData = "starcraft_player_data.csv"

# Outgoig Files
outputFile = "main.py_output.pdf"
outputFigure = "Figure1.pdf"

# PDF Set up
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial","",12)


print("")
print("Hello, Welcome to John Le's Data Science Assessment Project for Evil Genius\n\n")

print("STEP 1: DATA EXTRACTION\n")

###################################################################
#                   STEP 1: DATA EXTRACTION
#       -Import Data
#       -Remove unwanted columns
#       -Convert data to floats
#       -Convert labels to int
#       -Create Training & Testing Data
#       -Normalize Data
###################################################################

# Import Data
dataRaw = FileToList(fileData)

# Remove GamerID column
dataNoGamerID = RemoveColumn(0,dataRaw)

# Identify columns with "?" (missing data)
missingDataColumns = IdMissingDataColumns(dataNoGamerID)

# Remove columns with missing data 
missingDataColumns.sort(reverse=True)       # Sort in descending order
dataList = dataNoGamerID
for i in missingDataColumns:
    dataList = RemoveColumn(i,dataList)

# Convert data to Floats
dataListFloat = ConvertToFloats(dataList)

# Convert label to Int
dataListReady = ConvertColumnToInt(dataListFloat,0)

# Partition Test & Training Data (Test data is every "n"th row)
n = 5 
dataTrain, dataTest = PartitionTestTrainData(dataListReady,n)

# Normalize Data & split out labels
normalizedDataTrain, trainLabels = NormalizeData(dataTrain)
normalizedDataTest, testLabels = NormalizeData(dataTest)

print("The data from", fileData, "is partitioned into training & testing data.")
print("Every", str(n)+"th", "entry is considered test data, the rest is training data.\n")
print("Data is reformated & normalized.\n\n")

print("STEP 2: Determine which attributes/features to use\n")

###################################################################
#                   STEP 2: Determine which attributes/features to use
#       -Generate/Test Kmeans Clustering Model (using all attributes)
#       -Generate/Test BIRCH Clustering Model (using all attributes)
#       -Generate/Test Decision Tree Model (using all attributes)
#       -Repeat process but for a single attribute
#       -Of those single attributes pick the ones that contribute the most to predicting the most accurate label
#       -Repeat classification method but with those attributes only
###################################################################

# Lets first try to use all the attribute and see which method provides the most accurate result:

accKMeansClustering, predictedKmeans = KMeansClustering(25,trainLabels,normalizedDataTrain,testLabels,normalizedDataTest)
accBIRCHClustering, predictedBIRCH = BIRCHClustering(14,trainLabels,normalizedDataTrain,testLabels,normalizedDataTest)
accDecisionTree, predictedDecisionTree = DecisionTreeClassifying(trainLabels,normalizedDataTrain,testLabels,normalizedDataTest)

print("Using all attributes, the training data is used to build 3 different models (KMeans Clustering, BIRCH Clustering, Decision Tree Classification).")
print("The testing data is used to calculate the accuracy of each model. Results can be seen below.")

print("KMeans Accuracy:", accKMeansClustering)
print("BIRCH Accuracy:", accBIRCHClustering)
print("Decision Tree Accuracy:", accDecisionTree)
print("")

# The above results, yield accuracy scores of 35% or less. Lets try to determine if some attributes are more imporantant that others.
# Use KMeans Clustering, BIRCH Clustering, & Decision Tree Classfication to see which single attribute can help predict rank.

print("The above process is repeated for each attribute individually")

# Create an accuracy dictionary where the key is "modelType:column" and the value is "prediction accuracy"
dictAcc = {}

for i in range(0,3,1):
    name = ""
    for j in range(0,len(normalizedDataTrain[0]),1):
        newDataTrain = KeepColumn([j],normalizedDataTrain)
        newDataTest = KeepColumn([j],normalizedDataTest)
        if i == 0:
            name = "kMeans: "
            acc, predict = KMeansClustering(11,trainLabels,newDataTrain,testLabels,newDataTest)
            key = name + str(j)
            dictAcc[key] = acc
        elif i == 1:
            name = "BIRCH: "
            acc, predict = BIRCHClustering(4,trainLabels,newDataTrain,testLabels,newDataTest)
            key = name + str(j)
            dictAcc[key] = acc
        elif i == 2:
            name = "DT: "
            acc, predict = DecisionTreeClassifying(trainLabels,newDataTrain,testLabels,newDataTest)
            key = name + str(j)
            dictAcc[key] = acc

# Sort by prediction accuracy value
sortedDictAcc = sorted(dictAcc.items(), key = lambda kv:kv[1])

print("The columns (attributes) with the highest accuracy is seen below.")
for i in range(-1,-8,-1):
    print(sortedDictAcc[i])


# Based on the above results, it we can see columns 0, 6, 8 individually are the best at predicting labels.
print("We can deduce that columns 0, 6, 8 consistently have the highest accuracy.\n\n")


# Lets combine them and see if it improves our accuracy overall.

###################################################################
#                   STEP 3: Create Model & Analyze Results
#       -Generate model with the highest accuracy.
#       -Review dataset
#       -Search for alternative accuracy measurement
###################################################################

print("STEP 3: Create Model & Analyze Results\n")
print("Models using KMeans Clustering, BIRCH Clustering, & Decision Tree Classification are rebuilt using the selected columns(attributes) only.")
print("The results are shown below.")

# Generate model using columns 0, 6, 8 only (APM, NumberOfPACs, ActionLatency)
keepersList = [0,6,8]
newDataTrain = KeepColumn(keepersList,normalizedDataTrain)
newDataTest = KeepColumn(keepersList,normalizedDataTest)

accKMeans2, predictedKmeans2 = KMeansClustering(25,trainLabels,newDataTrain,testLabels,newDataTest)
accBIRCH2, predictBIRCH2 = BIRCHClustering(14,trainLabels,newDataTrain,testLabels,newDataTest)
accDT2, predictDT2 = DecisionTreeClassifying(trainLabels,newDataTrain,testLabels,newDataTest)

print("Kmeans Accuracy Before", accKMeansClustering, "Kmeans Accuracy After", accKMeans2)
print("BIRCH Accuracy Before", accBIRCHClustering, "BIRCH Accuracy After", accBIRCH2)
print("Decision Tree Accuracy Before", accDecisionTree, "Decision Tree Accuracy After", accDT2)

print("")
# The model with the most consistent highest accuracy is the BIRCH clustering model. (Approx. 37%)
print("The highest consistent accuracy is",str(accBIRCH2*100)+"%", "using BIRCH Clustering.")
print("This accuracy seems fairly low, lets review the dataset to see if we can understand whats going on.\n")

# Review Dataset
# Though the accuracy may not be very high. Lets review our initial dataset to see if we can learn anything

# Lets start with the distribution of ranks
PlotPieDistribution(trainLabels,outputFigure)

# We can see in the plot that Ranks 1, 7, & 8 make up less than 10% of the data
# Lets try grouping the ranks to see if that allows for a more accurate "Approximate Ranking"
# Lets start by grouping ranks 1,2,3 and 6,7,8 into their own groups. (this gives us 4 different groups, 1-3, 4, 5, 6-8)

print("Figure 1 shows the distribution of ranks among the test data.\n")
print("From this we can see ranks 1, 7, & 8 make up only 10% of the data.")
print("Lets group ranks 1, 2, & 3 into a group and 6, 7, 8 into another group, and keep ranks 4 & 5 individual.\n")

testLabelsGroup1 = []
for i in testLabels:
    if i == 1 or i == 2 or i == 3:
        testLabelsGroup1.append('1,2,3')
    elif i == 6 or i == 7 or i == 8:
        testLabelsGroup1.append('6,7,8')
    else:
        testLabelsGroup1.append(i)

predictLabelsGroup1 = []
for i in predictBIRCH2:
    if i == 1 or i == 2 or i == 3:
        predictLabelsGroup1.append('1,2,3')
    elif i == 6 or i == 7 or i == 8:
        predictLabelsGroup1.append('6,7,8')
    else:
        predictLabelsGroup1.append(i)

accGroup1 = round(accuracy_score(testLabelsGroup1, predictLabelsGroup1),4)

print("The accuracy for this 'Approximate Ranking' is", str(accGroup1*100)+"%.")
print("This is still not that great...\n")

# This gave us an accuracy of 49%. Still not that great...

# Alternative Accuracy Measure

# Lets see if we can reduce the precision and find an alternative accuracy measure
# Lets see how close our BIRCH clustering model predicts our ranking within +- 1 ranking.

print("Lets reduce the precision and see if the model can predict rankings within +1 or -1 of the actual ranking.")

counter = 0
for i in range(0,len(testLabels),1):
    if predictBIRCH2[i] == testLabels[i] \
    or predictBIRCH2[i] + 1 == testLabels[i] \
    or predictBIRCH2[i] - 1 == testLabels[i]:
        counter += 1
accNew = round(counter/len(testLabels),4)

print("The accuracy for 'Ranking ±1' is",str(round(accNew*100,2))+"%.\n\n")

# Reducing the precision helped alot! With that we can show an accuracy of 80%! Which will be must eaiser to explain to a non-technical stakeholder

print('Therefore...\n')
print("The accuracy of our model to predict the exact ranking is:", str(round(accBIRCH2*100,0))+"%")
print("The accuracy of our model to predict the ranking within ±1 ranking is:", str(round(accNew*100,0))+"%")




################################## CONVERT ALL PRINT OUT TO PDF ###########################################
pdf.write(6,"Hello, Welcome to John Le's Data Science Assessment Project for Evil Genius\n\n\n")


pdf.write(6,"STEP 1: DATA EXTRACTION\n\n")

pdf.write(6,"The data from {} is partitioned into training & testing data.\n".format(fileData))
pdf.write(6,"Every {}th entry is considered test data, the rest is training data.\n\n".format(n))

pdf.write(6,"Data is reformated & normalized.\n\n\n")


pdf.write(6,"STEP 2: Determine which attributes/features to use\n\n")

pdf.write(6,"Using all attributes, the training data is used to build 3 different models (KMeans Clustering, BIRCH Clustering, Decision Tree Classification).\n\n")

pdf.write(6,"The testing data is used to calculate the accuracy of each model. Results can be seen below.\n")
pdf.write(6,"KMeans Accuracy: {} \n".format(accKMeansClustering))
pdf.write(6,"BIRCH Accuracy: {} \n".format(accBIRCHClustering))
pdf.write(6,"Decision Tree Accuracy: {} \n".format(accDecisionTree))
pdf.write(6,"\n")
pdf.write(6,"The above process is repeated for each attribute individually\n")
pdf.write(6,"The columns (attributes) with the highest accuracy is seen below.\n")

for i in range(-1,-8,-1):
    pdf.write(6,"{} \n".format(sortedDictAcc[i]))

pdf.write(6,"We can deduce that columns 0, 6, 8 consistently have the highest accuracy.\n\n\n")


pdf.write(6,"STEP 3: Create Model & Analyze Results\n\n")

pdf.write(6,"Models using KMeans Clustering, BIRCH Clustering, & Decision Tree Classification are rebuilt using the selected columns(attributes) only.\n")
pdf.write(6,"The results are shown below.\n")
pdf.write(6,"Kmeans Accuracy Before: {}, Kmeans Accuracy After: {}\n".format(accKMeansClustering, accKMeans2))
pdf.write(6,"BIRCH Accuracy Before: {}, BIRCH Accuracy After: {}\n".format(accBIRCHClustering, accBIRCH2))
pdf.write(6,"Decision Tree Accuracy Accuracy Before: {}, Decision Tree Accuracy After: {}\n\n".format(accDecisionTree, accDT2))

pdf.write(6,"The highest consistent accuracy is {}% using BIRCH Clustering. \n".format(str(accBIRCH2*100)))
pdf.write(6,"This accuracy seems fairly low, lets review the dataset to see if we can understand whats going on.\n\n")

pdf.write(6,"The pie chart (See Figure1.pdf) shows the distribution of ranks among the training data\n\n")

pdf.write(6,"From this we can see ranks 1, 7, & 8 make up only 10% of the data.\n")
pdf.write(6,"Lets group ranks 1, 2, & 3 into a group and 6, 7, 8 into another group, and keep ranks 4 & 5 individual.\n\n")

pdf.write(6,"The accuracy for this 'Approximate Ranking' is {}%.\n".format(str(accGroup1*100)))
pdf.write(6,"This is still not that great...\n\n")

pdf.write(6,"Lets reduce the precision and see if the model can predict rankings within +1 or -1 of the actual ranking.\n")
pdf.write(6,"The accuracy for 'Ranking ±1' is {}%.\n\n\n".format(str(round(accNew*100,2))))


pdf.write(6,"Therefore...\n\n")
pdf.write(6,"The accuracy of our model to predict the exact ranking is: {}%.\n".format(str(round(accBIRCH2*100,0))))
pdf.write(6,"The accuracy of our model to predict the ranking within ±1 ranking is: {}%.\n".format(str(round(accNew*100,0))))

pdf.output(outputFile)
###################################