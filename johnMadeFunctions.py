# John Le's made functions for Evil Geniuses x Genius League: Data Scientist Assessment

# The following functions are listed in this file:
    # ExtractData(filePath)                                                     Imports file, remove missing data, convert to floats
    # FileToList(filePath)                                                      Import file
    # IdMissingDataColumns(list)                                                return list of columns that have missing data
    # RemoveColumn(columnNumber, list)                                          Remove specific column (Starts at 0) from list
    # KeepColumns(columnslist, list)                                            Given a matrix, keeps only the specified column.
    # ConvertToFloats(list)                                                     Convert data into floats
    # ConvertColumnToInt(list, column)                                          Convert given column into interger
    # PartitionTestTrainData(list,n)                                            Given a list, return a list of training & testing data. Testing data is every "n" rows
    # NormalizeData(list)                                                       Normalize Data based on min/max range (assumes 1st column is label) & split out labels
    # KMeansClustering(clusters,trainLabels,trainData,testLabels,testData)      Return Accuracy of using KmeansClustering on this data
    # BIRCHClustering(clusters,trainLabels,trainData,testLabels,testData)       Return Accuracy of using BIRCHClustering on this data
    # DecisionTreeClassifying(trainLabels,trainData,testLabels,testData)        Return Accuracy of using Decision Tree Classification on this data
    # PlotPieDistribution(data_list,pdfFile,title)                              Given a list, output name, title. Output the distribution to a PDF file



import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

### Testing Variables

filePath = "starcraft_player_data.csv"


####################################################################################################
# ExtractData Function
####################################################################################################

# # Secondary Functions

# # # Import data, make big list:
def FileToList(filePath):
    infile = open(filePath)
    list = []
    next(infile)
    for line in infile:
        list.append(line.split(","))
    infile.close()
    return list

# # # Remove data points with missing data (Assumes ? are missing data) & Convert to Floats
def RemoveMissingData(list):
    missingSymbol = '"?"'
    removedDataCounter = 0
    newList = []
    for i in range(0,len(list),1):
        missingSymbolFound = False
        newLine = []
        for j in range(0,len(list[i]),1):
            if list[i][j] == missingSymbol:
                missingSymbolFound = True
            else:
                newLine.append(float(list[i][j].strip().strip('"')))
        if missingSymbolFound == True:
            removedDataCounter += 1
        else:
            newList.append(newLine)
    # print(removedDataCounter)
    return newList


# # Primary Function
def ExtractData(filePath):
    dataList = FileToList(filePath)
    result = RemoveMissingData(dataList)
    return result

####################################################################################################
####################################################################################################



####################################################################################################
# IdMissingDataColumns Function
####################################################################################################

def IdMissingDataColumns(list):
    missingSymbol = '"?"'
    columnsWithMissingData = []
    for i in range(0,len(list),1):
        for j in range(0,len(list[i]),1):
            if list[i][j] == missingSymbol:
                if j in columnsWithMissingData:
                    pass
                else:
                    columnsWithMissingData.append(j)
    return columnsWithMissingData


# # Primary Function
def ExtractData(filePath):
    dataList = FileToList(filePath)
    result = RemoveMissingData(dataList)
    return result

####################################################################################################
####################################################################################################


####################################################################################################
# RemoveColumn Function
####################################################################################################

def RemoveColumn(columnNumber, list):
    newList = []
    for i in range(0,len(list),1):
        newLine = []
        for j in range(0,len(list[i]),1):
            if j == columnNumber:
                pass
            else:
                newLine.append(list[i][j])
        newList.append(newLine)
    return newList


####################################################################################################
####################################################################################################



####################################################################################################
# KeepColumns Function
####################################################################################################

def KeepColumn(columnsList, list):
    newList = []
    for i in range(0,len(list),1):
        newLine = []
        for j in range(0,len(list[i]),1):
            if j in columnsList:
                newLine.append(list[i][j])
            else:
                pass
        newList.append(newLine)
    return newList


####################################################################################################
####################################################################################################



####################################################################################################
# ConvertToFloats
####################################################################################################

def ConvertToFloats(list):
    newList = []
    for  i in range(0,len(list),1):
        newLine = []
        for j in range(0,len(list[i]),1):
            newLine.append(float(list[i][j].strip().strip('"')))
        newList.append(newLine)
    return newList

####################################################################################################
####################################################################################################




####################################################################################################
# ConvertColumnToInt(list, column)
####################################################################################################

def ConvertColumnToInt(list, column):
    newList = []
    for  i in range(0,len(list),1):
        newLine = []
        for j in range(0,len(list[i]),1):
            if j == column:
                newLine.append(int(list[i][j]))
            else:
                newLine.append(list[i][j])
        newList.append(newLine)
    return newList

####################################################################################################
####################################################################################################



####################################################################################################
#  PartitionTestTrainData Function
####################################################################################################

def PartitionTestTrainData(list,n):
    trainData = []
    testData = []
    for i in range(0,len(list),1):
        if (i+1)%n == 0:
            testData.append(list[i])
        else:
            trainData.append(list[i])
    return trainData, testData

####################################################################################################
####################################################################################################



####################################################################################################
# NormalizeData Function
####################################################################################################

# # Secondary Functions

# # # Transpose Matrix

def TransposeMatrix(matrix):
    resultsMatrix = []

    for k in range(0,len(matrix[0]),1):
        resultsMatrix.append([])

    for i in matrix:
        for k in range(0,len(matrix[0]),1):
            resultsMatrix[k].append(i[k])
    return resultsMatrix


# # Primary Function

def NormalizeData(list):
    transposedList = TransposeMatrix(list)
    normalizedTransposedList = []
    labelList = transposedList[0]
    # normalizedTransposedList.append(transposedList[0])
    for i in range(1,len(transposedList),1):    # Skip first column (label)
        maxValue = max(transposedList[i])
        minValue = min(transposedList[i])
        rangeValue = maxValue - minValue
        # print(rangeValue)
        # print(i)
        newLine = []
        for j in range(0,len(transposedList[i]),1):
            normalizedValue = (transposedList[i][j] - minValue) / rangeValue
            newLine.append(normalizedValue)
        normalizedTransposedList.append(newLine)
    resultList = TransposeMatrix(normalizedTransposedList)
    return resultList, labelList


####################################################################################################
####################################################################################################



####################################################################################################
# KMeansClustering(clusters,trainLabels,trainData,testLabels,testData) Function
####################################################################################################

# This template is originally generated using CHATGPT & modified by John Le to work

def KMeansClustering(clusters,trainLabels,trainData,testLabels,testData):

    # Due to typeError: only integer scalar arrays can be converted to a scalar index
    # Convert train_labels to NumPy array and then to integer type
    train_labels = np.array(trainLabels, dtype=int)

    #### KMEANS
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(trainData)


    # Predict the clusters for training data
    train_clusters = kmeans.predict(trainData)

    # Predict the clusters for test data
    test_clusters = kmeans.predict(testData)

    # Create a dictionary to map the majority cluster label to the original class label
    cluster_label_map = {}

    # Iterate over each cluster and find the majority class label in that cluster
    for cluster in range(kmeans.n_clusters):
        cluster_labels = train_labels[train_clusters == cluster]
        majority_label = np.bincount(cluster_labels).argmax()
        cluster_label_map[cluster] = majority_label

    # Predict the labels for test data based on majority class label in each cluster
    predicted_labels = [cluster_label_map[cluster] for cluster in test_clusters]

    # Calculate the accuracy score of the predictions
    accuracy = round(accuracy_score(testLabels, predicted_labels),4)

    return accuracy, predicted_labels

####################################################################################################
####################################################################################################



####################################################################################################
#  BIRCHClustering(clusters,trainLabels,trainData,testLabels,testData) Function
####################################################################################################

# This template is originally generated using CHATGPT & modified by John Le to work

def BIRCHClustering(clusters,trainLabels,trainData,testLabels,testData):

    # Standardize the training and test data
    scaler = StandardScaler()
    train_data = scaler.fit_transform(trainData)
    test_data = scaler.transform(testData)

    ## BIRCH

    # Create a Bisecting K-means clustering model
    bkmeans = Birch(n_clusters=clusters)  # Adjust the number of clusters as needed

    # Fit the training data to the model
    bkmeans.fit(train_data)

    # Predict the clusters for training data
    train_clusters = bkmeans.predict(train_data)

    # Convert train_labels to NumPy array and then to integer type
    train_labels = np.array(trainLabels, dtype=int)

    # Predict the clusters for test data
    test_clusters = bkmeans.predict(test_data)

    # Create a dictionary to map the majority cluster label to the original class label
    cluster_label_map = {}

    # Iterate over each cluster and find the majority class label in that cluster
    for cluster in range(bkmeans.n_clusters):
        cluster_labels = train_labels[train_clusters == cluster]
        majority_label = np.bincount(cluster_labels).argmax()
        cluster_label_map[cluster] = majority_label

    # Predict the labels for test data based on majority class label in each cluster
    predicted_labels = [cluster_label_map[cluster] for cluster in test_clusters]

    # Calculate the accuracy score of the predictions
    accuracy = round(accuracy_score(testLabels, predicted_labels),4)

    return accuracy, predicted_labels


####################################################################################################
####################################################################################################



####################################################################################################
# DecisionTreeClassifying(trainLabels,trainData,testLabels,testData) Function
####################################################################################################

# This template is originally generated using CHATGPT & modified by John Le to work

def DecisionTreeClassifying(trainLabels,trainData,testLabels,testData):

    # Standardize the training and test data
    scaler = StandardScaler()
    train_data = scaler.fit_transform(trainData)
    test_data = scaler.transform(testData)

    # Create a Decision Tree classifier
    decision_tree = DecisionTreeClassifier()

    # Fit the training data to the model
    decision_tree.fit(train_data, trainLabels)

    # Predict the labels for test data
    predicted_labels = decision_tree.predict(test_data)

    # Calculate the accuracy score of the predictions
    accuracy = round(accuracy_score(testLabels, predicted_labels),4)

    return accuracy, predicted_labels
####################################################################################################
####################################################################################################



####################################################################################################
# PlotPieDistribution(list, pdfFile, title) Function
####################################################################################################

# This template is originally generated using CHATGPT & modified by John Le to work

def PlotPieDistribution(data_list, pdfFile, title):
    # Count the occurrences of unique elements in the list
    unique_elements = set(data_list)
    counts = [data_list.count(element) for element in unique_elements]

    # Create labels for the pie chart
    labels = list(unique_elements)

    # Plot the pie chart
    plt.figure(figsize=(9, 10))  # Set the figure size to make the pie chart smaller
    plt.pie(counts, labels=labels, autopct='%1.1f%%')
    plt.axis('equal')  # Equal aspect ratio ensures circular chart

    # Add a title to the pie chart
    plt.title(title)

    # Add data to the chart as a text box
    data_text = '\n'.join([f'{label}: {count}' for label, count in zip(labels, counts)])
    plt.text(1.2, 0.5, data_text, bbox=dict(facecolor='white', alpha=0.5), transform=plt.gcf().transFigure)
    
    
    # plt.show()
    plt.savefig(pdfFile, format='pdf')
    plt.close()
    
####################################################################################################
####################################################################################################
