from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics

import pyspark.sql.functions as func
import pyspark
# import findspark

# findspark.init()

conf = SparkConf().setAppName("Wine Quality Prediction").setMaster("local[4]")
sc = SparkContext(conf=conf)

spark = SparkSession.builder.getOrCreate()

# Read the data and Print the schema
print("\n\nThe Program has started...\n\n")
defTrain = spark.read.format('csv').options(header='true', inferSchema='true', delimiter=';').csv(
    "s3://myprogrambucket/TrainingDataset.csv")

print("\n\nPrinting Training Schema\n\n")
defTrain.printSchema()
defTrain.count()

# Make an array of features excluding the column to classify

featureColumns = [col for col in defTrain.columns if (
    col != '""""quality"""""')]

assembler = VectorAssembler(inputCols=featureColumns, outputCol='features')

dataDF = assembler.transform(defTrain)

print("\n\nPrinting Training Schema with Features Table\n\n")
dataDF.printSchema()

# Random Splitting of Data

splitValue = 0.7
trainingDF, testDF = defTrain.randomSplit([splitValue, 1 - splitValue])
print("\nSplitted Data into Training and Testing Dataset\n")

# Random Forest Regression on TrainingDataset

rf = RandomForestClassifier(featuresCol='features', labelCol='""""quality"""""',
                            numTrees=100, maxBins=484, maxDepth=25, minInstancesPerNode=5, seed=34)
rfPipeline = Pipeline(stages=[assembler, rf])
rfPipelineModel = rfPipeline.fit(trainingDF)
evaluator = RegressionEvaluator(
    labelCol='""""quality"""""', predictionCol="prediction", metricName="rmse")
rfTrainingPredictions = rfPipelineModel.transform(defTrain)
rfTestPredictions = rfPipelineModel.transform(testDF)

print("\nCompleted Model Training...\n\nRandom Forest RMSE on traning data = %g\n" %
        evaluator.evaluate(rfTrainingPredictions))
print("\nRandom Forest RMSE on test data = %g\n" %
        evaluator.evaluate(rfTestPredictions))

rf.save("s3://myprogrambucket/rfwine_model.model")
