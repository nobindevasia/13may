using System;
using System.Linq;
using System.IO;
using System.Threading.Tasks;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.AutoML;
using D2G.Iris.ML.Core.Interfaces;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Utils;

namespace D2G.Iris.ML.Training
{
    public abstract class BaseModelTrainer : IModelTrainer
    {
        protected readonly MLContext _mlContext;
        protected readonly TrainerFactory _trainerFactory;

        protected BaseModelTrainer(MLContext mlContext, TrainerFactory trainerFactory)
        {
            _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
            _trainerFactory = trainerFactory ?? throw new ArgumentNullException(nameof(trainerFactory));
        }

        public abstract Task<ITransformer> TrainModel(
            MLContext mlContext,
            IDataView dataView,
            string[] featureNames,
            ModelConfig config,
            ProcessedData processedData);

        protected void CreateCacheDirectory(string cacheDirName = "AutoMLCache")
        {
            if (!Directory.Exists(cacheDirName))
            {
                Directory.CreateDirectory(cacheDirName);
            }
        }

        protected DataSplit SplitTrainTestData(MLContext mlContext, IDataView dataView, double testFraction)
        {
            var split = mlContext.Data.TrainTestSplit(dataView, testFraction: testFraction);

            Console.WriteLine($"Training set size: {GetDataViewRowCount(split.TrainSet)}");
            Console.WriteLine($"Test set size: {GetDataViewRowCount(split.TestSet)}");

            return new DataSplit
            {
                TrainSet = split.TrainSet,
                TestSet = split.TestSet
            };
        }

        protected IEstimator<ITransformer> GetBasePipeline(MLContext mlContext)
        {
            return mlContext.Transforms.NormalizeMinMax("Features");
        }

        protected async Task<ITransformer> TrainModelAsync(IEstimator<ITransformer> pipeline, IDataView trainData)
        {
            return await Task.Run(() => pipeline.Fit(trainData));
        }

        protected BinaryClassificationMetrics EvaluateBinaryClassification(
            MLContext mlContext,
            ITransformer model,
            IDataView testData,
            string algorithmName)
        {
            var predictions = model.Transform(testData);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions);

            PrintBinaryClassificationMetrics(metrics, algorithmName);

            return metrics;
        }

        protected MulticlassClassificationMetrics EvaluateMultiClassClassification(
            MLContext mlContext,
            ITransformer model,
            IDataView testData,
            string algorithmName)
        {
            var predictions = model.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            PrintMultiClassClassificationMetrics(metrics, algorithmName);

            return metrics;
        }

        protected RegressionMetrics EvaluateRegression(
            MLContext mlContext,
            ITransformer model,
            IDataView testData,
            string algorithmName)
        {
            var predictions = model.Transform(testData);
            var metrics = mlContext.Regression.Evaluate(predictions);

            PrintRegressionMetrics(metrics, algorithmName);

            return metrics;
        }

        protected string CleanTrainerName(string trainerName)
        {
            if (string.IsNullOrEmpty(trainerName))
                return "Unknown";
            if (trainerName.Contains("=>"))
            {
                var parts = trainerName.Split("=>")
                    .Select(p => p.Trim())
                    .Where(p => !string.IsNullOrEmpty(p) &&
                           !p.Contains("Unknown") &&
                           p != "Concatenate" &&
                           p != "ReplaceMissingValues")
                    .ToList();

                string result = parts.Any() ? parts.Last() : trainerName.Split("=>").Last().Trim();
                if (result.EndsWith("Multi"))
                {
                    result = result.Substring(0, result.Length - 5);
                }

                return result;
            }

            if (trainerName.EndsWith("Multi"))
            {
                return trainerName.Substring(0, trainerName.Length - 5);
            }

            return trainerName;
        }

        protected string SanitizeFileName(string name, string defaultName = "Model")
        {
            if (string.IsNullOrWhiteSpace(name) || name.Trim() == "Unknown")
                return defaultName;

            string safeName = string.Concat(name.Split(Path.GetInvalidFileNameChars()));
            return safeName.Replace("=>", "_").Replace(">", "_").Replace("<", "_");
        }

        protected string SaveModel(
            MLContext mlContext,
            ITransformer model,
            IDataView dataView,
            string modelType,
            string algorithmName)
        {
            string safeName = SanitizeFileName(algorithmName);
            string modelPath = $"{modelType}_{safeName}_Model.zip";

            mlContext.Model.Save(model, dataView.Schema, modelPath);
            Console.WriteLine($"Model saved to: {modelPath}");

            return modelPath;
        }
        protected async Task SaveModelInfo(
            object metrics,
            IDataView dataView,
            string[] featureNames,
            ModelConfig config,
            ProcessedData processedData)
        {
            await Task.CompletedTask;
        }

        protected long GetDataViewRowCount(IDataView dataView)
        {
            return dataView.GetRowCount() ?? -1;
        }


        protected void PrintBinaryClassificationMetrics(BinaryClassificationMetrics metrics, string algorithmName)
        {
            Console.WriteLine($"\nEvaluation metrics for {algorithmName}:");
            Console.WriteLine($"  Accuracy:                 {metrics.Accuracy:F4}");
            Console.WriteLine($"  AUC:                      {metrics.AreaUnderRocCurve:F4}");
            Console.WriteLine($"  F1 Score:                 {metrics.F1Score:F4}");
            Console.WriteLine($"  Positive Precision:       {metrics.PositivePrecision:F4}");
            Console.WriteLine($"  Positive Recall:          {metrics.PositiveRecall:F4}");
            Console.WriteLine($"  Negative Precision:       {metrics.NegativePrecision:F4}");
            Console.WriteLine($"  Negative Recall:          {metrics.NegativeRecall:F4}");
            Console.WriteLine($"  Area Under PRC:           {metrics.AreaUnderPrecisionRecallCurve:F4}");
        }

        protected void PrintMultiClassClassificationMetrics(MulticlassClassificationMetrics metrics, string algorithmName)
        {
            Console.WriteLine($"\nEvaluation metrics for {algorithmName}:");
            Console.WriteLine($"  Micro-Accuracy:            {metrics.MicroAccuracy:F4}");
            Console.WriteLine($"  Macro-Accuracy:            {metrics.MacroAccuracy:F4}");
            Console.WriteLine($"  Log Loss:                  {metrics.LogLoss:F4}");
            Console.WriteLine($"  Log Loss Reduction:        {metrics.LogLossReduction:F4}");
            Console.WriteLine($"  Top K Accuracy:            {metrics.TopKAccuracy:F4}");
        }

        protected void PrintRegressionMetrics(RegressionMetrics metrics, string algorithmName)
        {
            Console.WriteLine($"\nEvaluation metrics for {algorithmName}:");
            Console.WriteLine($"  R²:                         {metrics.RSquared:F4}");
            Console.WriteLine($"  Mean Absolute Error:        {metrics.MeanAbsoluteError:F4}");
            Console.WriteLine($"  Mean Squared Error:         {metrics.MeanSquaredError:F4}");
            Console.WriteLine($"  Root Mean Squared Error:    {metrics.RootMeanSquaredError:F4}");
        }
    }

    public class DataSplit
    {
        public IDataView TrainSet { get; set; }
        public IDataView TestSet { get; set; }
    }
}