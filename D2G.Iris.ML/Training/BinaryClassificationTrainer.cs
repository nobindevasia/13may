using System;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.AutoML;
using Microsoft.ML.Trainers;
using D2G.Iris.ML.Core.Interfaces;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Utils;
using Microsoft.ML.Calibrators;

namespace D2G.Iris.ML.Training
{
    public class BinaryClassificationTrainer : BaseModelTrainer
    {
        public BinaryClassificationTrainer(MLContext mlContext, TrainerFactory trainerFactory)
            : base(mlContext, trainerFactory)
        {
        }

        public override async Task<ITransformer> TrainModel(
            MLContext mlContext,
            IDataView dataView,
            string[] featureNames,
            ModelConfig config,
            ProcessedData processedData)
        {
            Console.WriteLine($"\nStarting binary classification using {(config.AutoML?.Enabled == true ? "AutoML" : config.TrainingParameters.Algorithm)}...");

            // Verify that target field exists in the data
            if (!dataView.Schema.GetColumnOrNull(config.TargetField).HasValue)
            {
                throw new InvalidOperationException($"Target column '{config.TargetField}' not found in dataset. Available columns: {string.Join(", ", dataView.Schema.Select(c => c.Name))}");
            }

            try
            {
                // 1. Label setup
                var labelPipeline = mlContext.Transforms.CopyColumns(
                        outputColumnName: "RawLabel", inputColumnName: config.TargetField)
                    .Append(mlContext.Transforms.Conversion.ConvertType(
                        outputColumnName: "Label", inputColumnName: "RawLabel", outputKind: DataKind.Boolean));
                var labeledData = labelPipeline.Fit(dataView).Transform(dataView);

                // 2. Feature setup
                IDataView fixedData = PrepareData(labeledData, featureNames);

                // Log dataset statistics for diagnostics
                Console.WriteLine($"Rows in dataset: {fixedData.GetRowCount()}");

                // 3. AutoML branch
                if (config.AutoML?.Enabled == true)
                {
                    return await TrainWithAdvancedAutoML(mlContext, fixedData, featureNames, config, processedData);
                }

                // 4. Manual pipeline branch
                return await TrainWithTraditionalApproach(mlContext, fixedData, featureNames, config, processedData);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in training process: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"Inner exception: {ex.InnerException.Message}");
                }
                throw;
            }
        }

        private string GetCleanTrainerName(string fullTrainerName)
        {
            // Extract just the algorithm name from the pipeline string
            if (fullTrainerName.Contains("=>"))
            {
                var parts = fullTrainerName.Split("=>");
                return parts[parts.Length - 1].Trim();
            }
            return fullTrainerName;
        }

        private async Task<ITransformer> TrainWithAdvancedAutoML(
            MLContext mlContext,
            IDataView preparedData,
            string[] featureNames,
            ModelConfig config,
            ProcessedData processedData)
        {
            Console.WriteLine($"Maximum experiment time: {config.AutoML.MaxExperimentTimeInSeconds} seconds");
            Console.WriteLine($"Optimizing Metric: {config.AutoML.OptimizingMetric}");

            try
            {
                // Create cache directory if needed
                string cacheDir = "AutoMLCache";
                if (!Directory.Exists(cacheDir))
                    Directory.CreateDirectory(cacheDir);

                // Map string metric to BinaryClassificationMetric enum
                if (!Enum.TryParse(config.AutoML.OptimizingMetric, out BinaryClassificationMetric metric))
                {
                    Console.WriteLine($"Warning: Unknown OptimizingMetric '{config.AutoML.OptimizingMetric}', defaulting to {nameof(BinaryClassificationMetric.Accuracy)}");
                    metric = BinaryClassificationMetric.Accuracy;
                }

                // Set up experiment settings
                var experimentSettings = new BinaryExperimentSettings
                {
                    MaxExperimentTimeInSeconds = (uint)config.AutoML.MaxExperimentTimeInSeconds,
                    OptimizingMetric = metric
                };

                // Optionally set MaxModels via reflection if provided
                try
                {
                    if (config.AutoML.MaxModels > 0)
                    {
                        var prop = experimentSettings.GetType().GetProperty("MaxModels");
                        prop?.SetValue(experimentSettings, (uint)config.AutoML.MaxModels);
                        Console.WriteLine($"Set MaxModels to {config.AutoML.MaxModels}");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Note: Could not set MaxModels: {ex.Message}");
                }

                // Create and run experiment using settings
                // Try this if the API only accepts a uint
                var experiment = mlContext.Auto()
                    .CreateBinaryClassificationExperiment(experimentSettings.MaxExperimentTimeInSeconds);

                var experimentStartTime = DateTime.Now;

                // Execute experiment
                var experimentResult = experiment.Execute(
                    trainData: preparedData,
                    labelColumnName: "Label");

                var experimentDuration = DateTime.Now - experimentStartTime;
                Console.WriteLine($"AutoML experiment completed in {experimentDuration.TotalMinutes:F1} minutes");

                // Enhanced results analysis
                Console.WriteLine("\n=== AutoML Experiment Summary ===");
                Console.WriteLine($"Models evaluated: {experimentResult.RunDetails.Count()}");

                // Show top 5 models tried, sorted by the optimizing metric
                Console.WriteLine("\nTop 5 models evaluated (ranked by {0}):", metric);
                Console.WriteLine("Rank | Model Type                | AUC      | Accuracy | F1 Score | Runtime");
                Console.WriteLine("-----|---------------------------|----------|----------|----------|--------");

                int rank = 1;
                var orderedRuns = OrderRunsByMetric(experimentResult.RunDetails, metric);

                foreach (var run in orderedRuns.Take(5))
                {
                    var cleanName = GetCleanTrainerName(run.TrainerName);
                    Console.WriteLine($"{rank,4} | {cleanName,-24} | {run.ValidationMetrics.AreaUnderRocCurve,8:F4} | {run.ValidationMetrics.Accuracy,8:F4} | {run.ValidationMetrics.F1Score,8:F4} | {run.RuntimeInSeconds,6:F1}s");
                    rank++;
                }

                // Get details about the best model
                var bestRun = experimentResult.BestRun;
                var cleanTrainerName = GetCleanTrainerName(bestRun.TrainerName);
                Console.WriteLine($"\nBest model: {cleanTrainerName}");
                Console.WriteLine($"Training time: {bestRun.RuntimeInSeconds:F1} seconds");

                // Show the best model's value for the optimizing metric
                double bestMetricValue = GetMetricValue(bestRun.ValidationMetrics, metric);
                Console.WriteLine($"Best {metric} value: {bestMetricValue:F4}");

                // Detailed metrics for the best model
                var metrics = bestRun.ValidationMetrics;
                Console.WriteLine("\nBest model validation metrics:");
                Console.WriteLine($"  AUC:                      {metrics.AreaUnderRocCurve:F4}");
                Console.WriteLine($"  Accuracy:                 {metrics.Accuracy:F4}");
                Console.WriteLine($"  F1 Score:                 {metrics.F1Score:F4}");
                Console.WriteLine($"  Positive Precision:       {metrics.PositivePrecision:F4}");
                Console.WriteLine($"  Positive Recall:          {metrics.PositiveRecall:F4}");
                Console.WriteLine($"  Negative Precision:       {metrics.NegativePrecision:F4}");
                Console.WriteLine($"  Negative Recall:          {metrics.NegativeRecall:F4}");
                Console.WriteLine($"  Area Under PRC:           {metrics.AreaUnderPrecisionRecallCurve:F4}");

                // Save model info
                await SaveModelInfo(
                    metrics,
                    preparedData,
                    featureNames,
                    config,
                    processedData);

                // Save the model with sanitized filename
                var sanitizedTrainerName = cleanTrainerName.Replace(">=>", "_").Replace(">", "")
                    .Replace("<", "").Replace(":", "").Replace("/", "").Replace("\\", "")
                    .Replace("*", "").Replace("?", "").Replace("\"", "").Replace("|", "");
                var modelPath = $"BinaryClassification_AutoML_{sanitizedTrainerName}_Model.zip";
                mlContext.Model.Save(bestRun.Model, preparedData.Schema, modelPath);
                Console.WriteLine($"\nModel saved to: {modelPath}");

                return bestRun.Model;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in AutoML process: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"Inner exception: {ex.InnerException.Message}");
                }

                Console.WriteLine("Falling back to traditional approach...");
                return await TrainWithTraditionalApproach(mlContext, preparedData, featureNames, config, processedData);
            }
        }

        // Order the runs by the selected metric
        private IEnumerable<RunDetail<BinaryClassificationMetrics>> OrderRunsByMetric(
            IEnumerable<RunDetail<BinaryClassificationMetrics>> runs,
            BinaryClassificationMetric metric)
        {
            switch (metric)
            {
                case BinaryClassificationMetric.Accuracy:
                    return runs.OrderByDescending(r => r.ValidationMetrics.Accuracy);
                case BinaryClassificationMetric.AreaUnderRocCurve:
                    return runs.OrderByDescending(r => r.ValidationMetrics.AreaUnderRocCurve);
                case BinaryClassificationMetric.AreaUnderPrecisionRecallCurve:
                    return runs.OrderByDescending(r => r.ValidationMetrics.AreaUnderPrecisionRecallCurve);
                case BinaryClassificationMetric.F1Score:
                    return runs.OrderByDescending(r => r.ValidationMetrics.F1Score);
                case BinaryClassificationMetric.NegativePrecision:
                    return runs.OrderByDescending(r => r.ValidationMetrics.NegativePrecision);
                case BinaryClassificationMetric.NegativeRecall:
                    return runs.OrderByDescending(r => r.ValidationMetrics.NegativeRecall);
                case BinaryClassificationMetric.PositivePrecision:
                    return runs.OrderByDescending(r => r.ValidationMetrics.PositivePrecision);
                case BinaryClassificationMetric.PositiveRecall:
                    return runs.OrderByDescending(r => r.ValidationMetrics.PositiveRecall);
                default:
                    return runs.OrderByDescending(r => r.ValidationMetrics.AreaUnderRocCurve);
            }
        }

        // Get the value of a specific metric from BinaryClassificationMetrics
        private double GetMetricValue(BinaryClassificationMetrics metrics, BinaryClassificationMetric metric)
        {
            switch (metric)
            {
                case BinaryClassificationMetric.Accuracy:
                    return metrics.Accuracy;
                case BinaryClassificationMetric.AreaUnderRocCurve:
                    return metrics.AreaUnderRocCurve;
                case BinaryClassificationMetric.AreaUnderPrecisionRecallCurve:
                    return metrics.AreaUnderPrecisionRecallCurve;
                case BinaryClassificationMetric.F1Score:
                    return metrics.F1Score;
                case BinaryClassificationMetric.NegativePrecision:
                    return metrics.NegativePrecision;
                case BinaryClassificationMetric.NegativeRecall:
                    return metrics.NegativeRecall;
                case BinaryClassificationMetric.PositivePrecision:
                    return metrics.PositivePrecision;
                case BinaryClassificationMetric.PositiveRecall:
                    return metrics.PositiveRecall;
                default:
                    return metrics.AreaUnderRocCurve;
            }
        }

        private async Task<ITransformer> TrainWithTraditionalApproach(
            MLContext mlContext,
            IDataView preparedData,
            string[] featureNames,
            ModelConfig config,
            ProcessedData processedData)
        {
            Console.WriteLine($"Using traditional approach with {config.TrainingParameters.Algorithm}");

            var split = SplitTrainTestData(
                mlContext,
                preparedData,
                config.TrainingParameters.TestFraction);

            var trainer = _trainerFactory.GetTrainer(
                config.ModelType,
                config.TrainingParameters);

            Console.WriteLine($"Training with algorithm: {config.TrainingParameters.Algorithm}");
            Console.WriteLine("Algorithm parameters:");
            if (config.TrainingParameters.AlgorithmParameters != null)
            {
                foreach (var param in config.TrainingParameters.AlgorithmParameters)
                {
                    Console.WriteLine($"  {param.Key}: {param.Value}");
                }
            }

            var pipeline = GetBasePipeline(mlContext)
                .Append(trainer)
                .Append(mlContext.Transforms.CopyColumns("Probability", "Score"));

            var trainingStartTime = DateTime.Now;
            var model = await TrainModelAsync(pipeline, split.TrainSet);
            var trainingDuration = DateTime.Now - trainingStartTime;
            Console.WriteLine($"Training completed in {trainingDuration.TotalSeconds:F1} seconds");

            var metrics = EvaluateBinaryClassification(
                mlContext,
                model,
                split.TestSet,
                config.TrainingParameters.Algorithm);

            await SaveModelInfo(
                metrics,
                preparedData,
                featureNames,
                config,
                processedData);

            SaveModel(
                mlContext,
                model,
                preparedData,
                "BinaryClassification",
                config.TrainingParameters.Algorithm);

            return model;
        }

        private IDataView PrepareData(IDataView labeledData, string[] featureNames)
        {
            if (labeledData.Schema.GetColumnOrNull("Features").HasValue)
            {
                var temp = labeledData.GetColumn<VBuffer<float>>("Features")
                    .Zip(labeledData.GetColumn<bool>("Label"), (feat, lbl) => new BinaryVector { Features = feat.GetValues().ToArray(), Label = lbl })
                    .ToList();

                var schemaDef = SchemaDefinition.Create(typeof(BinaryVector));
                schemaDef[nameof(BinaryVector.Features)].ColumnType =
                    new VectorDataViewType(NumberDataViewType.Single, featureNames.Length);

                return _mlContext.Data.LoadFromEnumerable(temp, schemaDef);
            }
            else
            {
                return _mlContext.Transforms.Concatenate("Features", featureNames)
                    .Fit(labeledData)
                    .Transform(labeledData);
            }
        }

        private class BinaryVector
        {
            [VectorType]
            public float[] Features { get; set; }
            public bool Label { get; set; }
        }
    }
}