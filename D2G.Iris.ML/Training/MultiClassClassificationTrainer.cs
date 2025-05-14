using System;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.AutoML;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Core.Interfaces;
using D2G.Iris.ML.Utils;

namespace D2G.Iris.ML.Training
{
    public class MultiClassClassificationTrainer : BaseModelTrainer
    {
        public MultiClassClassificationTrainer(MLContext mlContext, TrainerFactory trainerFactory)
            : base(mlContext, trainerFactory)
        {
        }

        private class ModelInput
        {
            [VectorType]
            public float[] Features { get; set; }
            public long Label { get; set; }
        }

        public override async Task<ITransformer> TrainModel(
            MLContext mlContext,
            IDataView dataView,
            string[] featureNames,
            ModelConfig config,
            ProcessedData processedData)
        {
            Console.WriteLine($"\nStarting multiclass classification model training using {(config.AutoML?.Enabled == true ? "AutoML" : config.TrainingParameters.Algorithm)}...");

            try
            {
                IDataView fixedData = PrepareData(dataView, featureNames);

                if (config.AutoML?.Enabled == true)
                {
                    return await TrainWithAdvancedAutoML(mlContext, fixedData, featureNames, config, processedData);
                }

                var splitData = SplitTrainTestData(
                    _mlContext,
                    fixedData,
                    config.TrainingParameters.TestFraction);
                IEstimator<ITransformer> pipeline = _mlContext.Transforms
                    .NormalizeMinMax("Features")
                    .Append(_mlContext.Transforms.Conversion
                        .MapValueToKey(outputColumnName: "Label", inputColumnName: "Label"))
                    .AppendCacheCheckpoint(_mlContext);
                var trainer = _trainerFactory.GetTrainer(
                    config.ModelType,
                    config.TrainingParameters);

                pipeline = pipeline
                    .Append(trainer)
                    .Append(_mlContext.Transforms.Conversion
                        .MapKeyToValue("PredictedLabel", "PredictedLabel"));
                var model = await TrainModelAsync(pipeline, splitData.TrainSet);
                var metrics = EvaluateMultiClassClassification(
                    _mlContext,
                    model,
                    splitData.TestSet,
                    config.TrainingParameters.Algorithm);

                await SaveModelInfo(
                    metrics,
                    dataView,
                    featureNames,
                    config,
                    processedData);
                SaveModel(
                    _mlContext,
                    model,
                    fixedData,
                    "MultiClassClassification",
                    config.TrainingParameters.Algorithm);
                return model;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\nError during model training: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
                if (ex.InnerException != null)
                    Console.WriteLine($"Inner Exception: {ex.InnerException.Message}");
                throw;
            }
        }

        private string CleanTrainerName(string trainerName)
        {
            if (string.IsNullOrEmpty(trainerName))
                return "Unknown";

            if (trainerName.Contains("=>"))
            {
                var parts = trainerName.Split("=>");

                foreach (var part in parts)
                {
                    string trimmedPart = part.Trim();
                    if (!string.IsNullOrEmpty(trimmedPart) &&
                        !trimmedPart.Contains("Unknown") &&
                        trimmedPart != "Concatenate" &&
                        trimmedPart != "ReplaceMissingValues")
                    {
                        if (trimmedPart.EndsWith("Multi"))
                        {
                            trimmedPart = trimmedPart.Substring(0, trimmedPart.Length - 5);
                        }

                        return trimmedPart;
                    }
                }

                if (parts.Length > 0)
                {
                    if (parts[parts.Length - 1].Trim().Contains("Unknown") && parts.Length > 1)
                    {
                        return parts[parts.Length - 2].Trim();
                    }
                    else
                    {
                        string lastPart = parts[parts.Length - 1].Trim();

                        if (lastPart.EndsWith("Multi"))
                        {
                            lastPart = lastPart.Substring(0, lastPart.Length - 5);
                        }

                        return lastPart;
                    }
                }
            }

            if (trainerName.EndsWith("Multi"))
            {
                trainerName = trainerName.Substring(0, trainerName.Length - 5);
            }

            return string.IsNullOrEmpty(trainerName) ? "Unknown" : trainerName;
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
                string cacheDir = "AutoMLCache";
                if (!Directory.Exists(cacheDir))
                {
                    Directory.CreateDirectory(cacheDir);
                }

                if (!Enum.TryParse(config.AutoML.OptimizingMetric, out MulticlassClassificationMetric metric))
                {
                    Console.WriteLine($"Warning: Unknown OptimizingMetric '{config.AutoML.OptimizingMetric}', defaulting to {nameof(MulticlassClassificationMetric.MicroAccuracy)}");
                    metric = MulticlassClassificationMetric.MicroAccuracy;
                }

                var experimentSettings = new MulticlassExperimentSettings
                {
                    MaxExperimentTimeInSeconds = (uint)config.AutoML.MaxExperimentTimeInSeconds,
                    OptimizingMetric = metric
                };

             
                Console.WriteLine("Creating experiment");
                var experiment = mlContext.Auto().CreateMulticlassClassificationExperiment(experimentSettings);

                Console.WriteLine("Starting AutoML experiment - this may take a while...");
                var experimentStartTime = DateTime.Now;

                var experimentResult = experiment.Execute(
                    trainData: preparedData,
                    labelColumnName: "Label");

                var experimentDuration = DateTime.Now - experimentStartTime;
                Console.WriteLine($"AutoML experiment completed in {experimentDuration.TotalMinutes:F1} minutes");

                Console.WriteLine("\n=== AutoML Experiment Summary ===");
                Console.WriteLine($"Models evaluated: {experimentResult.RunDetails.Count()}");

                Console.WriteLine($"\nTop 5 models evaluated (ranked by {metric}):");
                Console.WriteLine("Rank | Model Type                | MicroAcc | MacroAcc | LogLoss | Runtime");
                Console.WriteLine("-----|---------------------------|----------|----------|---------|--------");

                int rank = 1;
                var orderedRuns = OrderRunsByMetric(experimentResult.RunDetails.Where(r => r.ValidationMetrics != null), metric);

                foreach (var run in orderedRuns.Take(5))
                {
                    string trainerName = CleanTrainerName(run.TrainerName);

                    Console.WriteLine($"{rank,4} | {trainerName,-24} | {run.ValidationMetrics.MicroAccuracy,8:F4} | {run.ValidationMetrics.MacroAccuracy,8:F4} | {run.ValidationMetrics.LogLoss,7:F4} | {run.RuntimeInSeconds,6:F1}s");
                    rank++;
                }

                var bestRun = experimentResult.BestRun;
                string bestTrainerName = CleanTrainerName(bestRun.TrainerName);

                Console.WriteLine($"\nBest model: {bestTrainerName}");
                Console.WriteLine($"Training time: {bestRun.RuntimeInSeconds:F1} seconds");

                double bestMetricValue = GetMetricValue(bestRun.ValidationMetrics, metric);
                Console.WriteLine($"Best {metric} value: {bestMetricValue:F4}");


                Console.WriteLine("\nBest model validation metrics:");
                var metrics = bestRun.ValidationMetrics;
                Console.WriteLine($"  Micro-Accuracy:            {metrics.MicroAccuracy:F4}");
                Console.WriteLine($"  Macro-Accuracy:            {metrics.MacroAccuracy:F4}");
                Console.WriteLine($"  Log Loss:                  {metrics.LogLoss:F4}");
                Console.WriteLine($"  Log Loss Reduction:        {metrics.LogLossReduction:F4}");
                Console.WriteLine($"  Top K Accuracy:            {metrics.TopKAccuracy:F4}");
                

                await SaveModelInfo(
                    metrics,
                    preparedData,
                    featureNames,
                    config,
                    processedData);

                var safeName = bestTrainerName;

                safeName = string.Concat(safeName.Split(Path.GetInvalidFileNameChars()));
                safeName = safeName.Replace("=>", "_").Replace(">", "_").Replace("<", "_");

                if (string.IsNullOrWhiteSpace(safeName) || safeName.Trim() == "Unknown")
                    safeName = "MulticlassModel";

                var modelPath = $"MultiClassClassification_AutoML_{safeName}_Model.zip";
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

                Console.WriteLine("Falling back to traditional approach");

                var splitData = SplitTrainTestData(
                    mlContext,
                    preparedData,
                    config.TrainingParameters.TestFraction);
                IEstimator<ITransformer> pipeline = mlContext.Transforms
                    .NormalizeMinMax("Features")
                    .Append(mlContext.Transforms.Conversion
                        .MapValueToKey(outputColumnName: "Label", inputColumnName: "Label"))
                    .AppendCacheCheckpoint(mlContext);
                var trainer = _trainerFactory.GetTrainer(
                    config.ModelType,
                    config.TrainingParameters);

                pipeline = pipeline
                    .Append(trainer)
                    .Append(mlContext.Transforms.Conversion
                        .MapKeyToValue("PredictedLabel", "PredictedLabel"));
                var model = await TrainModelAsync(pipeline, splitData.TrainSet);
                var metrics = EvaluateMultiClassClassification(
                    mlContext,
                    model,
                    splitData.TestSet,
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
                    "MultiClassClassification",
                    config.TrainingParameters.Algorithm);
                return model;
            }
        }

        private IEnumerable<RunDetail<MulticlassClassificationMetrics>> OrderRunsByMetric(
            IEnumerable<RunDetail<MulticlassClassificationMetrics>> runs,
            MulticlassClassificationMetric metric)
        {
            switch (metric)
            {
                case MulticlassClassificationMetric.LogLoss:
                    return runs.OrderBy(r => r.ValidationMetrics.LogLoss);
                case MulticlassClassificationMetric.LogLossReduction:
                    return runs.OrderByDescending(r => r.ValidationMetrics.LogLossReduction);
                case MulticlassClassificationMetric.MacroAccuracy:
                    return runs.OrderByDescending(r => r.ValidationMetrics.MacroAccuracy);
                case MulticlassClassificationMetric.MicroAccuracy:
                    return runs.OrderByDescending(r => r.ValidationMetrics.MicroAccuracy);
                default:
                    return runs.OrderByDescending(r => r.ValidationMetrics.MicroAccuracy);
            }
        }
        private double GetMetricValue(MulticlassClassificationMetrics metrics, MulticlassClassificationMetric metric)
        {
            switch (metric)
            {
                case MulticlassClassificationMetric.LogLoss:
                    return metrics.LogLoss;
                case MulticlassClassificationMetric.LogLossReduction:
                    return metrics.LogLossReduction;
                case MulticlassClassificationMetric.MacroAccuracy:
                    return metrics.MacroAccuracy;
                case MulticlassClassificationMetric.MicroAccuracy:
                    return metrics.MicroAccuracy;
                default:
                    return metrics.MicroAccuracy;
            }
        }

        private IDataView PrepareData(IDataView dataView, string[] featureNames)
        {
            var data = _mlContext.Data
                .CreateEnumerable<ModelInput>(dataView, reuseRowObject: false)
                .Select(row => new ModelInput
                {
                    Features = row.Features,
                    Label = row.Label
                })
                .ToList();
            var schema = SchemaDefinition.Create(typeof(ModelInput));
            schema["Features"].ColumnType =
                new VectorDataViewType(NumberDataViewType.Single, featureNames.Length);
            return _mlContext.Data.LoadFromEnumerable(data, schema);
        }
    }
}