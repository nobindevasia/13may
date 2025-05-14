using System;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.AutoML;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Utils;
using D2G.Iris.ML.Core.Interfaces;

namespace D2G.Iris.ML.Training
{
    public class RegressionTrainer : BaseModelTrainer
    {
        public RegressionTrainer(MLContext mlContext, TrainerFactory trainerFactory)
            : base(mlContext, trainerFactory)
        {
        }

        private class RegressionDataPoint
        {
            [VectorType]
            public float[] Features { get; set; }
            public float Label { get; set; }
        }

        public override async Task<ITransformer> TrainModel(
            MLContext mlContext,
            IDataView dataView,
            string[] featureNames,
            ModelConfig config,
            ProcessedData processedData)
        {
            Console.WriteLine($"\nStarting regression model training using {(config.AutoML?.Enabled == true ? "AutoML" : config.TrainingParameters.Algorithm)}...");

            try
            {
                if (!dataView.Schema.GetColumnOrNull(config.TargetField).HasValue)
                {
                    throw new InvalidOperationException($"Target column '{config.TargetField}' not found in dataset. Available columns: {string.Join(", ", dataView.Schema.Select(c => c.Name))}");
                }

                IDataView labeledData;
                if (!dataView.Schema.GetColumnOrNull("Label").HasValue)
                {
                    var labelPipeline = mlContext.Transforms.CopyColumns("Label", config.TargetField);
                    labeledData = labelPipeline.Fit(dataView).Transform(dataView);
                }
                else
                {
                    labeledData = dataView;
                }

                IDataView preparedData = PrepareData(labeledData, featureNames);

                if (config.AutoML?.Enabled == true)
                {
                    return await TrainWithAdvancedAutoML(mlContext, preparedData, featureNames, config, processedData);
                }

                return await TrainWithTraditionalApproach(mlContext, preparedData, featureNames, config, processedData);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\nError during regression training: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"Inner Exception: {ex.InnerException.Message}");
                }
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
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
                        return parts[parts.Length - 1].Trim();
                    }
                }
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

                if (!Enum.TryParse(config.AutoML.OptimizingMetric, out RegressionMetric metric))
                {
                    Console.WriteLine($"Warning: Unknown OptimizingMetric '{config.AutoML.OptimizingMetric}', defaulting to {nameof(RegressionMetric.RSquared)}");
                    metric = RegressionMetric.RSquared;
                }

                var experimentSettings = new RegressionExperimentSettings
                {
                    MaxExperimentTimeInSeconds = (uint)config.AutoML.MaxExperimentTimeInSeconds,
                    OptimizingMetric = metric
                };

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

                Console.WriteLine("Creating experiment...");
                var experiment = mlContext.Auto().CreateRegressionExperiment(experimentSettings);

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
                Console.WriteLine("Rank | Model Type                | R²      | MAE      | RMSE     | Runtime");
                Console.WriteLine("-----|---------------------------|---------|----------|----------|--------");

                int rank = 1;
                var orderedRuns = OrderRunsByMetric(experimentResult.RunDetails.Where(r => r.ValidationMetrics != null), metric);

                foreach (var run in orderedRuns.Take(5))
                {
                    string trainerName = CleanTrainerName(run.TrainerName);


                    Console.WriteLine($"{rank,4} | {trainerName,-24} | {run.ValidationMetrics.RSquared,7:F4} | {run.ValidationMetrics.MeanAbsoluteError,8:F4} | {run.ValidationMetrics.RootMeanSquaredError,8:F4} | {run.RuntimeInSeconds,6:F1}s");
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
                Console.WriteLine($"  R²:                         {metrics.RSquared:F4}");
                Console.WriteLine($"  Mean Absolute Error:        {metrics.MeanAbsoluteError:F4}");
                Console.WriteLine($"  Mean Squared Error:         {metrics.MeanSquaredError:F4}");
                Console.WriteLine($"  Root Mean Squared Error:    {metrics.RootMeanSquaredError:F4}");

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
                    safeName = "RegressionModel";

                var modelPath = $"Regression_AutoML_{safeName}_Model.zip";
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

        private IEnumerable<RunDetail<RegressionMetrics>> OrderRunsByMetric(
            IEnumerable<RunDetail<RegressionMetrics>> runs,
            RegressionMetric metric)
        {
            switch (metric)
            {
                case RegressionMetric.MeanAbsoluteError:
                    return runs.OrderBy(r => r.ValidationMetrics.MeanAbsoluteError);
                case RegressionMetric.MeanSquaredError:
                    return runs.OrderBy(r => r.ValidationMetrics.MeanSquaredError);
                case RegressionMetric.RootMeanSquaredError:
                    return runs.OrderBy(r => r.ValidationMetrics.RootMeanSquaredError);
                case RegressionMetric.RSquared:           
                    return runs.OrderByDescending(r => r.ValidationMetrics.RSquared);
                default:
                    return runs.OrderByDescending(r => r.ValidationMetrics.RSquared);
            }
        }

        private double GetMetricValue(RegressionMetrics metrics, RegressionMetric metric)
        {
            switch (metric)
            {
                case RegressionMetric.MeanAbsoluteError:
                    return metrics.MeanAbsoluteError;
                case RegressionMetric.MeanSquaredError:
                    return metrics.MeanSquaredError;
                case RegressionMetric.RootMeanSquaredError:
                    return metrics.RootMeanSquaredError;
                case RegressionMetric.RSquared:
                    return metrics.RSquared;
                default:
                    return metrics.RSquared;
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
                _mlContext,
                preparedData,
                config.TrainingParameters.TestFraction);

            var trainer = _trainerFactory.GetTrainer(
                config.ModelType,
                config.TrainingParameters);

            var pipeline = GetBasePipeline(_mlContext)
                .Append(trainer);

            var model = await TrainModelAsync(pipeline, split.TrainSet);

            var metrics = EvaluateRegression(
                _mlContext,
                model,
                split.TestSet,
                config.TrainingParameters.Algorithm);

            SaveModel(
                _mlContext,
                model,
                preparedData,
                "Regression",
                config.TrainingParameters.Algorithm);

            await SaveModelInfo(
                metrics,
                preparedData,
                featureNames,
                config,
                processedData);

            return model;
        }

        private IDataView PrepareData(IDataView dataView, string[] featureNames)
        {
            try
            {
                if (dataView.Schema.GetColumnOrNull("Features").HasValue)
                {
                    var dataPoints = _mlContext.Data
                        .CreateEnumerable<RegressionDataPoint>(dataView, reuseRowObject: false)
                        .ToList();

                    var schemaDef = SchemaDefinition.Create(typeof(RegressionDataPoint));
                    schemaDef[nameof(RegressionDataPoint.Features)].ColumnType = new VectorDataViewType(
                        NumberDataViewType.Single,
                        featureNames.Length);

                    return _mlContext.Data.LoadFromEnumerable(dataPoints, schemaDef);
                }
                else
                {
                    var featuresPipeline = _mlContext.Transforms.Concatenate("Features", featureNames);
                    return featuresPipeline.Fit(dataView).Transform(dataView);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error preparing data: {ex.Message}");
                Console.WriteLine($"Available columns: {string.Join(", ", dataView.Schema.Select(c => c.Name))}");
                throw;
            }
        }

    }
}