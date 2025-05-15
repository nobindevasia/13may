using System;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.AutoML;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Core.Interfaces;
using D2G.Iris.ML.Utils;

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


                IDataView labeledData = EnsureLabelColumn(mlContext, dataView, config.TargetField);
                IDataView preparedData = PrepareData(labeledData, featureNames);


                return config.AutoML?.Enabled == true
                    ? await TrainWithAdvancedAutoML(mlContext, preparedData, featureNames, config, processedData)
                    : await TrainWithTraditionalApproach(mlContext, preparedData, featureNames, config, processedData);
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

        private IDataView EnsureLabelColumn(MLContext mlContext, IDataView dataView, string targetField)
        {
            if (!dataView.Schema.GetColumnOrNull("Label").HasValue)
            {
                var labelPipeline = mlContext.Transforms.CopyColumns("Label", targetField);
                return labelPipeline.Fit(dataView).Transform(dataView);
            }
            return dataView;
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
                CreateCacheDirectory();

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

                TrySetMaxModels(config, experimentSettings);

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

                PrintTopModels(experimentResult.RunDetails.Where(r => r.ValidationMetrics != null), metric);

                var bestRun = experimentResult.BestRun;
                string bestTrainerName = CleanTrainerName(bestRun.TrainerName);

                Console.WriteLine($"\nBest model: {bestTrainerName}");
                Console.WriteLine($"Training time: {bestRun.RuntimeInSeconds:F1} seconds");

                double bestMetricValue = GetMetricValue(bestRun.ValidationMetrics, metric);
                Console.WriteLine($"Best {metric} value: {bestMetricValue:F4}");

                PrintRegressionMetrics(bestRun.ValidationMetrics, bestTrainerName);

                await SaveModelInfo(
                    bestRun.ValidationMetrics,
                    preparedData,
                    featureNames,
                    config,
                    processedData);

                var safeName = SanitizeFileName(bestTrainerName, "RegressionModel");
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

        private void TrySetMaxModels(ModelConfig config, RegressionExperimentSettings experimentSettings)
        {
            try
            {
                if (config.AutoML.MaxModels > 0)
                {
                    PropertyInfo prop = experimentSettings.GetType().GetProperty("MaxModels");
                    prop?.SetValue(experimentSettings, (uint)config.AutoML.MaxModels);
                    Console.WriteLine($"Set MaxModels to {config.AutoML.MaxModels}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Note: Could not set MaxModels: {ex.Message}");
            }
        }

        private void PrintTopModels(IEnumerable<RunDetail<RegressionMetrics>> runs, RegressionMetric metric)
        {
            Console.WriteLine($"\nTop 5 models evaluated (ranked by {metric}):");
            Console.WriteLine("Rank | Model Type                | R²      | MAE      | RMSE     | Runtime");
            Console.WriteLine("-----|---------------------------|---------|----------|----------|--------");

            int rank = 1;
            var orderedRuns = OrderRunsByMetric(runs, metric);

            foreach (var run in orderedRuns.Take(5))
            {
                string trainerName = CleanTrainerName(run.TrainerName);
                Console.WriteLine($"{rank,4} | {trainerName,-24} | " +
                                 $"{run.ValidationMetrics.RSquared,7:F4} | " +
                                 $"{run.ValidationMetrics.MeanAbsoluteError,8:F4} | " +
                                 $"{run.ValidationMetrics.RootMeanSquaredError,8:F4} | " +
                                 $"{run.RuntimeInSeconds,6:F1}s");
                rank++;
            }
        }

        private IEnumerable<RunDetail<RegressionMetrics>> OrderRunsByMetric(
            IEnumerable<RunDetail<RegressionMetrics>> runs,
            RegressionMetric metric)
        {
            return metric switch
            {
                RegressionMetric.MeanAbsoluteError => runs.OrderBy(r => r.ValidationMetrics.MeanAbsoluteError),
                RegressionMetric.MeanSquaredError => runs.OrderBy(r => r.ValidationMetrics.MeanSquaredError),
                RegressionMetric.RootMeanSquaredError => runs.OrderBy(r => r.ValidationMetrics.RootMeanSquaredError),
                RegressionMetric.RSquared => runs.OrderByDescending(r => r.ValidationMetrics.RSquared),
                _ => runs.OrderByDescending(r => r.ValidationMetrics.RSquared)
            };
        }

        private double GetMetricValue(RegressionMetrics metrics, RegressionMetric metric)
        {
            return metric switch
            {
                RegressionMetric.MeanAbsoluteError => metrics.MeanAbsoluteError,
                RegressionMetric.MeanSquaredError => metrics.MeanSquaredError,
                RegressionMetric.RootMeanSquaredError => metrics.RootMeanSquaredError,
                RegressionMetric.RSquared => metrics.RSquared,
                _ => metrics.RSquared
            };
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

            var trainingStartTime = DateTime.Now;
            var model = await TrainModelAsync(pipeline, split.TrainSet);
            var trainingDuration = DateTime.Now - trainingStartTime;
            Console.WriteLine($"Training completed in {trainingDuration.TotalSeconds:F1} seconds");

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