using System;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Generic;
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
                IDataView preparedData = PrepareData(dataView, featureNames);

                return config.AutoML?.Enabled == true
                    ? await TrainWithAdvancedAutoML(mlContext, preparedData, featureNames, config, processedData)
                    : await TrainWithTraditionalApproach(mlContext, preparedData, featureNames, config, processedData);
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

                PrintTopModels(experimentResult.RunDetails.Where(r => r.ValidationMetrics != null), metric);

                var bestRun = experimentResult.BestRun;
                string bestTrainerName = CleanTrainerName(bestRun.TrainerName);

                Console.WriteLine($"\nBest model: {bestTrainerName}");
                Console.WriteLine($"Training time: {bestRun.RuntimeInSeconds:F1} seconds");

                double bestMetricValue = GetMetricValue(bestRun.ValidationMetrics, metric);
                Console.WriteLine($"Best {metric} value: {bestMetricValue:F4}");

                PrintMultiClassClassificationMetrics(bestRun.ValidationMetrics, bestTrainerName);

                await SaveModelInfo(
                    bestRun.ValidationMetrics,
                    preparedData,
                    featureNames,
                    config,
                    processedData);

                var safeName = SanitizeFileName(bestTrainerName, "MulticlassModel");
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
                return await TrainWithTraditionalApproach(mlContext, preparedData, featureNames, config, processedData);
            }
        }

        private async Task<ITransformer> TrainWithTraditionalApproach(
            MLContext mlContext,
            IDataView preparedData,
            string[] featureNames,
            ModelConfig config,
            ProcessedData processedData)
        {
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

        private void PrintTopModels(IEnumerable<RunDetail<MulticlassClassificationMetrics>> runs, MulticlassClassificationMetric metric)
        {
            Console.WriteLine($"\nTop 5 models evaluated (ranked by {metric}):");
            Console.WriteLine("Rank | Model Type                | MicroAcc | MacroAcc | LogLoss | Runtime");
            Console.WriteLine("-----|---------------------------|----------|----------|---------|--------");

            int rank = 1;
            var orderedRuns = OrderRunsByMetric(runs, metric);

            foreach (var run in orderedRuns.Take(5))
            {
                string trainerName = CleanTrainerName(run.TrainerName);
                Console.WriteLine($"{rank,4} | {trainerName,-24} | " +
                                 $"{run.ValidationMetrics.MicroAccuracy,8:F4} | " +
                                 $"{run.ValidationMetrics.MacroAccuracy,8:F4} | " +
                                 $"{run.ValidationMetrics.LogLoss,7:F4} | " +
                                 $"{run.RuntimeInSeconds,6:F1}s");
                rank++;
            }
        }

        private IEnumerable<RunDetail<MulticlassClassificationMetrics>> OrderRunsByMetric(
            IEnumerable<RunDetail<MulticlassClassificationMetrics>> runs,
            MulticlassClassificationMetric metric)
        {
            return metric switch
            {
                MulticlassClassificationMetric.LogLoss => runs.OrderBy(r => r.ValidationMetrics.LogLoss),
                MulticlassClassificationMetric.LogLossReduction => runs.OrderByDescending(r => r.ValidationMetrics.LogLossReduction),
                MulticlassClassificationMetric.MacroAccuracy => runs.OrderByDescending(r => r.ValidationMetrics.MacroAccuracy),
                MulticlassClassificationMetric.MicroAccuracy => runs.OrderByDescending(r => r.ValidationMetrics.MicroAccuracy),
                _ => runs.OrderByDescending(r => r.ValidationMetrics.MicroAccuracy)
            };
        }

        private double GetMetricValue(MulticlassClassificationMetrics metrics, MulticlassClassificationMetric metric)
        {
            return metric switch
            {
                MulticlassClassificationMetric.LogLoss => metrics.LogLoss,
                MulticlassClassificationMetric.LogLossReduction => metrics.LogLossReduction,
                MulticlassClassificationMetric.MacroAccuracy => metrics.MacroAccuracy,
                MulticlassClassificationMetric.MicroAccuracy => metrics.MicroAccuracy,
                _ => metrics.MicroAccuracy
            };
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