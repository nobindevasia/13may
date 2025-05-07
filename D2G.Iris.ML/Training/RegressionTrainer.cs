using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
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
            try
            {
                Console.WriteLine($"\nStarting regression model training using {config.TrainingParameters.Algorithm}...");
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
                    dataView,
                    featureNames,
                    config,
                    processedData);

                return model;
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