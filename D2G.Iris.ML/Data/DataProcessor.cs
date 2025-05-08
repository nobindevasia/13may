using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.DataBalancing;
using D2G.Iris.ML.FeatureEngineering;
using D2G.Iris.ML.Core.Interfaces;

namespace D2G.Iris.ML.Data
{
    public class DataProcessor : IDataProcessor
    {
        private readonly DataBalancerFactory _dataBalancerFactory;
        private readonly FeatureSelectorFactory _featureSelectorFactory;
        private readonly ISqlHandler _sqlHandler;

        public DataProcessor()
        {
            _dataBalancerFactory = new DataBalancerFactory();
            _featureSelectorFactory = new FeatureSelectorFactory(new MLContext());
            _sqlHandler = null; 
        }

        public DataProcessor(ISqlHandler sqlHandler)
        {
            _dataBalancerFactory = new DataBalancerFactory();
            _featureSelectorFactory = new FeatureSelectorFactory(new MLContext());
            _sqlHandler = sqlHandler;
        }

        public async Task<ProcessedData> ProcessData(
            MLContext mlContext,
            IDataView rawData,
            string[] enabledFields,
            ModelConfig config)
        {
            Console.WriteLine("\n=============== Processing Data ===============");

            IDataView processedData = rawData;
            string[] currentFeatures = enabledFields.Where(f => f != config.TargetField).ToArray();
            string selectionReport = string.Empty;

            long originalCount = rawData.GetRowCount() ?? 0;
            long balancedCount = originalCount;

            if (!rawData.Schema.GetColumnOrNull("Features").HasValue)
            {
                var initialPipeline = mlContext.Transforms.Concatenate("Features", currentFeatures);
                processedData = initialPipeline.Fit(rawData).Transform(rawData);
            }

            bool balancingFirst = config.DataBalancing.ExecutionOrder <= config.FeatureEngineering.ExecutionOrder;

            if (config.DataBalancing.Method != DataBalanceMethod.None &&
                config.FeatureEngineering.Method != FeatureSelectionMethod.None)
            {
                Console.WriteLine($"Processing order: {(balancingFirst ?
                    "Data Balancing then Feature Selection" :
                    "Feature Selection then Data Balancing")}");
            }

            try
            {
                if (balancingFirst)
                {
                    if (config.DataBalancing.Method != DataBalanceMethod.None)
                    {
                        Console.WriteLine("Applying data balancing first...");
                        var balanceResult = await ProcessDataBalancing(mlContext, processedData, currentFeatures, config);
                        processedData = balanceResult.balancedData;
                        balancedCount = balanceResult.count;

                        var schema = processedData.Schema;
                    }

                    if (config.FeatureEngineering.Method != FeatureSelectionMethod.None)
                    {
                        var featureResult = await ProcessFeatureSelection(mlContext, processedData, currentFeatures, config);
                        processedData = featureResult.transformedData;
                        currentFeatures = featureResult.selectedFeatures;
                        selectionReport = featureResult.report;
                    }
                }
                else
                {
                    if (config.FeatureEngineering.Method != FeatureSelectionMethod.None)
                    {
                        Console.WriteLine("Applying feature selection first...");
                        var featureResult = await ProcessFeatureSelection(mlContext, processedData, currentFeatures, config);
                        processedData = featureResult.transformedData;
                        currentFeatures = featureResult.selectedFeatures;
                        selectionReport = featureResult.report;
                    }
                    if (config.DataBalancing.Method != DataBalanceMethod.None)
                    {
                        Console.WriteLine("Applying data balancing after feature selection...");
                        var balanceResult = await ProcessDataBalancing(mlContext, processedData, currentFeatures, config);
                        processedData = balanceResult.balancedData;
                        balancedCount = balanceResult.count;
                    }
                }

                if (!processedData.Schema.GetColumnOrNull(config.TargetField).HasValue &&
                    processedData.Schema.GetColumnOrNull("Label").HasValue)
                {
                    var labelPipeline = mlContext.Transforms.CopyColumns(config.TargetField, "Label");
                    processedData = labelPipeline.Fit(processedData).Transform(processedData);
                }

                if (_sqlHandler != null && !string.IsNullOrEmpty(config.Database.OutputTableName))
                {
                    try
                    {
                        _sqlHandler.SaveToSql(
                            config.Database.OutputTableName,
                            processedData,
                            currentFeatures,
                            config.TargetField,
                            config.ModelType);

                        Console.WriteLine($"Processed data saved to: {config.Database.OutputTableName}");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error saving processed data: {ex.Message}");
                    }
                }

                return new ProcessedData
                {
                    Data = processedData,
                    FeatureNames = currentFeatures,
                    OriginalSampleCount = (int)originalCount,
                    BalancedSampleCount = (int)balancedCount,
                    FeatureSelectionReport = selectionReport,
                    FeatureSelectionMethod = config.FeatureEngineering.Method,
                    DataBalancingMethod = config.DataBalancing.Method,
                    DataBalancingExecutionOrder = config.DataBalancing.ExecutionOrder,
                    FeatureSelectionExecutionOrder = config.FeatureEngineering.ExecutionOrder
                };
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error during data processing: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
                throw;
            }
        }

        private async Task<(IDataView balancedData, long count)> ProcessDataBalancing(
            MLContext mlContext,
            IDataView data,
            string[] features,
            ModelConfig config)
        {
            var balancer = _dataBalancerFactory.CreateBalancer(config.DataBalancing.Method);
            var balancedData = await balancer.BalanceDataset(
                mlContext,
                data,
                features,
                config.DataBalancing,
                config.TargetField);

            var count = balancedData.GetRowCount() ?? 0;


            return (balancedData, count);
        }

        private async Task<(IDataView transformedData, string[] selectedFeatures, string report)> ProcessFeatureSelection(
            MLContext mlContext,
            IDataView data,
            string[] features,
            ModelConfig config)
        {
            var featureSelectorFactory = _featureSelectorFactory ?? new FeatureSelectorFactory(mlContext);
            var selector = featureSelectorFactory.CreateSelector(config.FeatureEngineering.Method);

            var result = await selector.SelectFeatures(
                mlContext,
                data,
                features,
                config.ModelType,
                config.TargetField,
                config.FeatureEngineering);

            Console.WriteLine(result.report);
            return result;
        }
    }
}