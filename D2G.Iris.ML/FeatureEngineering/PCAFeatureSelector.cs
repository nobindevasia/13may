using System;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;

namespace D2G.Iris.ML.FeatureEngineering
{
    public class PCAFeatureSelector : BaseFeatureSelector
    {
        public PCAFeatureSelector(MLContext mlContext)
            : base(mlContext)
        {
        }

        private class FeatureVector
        {
            [VectorType]
            public float[] Features { get; set; }
            public long Label { get; set; }
        }

        private class PcaInputRow
        {
            public long Label { get; set; }
            public float[] FeaturesArray { get; set; }
        }

        private class PcaOutputRow
        {
            [VectorType]
            public float[] Features { get; set; }
            public long Label { get; set; }
        }

        public override async Task<(IDataView transformedData, string[] selectedFeatures, string report)> SelectFeatures(
            MLContext mlContext,
            IDataView data,
            string[] candidateFeatures,
            ModelType modelType,
            string targetField,
            FeatureEngineeringConfig config)
        {
            InitializeReport("PCA");

            try
            {
                ValidatePcaConfiguration(config, candidateFeatures.Length);
                int numberOfComponents = config.NumberOfComponents;

                _report.AppendLine($"Applying PCA with {numberOfComponents} components");
                _report.AppendLine($"Original feature count: {candidateFeatures.Length}");

                Console.WriteLine("Extracting data for PCA processing");
                var rows = mlContext.Data.CreateEnumerable<FeatureVector>(
                    data, reuseRowObject: false).ToList();

                if (rows.Count == 0 || rows[0].Features == null)
                {
                    throw new InvalidOperationException("No valid feature data found");
                }

                int featureCount = rows[0].Features.Length;
                Console.WriteLine($"Feature vector dimension: {featureCount}");

                var pcaInputRows = rows.Select(r => new PcaInputRow
                {
                    Label = r.Label,
                    FeaturesArray = r.Features
                }).ToList();

                var schema = SchemaDefinition.Create(typeof(PcaInputRow));
                schema["FeaturesArray"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, featureCount);

                var inputData = mlContext.Data.LoadFromEnumerable(pcaInputRows, schema);

                var pcaPipeline = mlContext.Transforms.NormalizeMinMax("NormalizedFeatures", "FeaturesArray")
                    .Append(mlContext.Transforms.ProjectToPrincipalComponents(
                        outputColumnName: "Features",
                        inputColumnName: "NormalizedFeatures",
                        rank: numberOfComponents));


                Console.WriteLine("Applying ML.NET PCA transform");
                var pcaModel = await Task.Run(() => pcaPipeline.Fit(inputData));
                var pcaData = pcaModel.Transform(inputData);

                var resultRows = mlContext.Data.CreateEnumerable<PcaOutputRow>(
                    pcaData, reuseRowObject: false).ToList();

                var outputData = mlContext.Data.LoadFromEnumerable(resultRows);

                string[] pcaFeatureNames = Enumerable.Range(1, numberOfComponents)
                    .Select(i => $"PCA_Component_{i}")
                    .ToArray();

                _report.AppendLine("\nPCA transformation completed successfully using ML.NET.");
                _report.AppendLine("\nPCA Components:");
                foreach (var name in pcaFeatureNames)
                {
                    _report.AppendLine($"  - {name}");
                }

                AddFeatureSelectionSummary(
                    candidateFeatures.Length,
                    pcaFeatureNames.Length,
                    pcaFeatureNames);

                return (outputData, pcaFeatureNames, _report.ToString());
            }
            catch (Exception ex)
            {
                AddErrorToReport(ex);
                Console.WriteLine($"PCA Feature Selection Error: {ex.Message}");
                Console.WriteLine($"Stack Trace: {ex.StackTrace}");
                throw;
            }
        }

        private void ValidatePcaConfiguration(FeatureEngineeringConfig config, int maxComponents)
        {
            base.ValidateConfiguration(config);

            if (config.NumberOfComponents <= 0 || config.NumberOfComponents > maxComponents)
            {
                _report.AppendLine($"Warning: Invalid number of components ({config.NumberOfComponents}). " +
                                  $"Using {Math.Min(maxComponents, 3)} instead.");
                config.NumberOfComponents = Math.Min(maxComponents, 3);
            }
        }
    }
}