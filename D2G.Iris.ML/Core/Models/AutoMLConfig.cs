using System;
using System.Collections.Generic;

namespace D2G.Iris.ML.Core.Models
{
    public class AutoMLConfig
    {
        public bool Enabled { get; set; }
        public int MaxExperimentTimeInSeconds { get; set; }
        public int MaxModels { get; set; }
        public string OptimizingMetric { get; set; }
        public Dictionary<string, bool> AllowedTrainers { get; set; }
        public bool CacheDirectoryCleanupEnabled { get; set; }
    }
}