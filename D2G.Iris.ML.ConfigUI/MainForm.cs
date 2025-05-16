using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Windows.Forms;

namespace D2G.Iris.ML.ConfigGenerator
{
    public partial class MainForm : Form
    {
        private ModelConfig _modelConfig;
        private const string DefaultConfigFileName = "modelconfig.json";

        public MainForm()
        {
            InitializeComponent();
            InitializeModelConfig();
            BindModelConfigToUI();
        }

        private void InitializeModelConfig()
        {
            _modelConfig = new ModelConfig
            {
                Author = Environment.UserName,
                Description = "New ML Model Configuration",
                ModelType = Core.Enums.ModelType.BinaryClassification,
                TargetField = string.Empty,
                Database = new DatabaseConfig
                {
                    Server = string.Empty,
                    Database = string.Empty,
                    TableName = string.Empty,
                    OutputTableName = string.Empty,
                    WhereClause = string.Empty
                },
                DataBalancing = new DataBalancingConfig
                {
                    Method = Core.Enums.DataBalanceMethod.None,
                    ExecutionOrder = 1,
                    KNeighbors = 5,
                    UndersamplingRatio = 0.9f,
                    MinorityToMajorityRatio = 0.1f
                },
                FeatureEngineering = new FeatureEngineeringConfig
                {
                    Method = Core.Enums.FeatureSelectionMethod.None,
                    ExecutionOrder = 2,
                    NumberOfComponents = 3,
                    MaxFeatures = 8,
                    MulticollinearityThreshold = 0.7
                },
                AutoML = new AutoMLConfig
                {
                    Enabled = false,
                    MaxExperimentTimeInSeconds = 60,
                    MaxModels = 10,
                    OptimizingMetric = "Accuracy"
                },
                TrainingParameters = new TrainingParameters
                {
                    Algorithm = "fastforest",
                    AlgorithmParameters = new Dictionary<string, object>
                    {
                        { "NumberOfLeaves", 20 },
                        { "NumberOfTrees", 100 }
                    },
                    TestFraction = 0.2
                },
                InputFields = new List<InputField>()
            };
        }

        private void BindModelConfigToUI()
        {
            // General
            txtAuthor.Text = _modelConfig.Author;
            txtDescription.Text = _modelConfig.Description;
            cboModelType.DataSource = Enum.GetValues(typeof(Core.Enums.ModelType));
            cboModelType.SelectedItem = _modelConfig.ModelType;
            txtTargetField.Text = _modelConfig.TargetField;

            // Database
            txtDatabaseServer.Text = _modelConfig.Database.Server;
            txtDatabaseName.Text = _modelConfig.Database.Database;
            txtTableName.Text = _modelConfig.Database.TableName;
            txtOutputTableName.Text = _modelConfig.Database.OutputTableName;
            txtWhereClause.Text = _modelConfig.Database.WhereClause;

            // Data Balancing
            cboBalanceMethod.DataSource = Enum.GetValues(typeof(Core.Enums.DataBalanceMethod));
            cboBalanceMethod.SelectedItem = _modelConfig.DataBalancing.Method;
            numBalanceOrder.Value = _modelConfig.DataBalancing.ExecutionOrder;
            numKNeighbors.Value = _modelConfig.DataBalancing.KNeighbors;
            numUndersamplingRatio.Value = (decimal)_modelConfig.DataBalancing.UndersamplingRatio;
            numMinorityRatio.Value = (decimal)_modelConfig.DataBalancing.MinorityToMajorityRatio;

            // Feature Engineering
            cboFeatureMethod.DataSource = Enum.GetValues(typeof(Core.Enums.FeatureSelectionMethod));
            cboFeatureMethod.SelectedItem = _modelConfig.FeatureEngineering.Method;
            numFeatureOrder.Value = _modelConfig.FeatureEngineering.ExecutionOrder;
            numComponents.Value = _modelConfig.FeatureEngineering.NumberOfComponents;
            numMaxFeatures.Value = _modelConfig.FeatureEngineering.MaxFeatures;
            numMulticollinearity.Value = (decimal)_modelConfig.FeatureEngineering.MulticollinearityThreshold;

            // AutoML
            chkAutoML.Checked = _modelConfig.AutoML.Enabled;
            numExperimentTime.Value = _modelConfig.AutoML.MaxExperimentTimeInSeconds;
            numMaxModels.Value = _modelConfig.AutoML.MaxModels;
            txtOptimizingMetric.Text = _modelConfig.AutoML.OptimizingMetric;

            // Training parameters
            txtAlgorithm.Text = _modelConfig.TrainingParameters.Algorithm;
            numTestFraction.Value = (decimal)_modelConfig.TrainingParameters.TestFraction;

            // Algorithm parameters
            lstAlgorithmParams.Items.Clear();
            foreach (var param in _modelConfig.TrainingParameters.AlgorithmParameters)
            {
                lstAlgorithmParams.Items.Add($"{param.Key}={param.Value}");
            }

            // Input fields
            lstInputFields.Items.Clear();
            foreach (var field in _modelConfig.InputFields)
            {
                lstInputFields.Items.Add($"{field.Name} [{(field.IsEnabled ? "Enabled" : "Disabled")}]");
            }

            UpdateUIState();
        }

        private void UpdateUIState()
        {
            // Data balancing controls
            bool isBalancingEnabled = cboBalanceMethod.SelectedItem != null &&
                (Core.Enums.DataBalanceMethod)cboBalanceMethod.SelectedItem != Core.Enums.DataBalanceMethod.None;
            numKNeighbors.Enabled = isBalancingEnabled;
            numUndersamplingRatio.Enabled = isBalancingEnabled;
            numMinorityRatio.Enabled = isBalancingEnabled;

            // Feature engineering controls
            bool isFeatureEngEnabled = cboFeatureMethod.SelectedItem != null &&
                (Core.Enums.FeatureSelectionMethod)cboFeatureMethod.SelectedItem != Core.Enums.FeatureSelectionMethod.None;

            bool isPcaSelected = cboFeatureMethod.SelectedItem != null &&
                (Core.Enums.FeatureSelectionMethod)cboFeatureMethod.SelectedItem == Core.Enums.FeatureSelectionMethod.PCA;

            bool isCorrelationSelected = cboFeatureMethod.SelectedItem != null &&
                (Core.Enums.FeatureSelectionMethod)cboFeatureMethod.SelectedItem == Core.Enums.FeatureSelectionMethod.Correlation;

            numComponents.Enabled = isFeatureEngEnabled && isPcaSelected;
            numMaxFeatures.Enabled = isFeatureEngEnabled && isCorrelationSelected;
            numMulticollinearity.Enabled = isFeatureEngEnabled && isCorrelationSelected;

            // AutoML controls
            bool isAutoMlEnabled = chkAutoML.Checked;
            numExperimentTime.Enabled = isAutoMlEnabled;
            numMaxModels.Enabled = isAutoMlEnabled;
            txtOptimizingMetric.Enabled = isAutoMlEnabled;

            // Traditional algorithm controls
            txtAlgorithm.Enabled = !isAutoMlEnabled;
            lstAlgorithmParams.Enabled = !isAutoMlEnabled;
            btnAddParam.Enabled = !isAutoMlEnabled;
            btnEditParam.Enabled = !isAutoMlEnabled && lstAlgorithmParams.SelectedIndex >= 0;
            btnDeleteParam.Enabled = !isAutoMlEnabled && lstAlgorithmParams.SelectedIndex >= 0;
        }

        private void SaveToModelConfig()
        {
            // General
            _modelConfig.Author = txtAuthor.Text;
            _modelConfig.Description = txtDescription.Text;
            _modelConfig.ModelType = (Core.Enums.ModelType)cboModelType.SelectedItem;
            _modelConfig.TargetField = txtTargetField.Text;

            // Database
            _modelConfig.Database.Server = txtDatabaseServer.Text;
            _modelConfig.Database.Database = txtDatabaseName.Text;
            _modelConfig.Database.TableName = txtTableName.Text;
            _modelConfig.Database.OutputTableName = txtOutputTableName.Text;
            _modelConfig.Database.WhereClause = txtWhereClause.Text;

            // Data Balancing
            _modelConfig.DataBalancing.Method = (Core.Enums.DataBalanceMethod)cboBalanceMethod.SelectedItem;
            _modelConfig.DataBalancing.ExecutionOrder = (int)numBalanceOrder.Value;
            _modelConfig.DataBalancing.KNeighbors = (int)numKNeighbors.Value;
            _modelConfig.DataBalancing.UndersamplingRatio = (float)numUndersamplingRatio.Value;
            _modelConfig.DataBalancing.MinorityToMajorityRatio = (float)numMinorityRatio.Value;

            // Feature Engineering
            _modelConfig.FeatureEngineering.Method = (Core.Enums.FeatureSelectionMethod)cboFeatureMethod.SelectedItem;
            _modelConfig.FeatureEngineering.ExecutionOrder = (int)numFeatureOrder.Value;
            _modelConfig.FeatureEngineering.NumberOfComponents = (int)numComponents.Value;
            _modelConfig.FeatureEngineering.MaxFeatures = (int)numMaxFeatures.Value;
            _modelConfig.FeatureEngineering.MulticollinearityThreshold = (double)numMulticollinearity.Value;

            // AutoML
            _modelConfig.AutoML.Enabled = chkAutoML.Checked;
            _modelConfig.AutoML.MaxExperimentTimeInSeconds = (int)numExperimentTime.Value;
            _modelConfig.AutoML.MaxModels = (int)numMaxModels.Value;
            _modelConfig.AutoML.OptimizingMetric = txtOptimizingMetric.Text;

            // Training parameters
            _modelConfig.TrainingParameters.Algorithm = txtAlgorithm.Text;
            _modelConfig.TrainingParameters.TestFraction = (double)numTestFraction.Value;
        }

        private void SaveModelConfig()
        {
            SaveToModelConfig();

            using (SaveFileDialog saveDialog = new SaveFileDialog())
            {
                saveDialog.Filter = "JSON Files|*.json";
                saveDialog.Title = "Save Model Configuration";
                saveDialog.FileName = DefaultConfigFileName;

                if (saveDialog.ShowDialog() == DialogResult.OK)
                {
                    try
                    {
                        var rootConfig = new Dictionary<string, ModelConfig>
                        {
                            { "modelConfig", _modelConfig }
                        };

                        var options = new JsonSerializerOptions
                        {
                            WriteIndented = true,
                            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
                            Converters = { new JsonStringEnumConverter() }
                        };

                        string json = JsonSerializer.Serialize(rootConfig, options);
                        File.WriteAllText(saveDialog.FileName, json);

                        MessageBox.Show($"Configuration saved to {saveDialog.FileName}", "Save Successful",
                            MessageBoxButtons.OK, MessageBoxIcon.Information);
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show($"Error saving configuration: {ex.Message}", "Save Error",
                            MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
                }
            }
        }

        private void LoadModelConfig()
        {
            using (OpenFileDialog openDialog = new OpenFileDialog())
            {
                openDialog.Filter = "JSON Files|*.json";
                openDialog.Title = "Load Model Configuration";

                if (openDialog.ShowDialog() == DialogResult.OK)
                {
                    try
                    {
                        string json = File.ReadAllText(openDialog.FileName);

                        var options = new JsonSerializerOptions
                        {
                            PropertyNameCaseInsensitive = true,
                            Converters = { new JsonStringEnumConverter() }
                        };

                        var rootConfig = JsonSerializer.Deserialize<Dictionary<string, ModelConfig>>(json, options);

                        if (rootConfig.TryGetValue("modelConfig", out var config))
                        {
                            _modelConfig = config;
                            BindModelConfigToUI();
                            MessageBox.Show("Configuration loaded successfully.", "Load Successful",
                                MessageBoxButtons.OK, MessageBoxIcon.Information);
                        }
                        else
                        {
                            MessageBox.Show("Invalid configuration file format.", "Load Error",
                                MessageBoxButtons.OK, MessageBoxIcon.Error);
                        }
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show($"Error loading configuration: {ex.Message}", "Load Error",
                            MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
                }
            }
        }

        private void btnSave_Click(object sender, EventArgs e)
        {
            SaveModelConfig();
        }

        private void btnLoad_Click(object sender, EventArgs e)
        {
            LoadModelConfig();
        }

        private void AddInputField()
        {
            using (InputFieldDialog dialog = new InputFieldDialog())
            {
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    var field = new InputField
                    {
                        Name = dialog.FieldName,
                        IsEnabled = dialog.IsEnabled
                    };

                    _modelConfig.InputFields.Add(field);
                    lstInputFields.Items.Add($"{field.Name} [{(field.IsEnabled ? "Enabled" : "Disabled")}]");
                }
            }
        }

        private void EditInputField()
        {
            if (lstInputFields.SelectedIndex >= 0)
            {
                int index = lstInputFields.SelectedIndex;
                var field = _modelConfig.InputFields[index];

                using (InputFieldDialog dialog = new InputFieldDialog(field.Name, field.IsEnabled))
                {
                    if (dialog.ShowDialog() == DialogResult.OK)
                    {
                        field.Name = dialog.FieldName;
                        field.IsEnabled = dialog.IsEnabled;

                        lstInputFields.Items[index] = $"{field.Name} [{(field.IsEnabled ? "Enabled" : "Disabled")}]";
                    }
                }
            }
        }

        private void DeleteInputField()
        {
            if (lstInputFields.SelectedIndex >= 0)
            {
                int index = lstInputFields.SelectedIndex;

                if (MessageBox.Show("Are you sure you want to delete this field?", "Confirm Delete",
                    MessageBoxButtons.YesNo, MessageBoxIcon.Question) == DialogResult.Yes)
                {
                    _modelConfig.InputFields.RemoveAt(index);
                    lstInputFields.Items.RemoveAt(index);
                }
            }
        }

        private void AddAlgorithmParameter()
        {
            using (ParameterDialog dialog = new ParameterDialog())
            {
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    _modelConfig.TrainingParameters.AlgorithmParameters[dialog.ParameterName] = dialog.ParameterValue;
                    lstAlgorithmParams.Items.Add($"{dialog.ParameterName}={dialog.ParameterValue}");
                }
            }
        }

        private void EditAlgorithmParameter()
        {
            if (lstAlgorithmParams.SelectedIndex >= 0)
            {
                string item = lstAlgorithmParams.SelectedItem.ToString();
                string[] parts = item.Split('=');

                if (parts.Length == 2)
                {
                    string key = parts[0];
                    object value = _modelConfig.TrainingParameters.AlgorithmParameters[key];

                    using (ParameterDialog dialog = new ParameterDialog(key, value.ToString()))
                    {
                        if (dialog.ShowDialog() == DialogResult.OK)
                        {
                            // Remove old parameter
                            _modelConfig.TrainingParameters.AlgorithmParameters.Remove(key);

                            // Add new parameter with potentially changed name
                            _modelConfig.TrainingParameters.AlgorithmParameters[dialog.ParameterName] = dialog.ParameterValue;

                            // Update list
                            lstAlgorithmParams.Items[lstAlgorithmParams.SelectedIndex] =
                                $"{dialog.ParameterName}={dialog.ParameterValue}";
                        }
                    }
                }
            }
        }

        private void DeleteAlgorithmParameter()
        {
            if (lstAlgorithmParams.SelectedIndex >= 0)
            {
                string item = lstAlgorithmParams.SelectedItem.ToString();
                string[] parts = item.Split('=');

                if (parts.Length == 2)
                {
                    string key = parts[0];

                    if (MessageBox.Show($"Delete parameter '{key}'?", "Confirm Delete",
                        MessageBoxButtons.YesNo, MessageBoxIcon.Question) == DialogResult.Yes)
                    {
                        _modelConfig.TrainingParameters.AlgorithmParameters.Remove(key);
                        lstAlgorithmParams.Items.RemoveAt(lstAlgorithmParams.SelectedIndex);
                    }
                }
            }
        }

        private void btnAddInputField_Click(object sender, EventArgs e)
        {
            AddInputField();
        }

        private void btnEditInputField_Click(object sender, EventArgs e)
        {
            EditInputField();
        }

        private void btnDeleteInputField_Click(object sender, EventArgs e)
        {
            DeleteInputField();
        }

        private void btnAddParam_Click(object sender, EventArgs e)
        {
            AddAlgorithmParameter();
        }

        private void btnEditParam_Click(object sender, EventArgs e)
        {
            EditAlgorithmParameter();
        }

        private void btnDeleteParam_Click(object sender, EventArgs e)
        {
            DeleteAlgorithmParameter();
        }

        private void cboBalanceMethod_SelectedIndexChanged(object sender, EventArgs e)
        {
            UpdateUIState();
        }

        private void cboFeatureMethod_SelectedIndexChanged(object sender, EventArgs e)
        {
            UpdateUIState();
        }

        private void chkAutoML_CheckedChanged(object sender, EventArgs e)
        {
            UpdateUIState();
        }

        private void lstInputFields_SelectedIndexChanged(object sender, EventArgs e)
        {
            btnEditInputField.Enabled = lstInputFields.SelectedIndex >= 0;
            btnDeleteInputField.Enabled = lstInputFields.SelectedIndex >= 0;
        }

        private void lstAlgorithmParams_SelectedIndexChanged(object sender, EventArgs e)
        {
            btnEditParam.Enabled = !chkAutoML.Checked && lstAlgorithmParams.SelectedIndex >= 0;
            btnDeleteParam.Enabled = !chkAutoML.Checked && lstAlgorithmParams.SelectedIndex >= 0;
        }

        private void lstInputFields_DoubleClick(object sender, EventArgs e)
        {
            if (lstInputFields.SelectedIndex >= 0)
            {
                EditInputField();
            }
        }

        private void lstAlgorithmParams_DoubleClick(object sender, EventArgs e)
        {
            if (lstAlgorithmParams.SelectedIndex >= 0 && !chkAutoML.Checked)
            {
                EditAlgorithmParameter();
            }
        }
    }

    public class InputFieldDialog : Form
    {
        private TextBox txtFieldName;
        private CheckBox chkEnabled;
        private Button btnOK;
        private Button btnCancel;

        public string FieldName { get; private set; }
        public bool IsEnabled { get; private set; }

        public InputFieldDialog(string fieldName = "", bool isEnabled = true)
        {
            FieldName = fieldName;
            IsEnabled = isEnabled;
            InitializeComponent();
        }

        private void InitializeComponent()
        {
            this.txtFieldName = new TextBox();
            this.chkEnabled = new CheckBox();
            this.btnOK = new Button();
            this.btnCancel = new Button();
            this.SuspendLayout();

            // txtFieldName
            this.txtFieldName.Location = new Point(12, 25);
            this.txtFieldName.Name = "txtFieldName";
            this.txtFieldName.Size = new Size(260, 23);
            this.txtFieldName.TabIndex = 0;
            this.txtFieldName.Text = FieldName;

            // chkEnabled
            this.chkEnabled.AutoSize = true;
            this.chkEnabled.Location = new Point(12, 54);
            this.chkEnabled.Name = "chkEnabled";
            this.chkEnabled.Size = new Size(68, 19);
            this.chkEnabled.TabIndex = 1;
            this.chkEnabled.Text = "Enabled";
            this.chkEnabled.UseVisualStyleBackColor = true;
            this.chkEnabled.Checked = IsEnabled;

            // btnOK
            this.btnOK.DialogResult = DialogResult.OK;
            this.btnOK.Location = new Point(116, 90);
            this.btnOK.Name = "btnOK";
            this.btnOK.Size = new Size(75, 23);
            this.btnOK.TabIndex = 2;
            this.btnOK.Text = "OK";
            this.btnOK.UseVisualStyleBackColor = true;
            this.btnOK.Click += new EventHandler(this.btnOK_Click);

            // btnCancel
            this.btnCancel.DialogResult = DialogResult.Cancel;
            this.btnCancel.Location = new Point(197, 90);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new Size(75, 23);
            this.btnCancel.TabIndex = 3;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.UseVisualStyleBackColor = true;

            // InputFieldDialog
            this.AcceptButton = this.btnOK;
            this.CancelButton = this.btnCancel;
            this.ClientSize = new Size(284, 125);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOK);
            this.Controls.Add(this.chkEnabled);
            this.Controls.Add(this.txtFieldName);
            this.FormBorderStyle = FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "InputFieldDialog";
            this.StartPosition = FormStartPosition.CenterParent;
            this.Text = "Input Field";
            this.ResumeLayout(false);
            this.PerformLayout();
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            if (string.IsNullOrWhiteSpace(txtFieldName.Text))
            {
                MessageBox.Show("Field name cannot be empty.", "Validation Error",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
                DialogResult = DialogResult.None;
                return;
            }

            FieldName = txtFieldName.Text;
            IsEnabled = chkEnabled.Checked;
        }
    }

    public class ParameterDialog : Form
    {
        private TextBox txtParamName;
        private TextBox txtParamValue;
        private Button btnOK;
        private Button btnCancel;

        public string ParameterName { get; private set; }
        public object ParameterValue { get; private set; }

        public ParameterDialog(string paramName = "", string paramValue = "")
        {
            ParameterName = paramName;
            ParameterValue = paramValue;
            InitializeComponent();
        }

        private void InitializeComponent()
        {
            this.txtParamName = new TextBox();
            this.txtParamValue = new TextBox();
            this.btnOK = new Button();
            this.btnCancel = new Button();
            this.SuspendLayout();

            // Labels
            Label lblName = new Label();
            lblName.AutoSize = true;
            lblName.Location = new Point(12, 9);
            lblName.Name = "lblName";
            lblName.Size = new Size(100, 15);
            lblName.Text = "Parameter Name:";

            Label lblValue = new Label();
            lblValue.AutoSize = true;
            lblValue.Location = new Point(12, 54);
            lblValue.Name = "lblValue";
            lblValue.Size = new Size(100, 15);
            lblValue.Text = "Parameter Value:";

            // txtParamName
            this.txtParamName.Location = new Point(12, 27);
            this.txtParamName.Name = "txtParamName";
            this.txtParamName.Size = new Size(260, 23);
            this.txtParamName.TabIndex = 0;
            this.txtParamName.Text = ParameterName?.ToString();

            // txtParamValue
            this.txtParamValue.Location = new Point(12, 72);
            this.txtParamValue.Name = "txtParamValue";
            this.txtParamValue.Size = new Size(260, 23);
            this.txtParamValue.TabIndex = 1;
            this.txtParamValue.Text = ParameterValue?.ToString();

            // btnOK
            this.btnOK.DialogResult = DialogResult.OK;
            this.btnOK.Location = new Point(116, 108);
            this.btnOK.Name = "btnOK";
            this.btnOK.Size = new Size(75, 23);
            this.btnOK.TabIndex = 2;
            this.btnOK.Text = "OK";
            this.btnOK.UseVisualStyleBackColor = true;
            this.btnOK.Click += new EventHandler(this.btnOK_Click);

            // btnCancel
            this.btnCancel.DialogResult = DialogResult.Cancel;
            this.btnCancel.Location = new Point(197, 108);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new Size(75, 23);
            this.btnCancel.TabIndex = 3;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.UseVisualStyleBackColor = true;

            // ParameterDialog
            this.AcceptButton = this.btnOK;
            this.CancelButton = this.btnCancel;
            this.ClientSize = new Size(284, 143);
            this.Controls.Add(lblName);
            this.Controls.Add(lblValue);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOK);
            this.Controls.Add(this.txtParamValue);
            this.Controls.Add(this.txtParamName);
            this.FormBorderStyle = FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "ParameterDialog";
            this.StartPosition = FormStartPosition.CenterParent;
            this.Text = "Algorithm Parameter";
            this.ResumeLayout(false);
            this.PerformLayout();
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            if (string.IsNullOrWhiteSpace(txtParamName.Text))
            {
                MessageBox.Show("Parameter name cannot be empty.", "Validation Error",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
                DialogResult = DialogResult.None;
                return;
            }

            ParameterName = txtParamName.Text;
            string valueText = txtParamValue.Text;

            // Try to convert to appropriate type
            if (int.TryParse(valueText, out int intValue))
            {
                ParameterValue = intValue;
            }
            else if (double.TryParse(valueText, out double doubleValue))
            {
                ParameterValue = doubleValue;
            }
            else if (bool.TryParse(valueText, out bool boolValue))
            {
                ParameterValue = boolValue;
            }
            else
            {
                ParameterValue = valueText;
            }
        }
    }

    // Form designer code would go here - simplified for brevity
    partial class MainForm
    {
        private void InitializeComponent()
        {
            this.tabControl = new TabControl();
            this.tabGeneral = new TabPage();
            this.tabDatabase = new TabPage();
            this.tabDataBalancing = new TabPage();
            this.tabFeatureEngineering = new TabPage();
            this.tabAutoML = new TabPage();
            this.tabTraining = new TabPage();
            this.tabInputFields = new TabPage();

            // General controls
            this.lblAuthor = new Label();
            this.txtAuthor = new TextBox();
            this.lblDescription = new Label();
            this.txtDescription = new TextBox();
            this.lblModelType = new Label();
            this.cboModelType = new ComboBox();
            this.lblTargetField = new Label();
            this.txtTargetField = new TextBox();

            // Database controls
            this.lblServer = new Label();
            this.txtDatabaseServer = new TextBox();
            this.lblDatabase = new Label();
            this.txtDatabaseName = new TextBox();
            this.lblTableName = new Label();
            this.txtTableName = new TextBox();
            this.lblOutputTable = new Label();
            this.txtOutputTableName = new TextBox();
            this.lblWhereClause = new Label();
            this.txtWhereClause = new TextBox();

            // Data Balancing controls
            this.lblBalanceMethod = new Label();
            this.cboBalanceMethod = new ComboBox();
            this.lblBalanceOrder = new Label();
            this.numBalanceOrder = new NumericUpDown();
            this.lblKNeighbors = new Label();
            this.numKNeighbors = new NumericUpDown();
            this.lblUndersamplingRatio = new Label();
            this.numUndersamplingRatio = new NumericUpDown();
            this.lblMinorityRatio = new Label();
            this.numMinorityRatio = new NumericUpDown();

            // Feature Engineering controls
            this.lblFeatureMethod = new Label();
            this.cboFeatureMethod = new ComboBox();
            this.lblFeatureOrder = new Label();
            this.numFeatureOrder = new NumericUpDown();
            this.lblComponents = new Label();
            this.numComponents = new NumericUpDown();
            this.lblMaxFeatures = new Label();
            this.numMaxFeatures = new NumericUpDown();
            this.lblMulticollinearity = new Label();
            this.numMulticollinearity = new NumericUpDown();

            // AutoML controls
            this.chkAutoML = new CheckBox();
            this.lblExperimentTime = new Label();
            this.numExperimentTime = new NumericUpDown();
            this.lblMaxModels = new Label();
            this.numMaxModels = new NumericUpDown();
            this.lblOptimizingMetric = new Label();
            this.txtOptimizingMetric = new TextBox();

            // Training controls
            this.lblAlgorithm = new Label();
            this.txtAlgorithm = new TextBox();
            this.lblTestFraction = new Label();
            this.numTestFraction = new NumericUpDown();
            this.lblAlgorithmParams = new Label();
            this.lstAlgorithmParams = new ListBox();
            this.btnAddParam = new Button();
            this.btnEditParam = new Button();
            this.btnDeleteParam = new Button();

            // Input Fields controls
            this.lstInputFields = new ListBox();
            this.btnAddInputField = new Button();
            this.btnEditInputField = new Button();
            this.btnDeleteInputField = new Button();

            // Main form controls
            this.btnSave = new Button();
            this.btnLoad = new Button();

            // Initialize the form
            this.SuspendLayout();

            // Set up the tab control
            this.tabControl.Dock = DockStyle.Fill;
            this.tabControl.Location = new Point(0, 0);
            this.tabControl.Name = "tabControl";
            this.tabControl.SelectedIndex = 0;
            this.tabControl.Size = new Size(684, 461);
            this.tabControl.TabIndex = 0;

            // Set up the tabs
            this.tabControl.Controls.Add(this.tabGeneral);
            this.tabControl.Controls.Add(this.tabDatabase);
            this.tabControl.Controls.Add(this.tabDataBalancing);
            this.tabControl.Controls.Add(this.tabFeatureEngineering);
            this.tabControl.Controls.Add(this.tabAutoML);
            this.tabControl.Controls.Add(this.tabTraining);
            this.tabControl.Controls.Add(this.tabInputFields);

            // Configure tab pages
            this.tabGeneral.Location = new Point(4, 24);
            this.tabGeneral.Name = "tabGeneral";
            this.tabGeneral.Padding = new Padding(3);
            this.tabGeneral.Size = new Size(676, 433);
            this.tabGeneral.TabIndex = 0;
            this.tabGeneral.Text = "General";
            this.tabGeneral.UseVisualStyleBackColor = true;

            this.tabDatabase.Location = new Point(4, 24);
            this.tabDatabase.Name = "tabDatabase";
            this.tabDatabase.Padding = new Padding(3);
            this.tabDatabase.Size = new Size(676, 433);
            this.tabDatabase.TabIndex = 1;
            this.tabDatabase.Text = "Database";
            this.tabDatabase.UseVisualStyleBackColor = true;

            this.tabDataBalancing.Location = new Point(4, 24);
            this.tabDataBalancing.Name = "tabDataBalancing";
            this.tabDataBalancing.Size = new Size(676, 433);
            this.tabDataBalancing.TabIndex = 2;
            this.tabDataBalancing.Text = "Data Balancing";
            this.tabDataBalancing.UseVisualStyleBackColor = true;

            this.tabFeatureEngineering.Location = new Point(4, 24);
            this.tabFeatureEngineering.Name = "tabFeatureEngineering";
            this.tabFeatureEngineering.Size = new Size(676, 433);
            this.tabFeatureEngineering.TabIndex = 3;
            this.tabFeatureEngineering.Text = "Feature Engineering";
            this.tabFeatureEngineering.UseVisualStyleBackColor = true;

            this.tabAutoML.Location = new Point(4, 24);
            this.tabAutoML.Name = "tabAutoML";
            this.tabAutoML.Size = new Size(676, 433);
            this.tabAutoML.TabIndex = 4;
            this.tabAutoML.Text = "AutoML";
            this.tabAutoML.UseVisualStyleBackColor = true;

            this.tabTraining.Location = new Point(4, 24);
            this.tabTraining.Name = "tabTraining";
            this.tabTraining.Size = new Size(676, 433);
            this.tabTraining.TabIndex = 5;
            this.tabTraining.Text = "Training";
            this.tabTraining.UseVisualStyleBackColor = true;

            this.tabInputFields.Location = new Point(4, 24);
            this.tabInputFields.Name = "tabInputFields";
            this.tabInputFields.Size = new Size(676, 433);
            this.tabInputFields.TabIndex = 6;
            this.tabInputFields.Text = "Input Fields";
            this.tabInputFields.UseVisualStyleBackColor = true;

            // Set up the General tab
            this.tabGeneral.Controls.Add(this.lblAuthor);
            this.tabGeneral.Controls.Add(this.txtAuthor);
            this.tabGeneral.Controls.Add(this.lblDescription);
            this.tabGeneral.Controls.Add(this.txtDescription);
            this.tabGeneral.Controls.Add(this.lblModelType);
            this.tabGeneral.Controls.Add(this.cboModelType);
            this.tabGeneral.Controls.Add(this.lblTargetField);
            this.tabGeneral.Controls.Add(this.txtTargetField);

            this.lblAuthor.AutoSize = true;
            this.lblAuthor.Location = new Point(20, 20);
            this.lblAuthor.Name = "lblAuthor";
            this.lblAuthor.Size = new Size(47, 15);
            this.lblAuthor.TabIndex = 0;
            this.lblAuthor.Text = "Author:";

            this.txtAuthor.Location = new Point(150, 17);
            this.txtAuthor.Name = "txtAuthor";
            this.txtAuthor.Size = new Size(300, 23);
            this.txtAuthor.TabIndex = 1;

            this.lblDescription.AutoSize = true;
            this.lblDescription.Location = new Point(20, 50);
            this.lblDescription.Name = "lblDescription";
            this.lblDescription.Size = new Size(70, 15);
            this.lblDescription.TabIndex = 2;
            this.lblDescription.Text = "Description:";

            this.txtDescription.Location = new Point(150, 47);
            this.txtDescription.Multiline = true;
            this.txtDescription.Name = "txtDescription";
            this.txtDescription.Size = new Size(300, 60);
            this.txtDescription.TabIndex = 3;

            this.lblModelType.AutoSize = true;
            this.lblModelType.Location = new Point(20, 120);
            this.lblModelType.Name = "lblModelType";
            this.lblModelType.Size = new Size(74, 15);
            this.lblModelType.TabIndex = 4;
            this.lblModelType.Text = "Model Type:";

            this.cboModelType.DropDownStyle = ComboBoxStyle.DropDownList;
            this.cboModelType.FormattingEnabled = true;
            this.cboModelType.Location = new Point(150, 117);
            this.cboModelType.Name = "cboModelType";
            this.cboModelType.Size = new Size(300, 23);
            this.cboModelType.TabIndex = 5;

            this.lblTargetField.AutoSize = true;
            this.lblTargetField.Location = new Point(20, 150);
            this.lblTargetField.Name = "lblTargetField";
            this.lblTargetField.Size = new Size(73, 15);
            this.lblTargetField.TabIndex = 6;
            this.lblTargetField.Text = "Target Field:";

            this.txtTargetField.Location = new Point(150, 147);
            this.txtTargetField.Name = "txtTargetField";
            this.txtTargetField.Size = new Size(300, 23);
            this.txtTargetField.TabIndex = 7;

            // Set up the Database tab
            this.tabDatabase.Controls.Add(this.lblServer);
            this.tabDatabase.Controls.Add(this.txtDatabaseServer);
            this.tabDatabase.Controls.Add(this.lblDatabase);
            this.tabDatabase.Controls.Add(this.txtDatabaseName);
            this.tabDatabase.Controls.Add(this.lblTableName);
            this.tabDatabase.Controls.Add(this.txtTableName);
            this.tabDatabase.Controls.Add(this.lblOutputTable);
            this.tabDatabase.Controls.Add(this.txtOutputTableName);
            this.tabDatabase.Controls.Add(this.lblWhereClause);
            this.tabDatabase.Controls.Add(this.txtWhereClause);

            this.lblServer.AutoSize = true;
            this.lblServer.Location = new Point(20, 20);
            this.lblServer.Name = "lblServer";
            this.lblServer.Size = new Size(42, 15);
            this.lblServer.TabIndex = 0;
            this.lblServer.Text = "Server:";

            this.txtDatabaseServer.Location = new Point(150, 17);
            this.txtDatabaseServer.Name = "txtDatabaseServer";
            this.txtDatabaseServer.Size = new Size(300, 23);
            this.txtDatabaseServer.TabIndex = 1;

            this.lblDatabase.AutoSize = true;
            this.lblDatabase.Location = new Point(20, 50);
            this.lblDatabase.Name = "lblDatabase";
            this.lblDatabase.Size = new Size(58, 15);
            this.lblDatabase.TabIndex = 2;
            this.lblDatabase.Text = "Database:";

            this.txtDatabaseName.Location = new Point(150, 47);
            this.txtDatabaseName.Name = "txtDatabaseName";
            this.txtDatabaseName.Size = new Size(300, 23);
            this.txtDatabaseName.TabIndex = 3;

            this.lblTableName.AutoSize = true;
            this.lblTableName.Location = new Point(20, 80);
            this.lblTableName.Name = "lblTableName";
            this.lblTableName.Size = new Size(73, 15);
            this.lblTableName.TabIndex = 4;
            this.lblTableName.Text = "Table Name:";

            this.txtTableName.Location = new Point(150, 77);
            this.txtTableName.Name = "txtTableName";
            this.txtTableName.Size = new Size(300, 23);
            this.txtTableName.TabIndex = 5;

            this.lblOutputTable.AutoSize = true;
            this.lblOutputTable.Location = new Point(20, 110);
            this.lblOutputTable.Name = "lblOutputTable";
            this.lblOutputTable.Size = new Size(81, 15);
            this.lblOutputTable.TabIndex = 6;
            this.lblOutputTable.Text = "Output Table:";

            this.txtOutputTableName.Location = new Point(150, 107);
            this.txtOutputTableName.Name = "txtOutputTableName";
            this.txtOutputTableName.Size = new Size(300, 23);
            this.txtOutputTableName.TabIndex = 7;

            this.lblWhereClause.AutoSize = true;
            this.lblWhereClause.Location = new Point(20, 140);
            this.lblWhereClause.Name = "lblWhereClause";
            this.lblWhereClause.Size = new Size(84, 15);
            this.lblWhereClause.TabIndex = 8;
            this.lblWhereClause.Text = "Where Clause:";

            this.txtWhereClause.Location = new Point(150, 137);
            this.txtWhereClause.Multiline = true;
            this.txtWhereClause.Name = "txtWhereClause";
            this.txtWhereClause.Size = new Size(300, 60);
            this.txtWhereClause.TabIndex = 9;

            // Set up the Data Balancing tab
            this.tabDataBalancing.Controls.Add(this.lblBalanceMethod);
            this.tabDataBalancing.Controls.Add(this.cboBalanceMethod);
            this.tabDataBalancing.Controls.Add(this.lblBalanceOrder);
            this.tabDataBalancing.Controls.Add(this.numBalanceOrder);
            this.tabDataBalancing.Controls.Add(this.lblKNeighbors);
            this.tabDataBalancing.Controls.Add(this.numKNeighbors);
            this.tabDataBalancing.Controls.Add(this.lblUndersamplingRatio);
            this.tabDataBalancing.Controls.Add(this.numUndersamplingRatio);
            this.tabDataBalancing.Controls.Add(this.lblMinorityRatio);
            this.tabDataBalancing.Controls.Add(this.numMinorityRatio);

            this.lblBalanceMethod.AutoSize = true;
            this.lblBalanceMethod.Location = new Point(20, 20);
            this.lblBalanceMethod.Name = "lblBalanceMethod";
            this.lblBalanceMethod.Size = new Size(52, 15);
            this.lblBalanceMethod.TabIndex = 0;
            this.lblBalanceMethod.Text = "Method:";

            this.cboBalanceMethod.DropDownStyle = ComboBoxStyle.DropDownList;
            this.cboBalanceMethod.FormattingEnabled = true;
            this.cboBalanceMethod.Location = new Point(150, 17);
            this.cboBalanceMethod.Name = "cboBalanceMethod";
            this.cboBalanceMethod.Size = new Size(300, 23);
            this.cboBalanceMethod.TabIndex = 1;
            this.cboBalanceMethod.SelectedIndexChanged += new EventHandler(this.cboBalanceMethod_SelectedIndexChanged);

            this.lblBalanceOrder.AutoSize = true;
            this.lblBalanceOrder.Location = new Point(20, 50);
            this.lblBalanceOrder.Name = "lblBalanceOrder";
            this.lblBalanceOrder.Size = new Size(96, 15);
            this.lblBalanceOrder.TabIndex = 2;
            this.lblBalanceOrder.Text = "Execution Order:";

            this.numBalanceOrder.Location = new Point(150, 47);
            this.numBalanceOrder.Minimum = new decimal(new int[] { 1, 0, 0, 0 });
            this.numBalanceOrder.Maximum = new decimal(new int[] { 10, 0, 0, 0 });
            this.numBalanceOrder.Name = "numBalanceOrder";
            this.numBalanceOrder.Size = new Size(100, 23);
            this.numBalanceOrder.TabIndex = 3;
            this.numBalanceOrder.Value = new decimal(new int[] { 1, 0, 0, 0 });

            this.lblKNeighbors.AutoSize = true;
            this.lblKNeighbors.Location = new Point(20, 80);
            this.lblKNeighbors.Name = "lblKNeighbors";
            this.lblKNeighbors.Size = new Size(77, 15);
            this.lblKNeighbors.TabIndex = 4;
            this.lblKNeighbors.Text = "K Neighbors:";

            this.numKNeighbors.Location = new Point(150, 77);
            this.numKNeighbors.Minimum = new decimal(new int[] { 1, 0, 0, 0 });
            this.numKNeighbors.Maximum = new decimal(new int[] { 20, 0, 0, 0 });
            this.numKNeighbors.Name = "numKNeighbors";
            this.numKNeighbors.Size = new Size(100, 23);
            this.numKNeighbors.TabIndex = 5;
            this.numKNeighbors.Value = new decimal(new int[] { 5, 0, 0, 0 });

            this.lblUndersamplingRatio.AutoSize = true;
            this.lblUndersamplingRatio.Location = new Point(20, 110);
            this.lblUndersamplingRatio.Name = "lblUndersamplingRatio";
            this.lblUndersamplingRatio.Size = new Size(121, 15);
            this.lblUndersamplingRatio.TabIndex = 6;
            this.lblUndersamplingRatio.Text = "Undersampling Ratio:";

            this.numUndersamplingRatio.DecimalPlaces = 2;
            this.numUndersamplingRatio.Increment = new decimal(new int[] { 1, 0, 0, 131072 });
            this.numUndersamplingRatio.Location = new Point(150, 107);
            this.numUndersamplingRatio.Maximum = new decimal(new int[] { 1, 0, 0, 0 });
            this.numUndersamplingRatio.Name = "numUndersamplingRatio";
            this.numUndersamplingRatio.Size = new Size(100, 23);
            this.numUndersamplingRatio.TabIndex = 7;
            this.numUndersamplingRatio.Value = new decimal(new int[] { 9, 0, 0, 65536 });

            this.lblMinorityRatio.AutoSize = true;
            this.lblMinorityRatio.Location = new Point(20, 140);
            this.lblMinorityRatio.Name = "lblMinorityRatio";
            this.lblMinorityRatio.Size = new Size(124, 15);
            this.lblMinorityRatio.TabIndex = 8;
            this.lblMinorityRatio.Text = "Minority/Majority Ratio:";

            this.numMinorityRatio.DecimalPlaces = 2;
            this.numMinorityRatio.Increment = new decimal(new int[] { 1, 0, 0, 131072 });
            this.numMinorityRatio.Location = new Point(150, 137);
            this.numMinorityRatio.Maximum = new decimal(new int[] { 1, 0, 0, 0 });
            this.numMinorityRatio.Name = "numMinorityRatio";
            this.numMinorityRatio.Size = new Size(100, 23);
            this.numMinorityRatio.TabIndex = 9;
            this.numMinorityRatio.Value = new decimal(new int[] { 1, 0, 0, 65536 });

            // Set up the Feature Engineering tab
            this.tabFeatureEngineering.Controls.Add(this.lblFeatureMethod);
            this.tabFeatureEngineering.Controls.Add(this.cboFeatureMethod);
            this.tabFeatureEngineering.Controls.Add(this.lblFeatureOrder);
            this.tabFeatureEngineering.Controls.Add(this.numFeatureOrder);
            this.tabFeatureEngineering.Controls.Add(this.lblComponents);
            this.tabFeatureEngineering.Controls.Add(this.numComponents);
            this.tabFeatureEngineering.Controls.Add(this.lblMaxFeatures);
            this.tabFeatureEngineering.Controls.Add(this.numMaxFeatures);
            this.tabFeatureEngineering.Controls.Add(this.lblMulticollinearity);
            this.tabFeatureEngineering.Controls.Add(this.numMulticollinearity);

            this.lblFeatureMethod.AutoSize = true;
            this.lblFeatureMethod.Location = new Point(20, 20);
            this.lblFeatureMethod.Name = "lblFeatureMethod";
            this.lblFeatureMethod.Size = new Size(52, 15);
            this.lblFeatureMethod.TabIndex = 0;
            this.lblFeatureMethod.Text = "Method:";

            this.cboFeatureMethod.DropDownStyle = ComboBoxStyle.DropDownList;
            this.cboFeatureMethod.FormattingEnabled = true;
            this.cboFeatureMethod.Location = new Point(200, 17);
            this.cboFeatureMethod.Name = "cboFeatureMethod";
            this.cboFeatureMethod.Size = new Size(300, 23);
            this.cboFeatureMethod.TabIndex = 1;
            this.cboFeatureMethod.SelectedIndexChanged += new EventHandler(this.cboFeatureMethod_SelectedIndexChanged);

            this.lblFeatureOrder.AutoSize = true;
            this.lblFeatureOrder.Location = new Point(20, 50);
            this.lblFeatureOrder.Name = "lblFeatureOrder";
            this.lblFeatureOrder.Size = new Size(96, 15);
            this.lblFeatureOrder.TabIndex = 2;
            this.lblFeatureOrder.Text = "Execution Order:";

            this.numFeatureOrder.Location = new Point(200, 47);
            this.numFeatureOrder.Minimum = new decimal(new int[] { 1, 0, 0, 0 });
            this.numFeatureOrder.Maximum = new decimal(new int[] { 10, 0, 0, 0 });
            this.numFeatureOrder.Name = "numFeatureOrder";
            this.numFeatureOrder.Size = new Size(100, 23);
            this.numFeatureOrder.TabIndex = 3;
            this.numFeatureOrder.Value = new decimal(new int[] { 2, 0, 0, 0 });

            this.lblComponents.AutoSize = true;
            this.lblComponents.Location = new Point(20, 80);
            this.lblComponents.Name = "lblComponents";
            this.lblComponents.Size = new Size(137, 15);
            this.lblComponents.TabIndex = 4;
            this.lblComponents.Text = "Number of Components:";

            this.numComponents.Location = new Point(200, 77);
            this.numComponents.Minimum = new decimal(new int[] { 1, 0, 0, 0 });
            this.numComponents.Maximum = new decimal(new int[] { 20, 0, 0, 0 });
            this.numComponents.Name = "numComponents";
            this.numComponents.Size = new Size(100, 23);
            this.numComponents.TabIndex = 5;
            this.numComponents.Value = new decimal(new int[] { 3, 0, 0, 0 });

            this.lblMaxFeatures.AutoSize = true;
            this.lblMaxFeatures.Location = new Point(20, 110);
            this.lblMaxFeatures.Name = "lblMaxFeatures";
            this.lblMaxFeatures.Size = new Size(81, 15);
            this.lblMaxFeatures.TabIndex = 6;
            this.lblMaxFeatures.Text = "Max Features:";

            this.numMaxFeatures.Location = new Point(200, 107);
            this.numMaxFeatures.Minimum = new decimal(new int[] { 1, 0, 0, 0 });
            this.numMaxFeatures.Maximum = new decimal(new int[] { 100, 0, 0, 0 });
            this.numMaxFeatures.Name = "numMaxFeatures";
            this.numMaxFeatures.Size = new Size(100, 23);
            this.numMaxFeatures.TabIndex = 7;
            this.numMaxFeatures.Value = new decimal(new int[] { 8, 0, 0, 0 });

            this.lblMulticollinearity.AutoSize = true;
            this.lblMulticollinearity.Location = new Point(20, 140);
            this.lblMulticollinearity.Name = "lblMulticollinearity";
            this.lblMulticollinearity.Size = new Size(151, 15);
            this.lblMulticollinearity.TabIndex = 8;
            this.lblMulticollinearity.Text = "Multicollinearity Threshold:";

            this.numMulticollinearity.DecimalPlaces = 2;
            this.numMulticollinearity.Increment = new decimal(new int[] { 1, 0, 0, 131072 });
            this.numMulticollinearity.Location = new Point(200, 137);
            this.numMulticollinearity.Maximum = new decimal(new int[] { 1, 0, 0, 0 });
            this.numMulticollinearity.Name = "numMulticollinearity";
            this.numMulticollinearity.Size = new Size(100, 23);
            this.numMulticollinearity.TabIndex = 9;
            this.numMulticollinearity.Value = new decimal(new int[] { 7, 0, 0, 65536 });

            // Set up the AutoML tab
            this.tabAutoML.Controls.Add(this.chkAutoML);
            this.tabAutoML.Controls.Add(this.lblExperimentTime);
            this.tabAutoML.Controls.Add(this.numExperimentTime);
            this.tabAutoML.Controls.Add(this.lblMaxModels);
            this.tabAutoML.Controls.Add(this.numMaxModels);
            this.tabAutoML.Controls.Add(this.lblOptimizingMetric);
            this.tabAutoML.Controls.Add(this.txtOptimizingMetric);

            this.chkAutoML.AutoSize = true;
            this.chkAutoML.Location = new Point(20, 20);
            this.chkAutoML.Name = "chkAutoML";
            this.chkAutoML.Size = new Size(103, 19);
            this.chkAutoML.TabIndex = 0;
            this.chkAutoML.Text = "Enable AutoML";
            this.chkAutoML.UseVisualStyleBackColor = true;
            this.chkAutoML.CheckedChanged += new EventHandler(this.chkAutoML_CheckedChanged);

            this.lblExperimentTime.AutoSize = true;
            this.lblExperimentTime.Location = new Point(40, 50);
            this.lblExperimentTime.Name = "lblExperimentTime";
            this.lblExperimentTime.Size = new Size(180, 15);
            this.lblExperimentTime.TabIndex = 1;
            this.lblExperimentTime.Text = "Max Experiment Time (seconds):";

            this.numExperimentTime.Location = new Point(240, 47);
            this.numExperimentTime.Maximum = new decimal(new int[] { 3600, 0, 0, 0 });
            this.numExperimentTime.Minimum = new decimal(new int[] { 10, 0, 0, 0 });
            this.numExperimentTime.Name = "numExperimentTime";
            this.numExperimentTime.Size = new Size(100, 23);
            this.numExperimentTime.TabIndex = 2;
            this.numExperimentTime.Value = new decimal(new int[] { 60, 0, 0, 0 });

            this.lblMaxModels.AutoSize = true;
            this.lblMaxModels.Location = new Point(40, 80);
            this.lblMaxModels.Name = "lblMaxModels";
            this.lblMaxModels.Size = new Size(75, 15);
            this.lblMaxModels.TabIndex = 3;
            this.lblMaxModels.Text = "Max Models:";

            this.numMaxModels.Location = new Point(240, 77);
            this.numMaxModels.Maximum = new decimal(new int[] { 100, 0, 0, 0 });
            this.numMaxModels.Minimum = new decimal(new int[] { 1, 0, 0, 0 });
            this.numMaxModels.Name = "numMaxModels";
            this.numMaxModels.Size = new Size(100, 23);
            this.numMaxModels.TabIndex = 4;
            this.numMaxModels.Value = new decimal(new int[] { 10, 0, 0, 0 });

            this.lblOptimizingMetric.AutoSize = true;
            this.lblOptimizingMetric.Location = new Point(40, 110);
            this.lblOptimizingMetric.Name = "lblOptimizingMetric";
            this.lblOptimizingMetric.Size = new Size(103, 15);
            this.lblOptimizingMetric.TabIndex = 5;
            this.lblOptimizingMetric.Text = "Optimizing Metric:";

            this.txtOptimizingMetric.Location = new Point(240, 107);
            this.txtOptimizingMetric.Name = "txtOptimizingMetric";
            this.txtOptimizingMetric.Size = new Size(200, 23);
            this.txtOptimizingMetric.TabIndex = 6;

            // Set up the Training tab
            this.tabTraining.Controls.Add(this.lblAlgorithm);
            this.tabTraining.Controls.Add(this.txtAlgorithm);
            this.tabTraining.Controls.Add(this.lblTestFraction);
            this.tabTraining.Controls.Add(this.numTestFraction);
            this.tabTraining.Controls.Add(this.lblAlgorithmParams);
            this.tabTraining.Controls.Add(this.lstAlgorithmParams);
            this.tabTraining.Controls.Add(this.btnAddParam);
            this.tabTraining.Controls.Add(this.btnEditParam);
            this.tabTraining.Controls.Add(this.btnDeleteParam);

            this.lblAlgorithm.AutoSize = true;
            this.lblAlgorithm.Location = new Point(20, 20);
            this.lblAlgorithm.Name = "lblAlgorithm";
            this.lblAlgorithm.Size = new Size(64, 15);
            this.lblAlgorithm.TabIndex = 0;
            this.lblAlgorithm.Text = "Algorithm:";

            this.txtAlgorithm.Location = new Point(150, 17);
            this.txtAlgorithm.Name = "txtAlgorithm";
            this.txtAlgorithm.Size = new Size(300, 23);
            this.txtAlgorithm.TabIndex = 1;

            this.lblTestFraction.AutoSize = true;
            this.lblTestFraction.Location = new Point(20, 50);
            this.lblTestFraction.Name = "lblTestFraction";
            this.lblTestFraction.Size = new Size(84, 15);
            this.lblTestFraction.TabIndex = 2;
            this.lblTestFraction.Text = "Test Fraction:";

            this.numTestFraction.DecimalPlaces = 2;
            this.numTestFraction.Increment = new decimal(new int[] { 1, 0, 0, 131072 });
            this.numTestFraction.Location = new Point(150, 47);
            this.numTestFraction.Maximum = new decimal(new int[] { 5, 0, 0, 65536 });
            this.numTestFraction.Minimum = new decimal(new int[] { 1, 0, 0, 65536 });
            this.numTestFraction.Name = "numTestFraction";
            this.numTestFraction.Size = new Size(100, 23);
            this.numTestFraction.TabIndex = 3;
            this.numTestFraction.Value = new decimal(new int[] { 2, 0, 0, 65536 });

            this.lblAlgorithmParams.AutoSize = true;
            this.lblAlgorithmParams.Location = new Point(20, 80);
            this.lblAlgorithmParams.Name = "lblAlgorithmParams";
            this.lblAlgorithmParams.Size = new Size(124, 15);
            this.lblAlgorithmParams.TabIndex = 4;
            this.lblAlgorithmParams.Text = "Algorithm Parameters:";

            this.lstAlgorithmParams.FormattingEnabled = true;
            this.lstAlgorithmParams.ItemHeight = 15;
            this.lstAlgorithmParams.Location = new Point(20, 98);
            this.lstAlgorithmParams.Name = "lstAlgorithmParams";
            this.lstAlgorithmParams.Size = new Size(430, 229);
            this.lstAlgorithmParams.TabIndex = 5;
            this.lstAlgorithmParams.SelectedIndexChanged += new EventHandler(this.lstAlgorithmParams_SelectedIndexChanged);
            this.lstAlgorithmParams.DoubleClick += new EventHandler(this.lstAlgorithmParams_DoubleClick);

            this.btnAddParam.Location = new Point(20, 333);
            this.btnAddParam.Name = "btnAddParam";
            this.btnAddParam.Size = new Size(75, 23);
            this.btnAddParam.TabIndex = 6;
            this.btnAddParam.Text = "Add";
            this.btnAddParam.UseVisualStyleBackColor = true;
            this.btnAddParam.Click += new EventHandler(this.btnAddParam_Click);

            this.btnEditParam.Location = new Point(101, 333);
            this.btnEditParam.Name = "btnEditParam";
            this.btnEditParam.Size = new Size(75, 23);
            this.btnEditParam.TabIndex = 7;
            this.btnEditParam.Text = "Edit";
            this.btnEditParam.UseVisualStyleBackColor = true;
            this.btnEditParam.Click += new EventHandler(this.btnEditParam_Click);

            this.btnDeleteParam.Location = new Point(182, 333);
            this.btnDeleteParam.Name = "btnDeleteParam";
            this.btnDeleteParam.Size = new Size(75, 23);
            this.btnDeleteParam.TabIndex = 8;
            this.btnDeleteParam.Text = "Delete";
            this.btnDeleteParam.UseVisualStyleBackColor = true;
            this.btnDeleteParam.Click += new EventHandler(this.btnDeleteParam_Click);

            // Set up the Input Fields tab
            this.tabInputFields.Controls.Add(this.lstInputFields);
            this.tabInputFields.Controls.Add(this.btnAddInputField);
            this.tabInputFields.Controls.Add(this.btnEditInputField);
            this.tabInputFields.Controls.Add(this.btnDeleteInputField);

            this.lstInputFields.FormattingEnabled = true;
            this.lstInputFields.ItemHeight = 15;
            this.lstInputFields.Location = new Point(20, 20);
            this.lstInputFields.Name = "lstInputFields";
            this.lstInputFields.Size = new Size(430, 304);
            this.lstInputFields.TabIndex = 0;
            this.lstInputFields.SelectedIndexChanged += new EventHandler(this.lstInputFields_SelectedIndexChanged);
            this.lstInputFields.DoubleClick += new EventHandler(this.lstInputFields_DoubleClick);

            this.btnAddInputField.Location = new Point(20, 330);
            this.btnAddInputField.Name = "btnAddInputField";
            this.btnAddInputField.Size = new Size(75, 23);
            this.btnAddInputField.TabIndex = 1;
            this.btnAddInputField.Text = "Add";
            this.btnAddInputField.UseVisualStyleBackColor = true;
            this.btnAddInputField.Click += new EventHandler(this.btnAddInputField_Click);

            this.btnEditInputField.Location = new Point(101, 330);
            this.btnEditInputField.Name = "btnEditInputField";
            this.btnEditInputField.Size = new Size(75, 23);
            this.btnEditInputField.TabIndex = 2;
            this.btnEditInputField.Text = "Edit";
            this.btnEditInputField.UseVisualStyleBackColor = true;
            this.btnEditInputField.Click += new EventHandler(this.btnEditInputField_Click);

            this.btnDeleteInputField.Location = new Point(182, 330);
            this.btnDeleteInputField.Name = "btnDeleteInputField";
            this.btnDeleteInputField.Size = new Size(75, 23);
            this.btnDeleteInputField.TabIndex = 3;
            this.btnDeleteInputField.Text = "Delete";
            this.btnDeleteInputField.UseVisualStyleBackColor = true;
            this.btnDeleteInputField.Click += new EventHandler(this.btnDeleteInputField_Click);

            // Set up the main form buttons
            this.btnSave = new Button();
            this.btnLoad = new Button();

            this.btnSave.Location = new Point(16, 470);
            this.btnSave.Name = "btnSave";
            this.btnSave.Size = new Size(120, 30);
            this.btnSave.TabIndex = 1;
            this.btnSave.Text = "Save Configuration";
            this.btnSave.UseVisualStyleBackColor = true;
            this.btnSave.Click += new EventHandler(this.btnSave_Click);

            this.btnLoad.Location = new Point(142, 470);
            this.btnLoad.Name = "btnLoad";
            this.btnLoad.Size = new Size(120, 30);
            this.btnLoad.TabIndex = 2;
            this.btnLoad.Text = "Load Configuration";
            this.btnLoad.UseVisualStyleBackColor = true;
            this.btnLoad.Click += new EventHandler(this.btnLoad_Click);

            // MainForm
            this.AutoScaleDimensions = new SizeF(7F, 15F);
            this.AutoScaleMode = AutoScaleMode.Font;
            this.ClientSize = new Size(684, 511);
            this.Controls.Add(this.tabControl);
            this.Controls.Add(this.btnSave);
            this.Controls.Add(this.btnLoad);
            this.FormBorderStyle = FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.Name = "MainForm";
            this.StartPosition = FormStartPosition.CenterScreen;
            this.Text = "D2G.Iris.ML Config Generator";
            this.ResumeLayout(false);
            this.PerformLayout();
        }

        private TabControl tabControl;
        private TabPage tabGeneral;
        private TabPage tabDatabase;
        private TabPage tabDataBalancing;
        private TabPage tabFeatureEngineering;
        private TabPage tabAutoML;
        private TabPage tabTraining;
        private TabPage tabInputFields;

        private Label lblAuthor;
        private TextBox txtAuthor;
        private Label lblDescription;
        private TextBox txtDescription;
        private Label lblModelType;
        private ComboBox cboModelType;
        private Label lblTargetField;
        private TextBox txtTargetField;

        private Label lblServer;
        private TextBox txtDatabaseServer;
        private Label lblDatabase;
        private TextBox txtDatabaseName;
        private Label lblTableName;
        private TextBox txtTableName;
        private Label lblOutputTable;
        private TextBox txtOutputTableName;
        private Label lblWhereClause;
        private TextBox txtWhereClause;

        private Label lblBalanceMethod;
        private ComboBox cboBalanceMethod;
        private Label lblBalanceOrder;
        private NumericUpDown numBalanceOrder;
        private Label lblKNeighbors;
        private NumericUpDown numKNeighbors;
        private Label lblUndersamplingRatio;
        private NumericUpDown numUndersamplingRatio;
        private Label lblMinorityRatio;
        private NumericUpDown numMinorityRatio;

        private Label lblFeatureMethod;
        private ComboBox cboFeatureMethod;
        private Label lblFeatureOrder;
        private NumericUpDown numFeatureOrder;
        private Label lblComponents;
        private NumericUpDown numComponents;
        private Label lblMaxFeatures;
        private NumericUpDown numMaxFeatures;
        private Label lblMulticollinearity;
        private NumericUpDown numMulticollinearity;

        private CheckBox chkAutoML;
        private Label lblExperimentTime;
        private NumericUpDown numExperimentTime;
        private Label lblMaxModels;
        private NumericUpDown numMaxModels;
        private Label lblOptimizingMetric;
        private TextBox txtOptimizingMetric;

        private Label lblAlgorithm;
        private TextBox txtAlgorithm;
        private Label lblTestFraction;
        private NumericUpDown numTestFraction;
        private Label lblAlgorithmParams;
        private ListBox lstAlgorithmParams;
        private Button btnAddParam;
        private Button btnEditParam;
        private Button btnDeleteParam;

        private ListBox lstInputFields;
        private Button btnAddInputField;
        private Button btnEditInputField;
        private Button btnDeleteInputField;

        private Button btnSave;
        private Button btnLoad;
    }
}

namespace D2G.Iris.ML.ConfigGenerator
{
    internal static class Program
    {
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new MainForm());
        }
    }

    #region Models
    // Included here for standalone operation - in a real implementation
    // these would reference the existing model classes

    namespace Core.Enums
    {
        public enum DataBalanceMethod
        {
            None,
            SMOTE
        }

        public enum FeatureSelectionMethod
        {
            None,
            Correlation,
            PCA
        }

        public enum ModelType
        {
            BinaryClassification = 0,
            MultiClassClassification = 1,
            Regression = 2
        }
    }

    public class ModelConfig
    {
        public string Author { get; set; }
        public string Description { get; set; }
        public Core.Enums.ModelType ModelType { get; set; }
        public List<InputField> InputFields { get; set; } = new List<InputField>();
        public TrainingParameters TrainingParameters { get; set; }
        public DatabaseConfig Database { get; set; }
        public FeatureEngineeringConfig FeatureEngineering { get; set; }
        public DataBalancingConfig DataBalancing { get; set; } = new DataBalancingConfig();
        public string TargetField { get; set; }
        public AutoMLConfig AutoML { get; set; }
    }

    public class InputField
    {
        public string Name { get; set; }
        public bool IsEnabled { get; set; }
    }

    public class TrainingParameters
    {
        public string Algorithm { get; set; }
        public Dictionary<string, object> AlgorithmParameters { get; set; } = new Dictionary<string, object>();
        public double TestFraction { get; set; }
    }

    public class DatabaseConfig
    {
        public string Server { get; set; }
        public string Database { get; set; }
        public string TableName { get; set; }
        public string OutputTableName { get; set; }
        public string WhereClause { get; set; }
    }

    public class FeatureEngineeringConfig
    {
        public Core.Enums.FeatureSelectionMethod Method { get; set; }
        public int ExecutionOrder { get; set; }
        public int NumberOfComponents { get; set; }
        public int MaxFeatures { get; set; }
        public double MulticollinearityThreshold { get; set; }
    }

    public class DataBalancingConfig
    {
        public Core.Enums.DataBalanceMethod Method { get; set; }
        public int ExecutionOrder { get; set; }
        public int KNeighbors { get; set; }
        public float UndersamplingRatio { get; set; }
        public float MinorityToMajorityRatio { get; set; }
    }

    public class AutoMLConfig
    {
        public bool Enabled { get; set; }
        public int MaxExperimentTimeInSeconds { get; set; }
        public int MaxModels { get; set; }
        public string OptimizingMetric { get; set; }
    }
    #endregion
}