# 🧬 Enhanced Drug Interaction Checker

An advanced educational machine learning project that identifies potential drug-drug interactions, classifies their severity, and suggests safer alternatives using state-of-the-art ML techniques with a modern interactive web interface.

## ⚠️ Educational Disclaimer
This project is developed **strictly for educational purposes** and should **NEVER** be used for real-world medical decisions. Always consult qualified healthcare professionals for medical advice.

##  Key Features

###  Core Functionality
- **Drug-drug interaction detection** using multiple ML algorithms
- **Severity classification** (Low, Moderate, High) with confidence scores
- **Alternative drug suggestions** for high-severity interactions
- **Batch processing** with CSV file upload support
- **Interactive web interface** with modern UI/UX


###  Advanced ML Pipeline
- **Feature Engineering**: Drug embeddings, interaction features, statistical features
- **Feature Selection**: SelectKBest, Recursive Feature Elimination (RFE)
- **Data Balancing**: SMOTE, under-sampling, combined techniques
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Model Ensemble**: Voting classifier combining best models
- **Cross-validation Analysis**: Robust performance estimation

###  Machine Learning Models
#### Basic Models
- **Logistic Regression** - Linear classification baseline
- **Decision Tree** - Rule-based interpretable model
- **Random Forest** - Ensemble of decision trees
- **Naive Bayes** - Probabilistic classifier

#### Advanced Models
- **Support Vector Machine (SVM)** - High-dimensional classification
- **Neural Network (MLP)** - Deep learning approach
- **Gradient Boosting** - Sequential ensemble method
- **AdaBoost** - Adaptive boosting algorithm
- **Ensemble Methods** - Voting classifier combining best models

###  Modern Web Interface
- **Responsive Design** - Works on all devices
- **Interactive Elements** - Hover effects, animations, ripple buttons
- **Real-time Validation** - Form validation and progress indicators
- **Multiple Sections** - Home, About, Models, Drug Database, Help
- **CSV Upload** - Drag-and-drop file processing
- **Enhanced Notifications** - Toast notifications and loading overlays

##  Technical Stack

### Backend
- **Python 3.8+** - Core programming language
- **Scikit-learn** - Machine learning algorithms
- **Pandas & NumPy** - Data manipulation and analysis
- **Matplotlib & Seaborn** - Data visualization
- **Imbalanced-learn** - Data balancing techniques
- **Flask** - Web framework and REST API

### Frontend
- **HTML5 & CSS3** - Modern web standards
- **Bootstrap 5** - Responsive design framework
- **JavaScript (ES6+)** - Interactive functionality
- **Font Awesome** - Icons and visual elements
- **Custom CSS** - Enhanced styling with gradients and animations

### Development Tools
- **Jupyter Notebooks** - Interactive data analysis
- **Git** - Version control
- **Virtual Environment** - Dependency management

##  Project Structure

```
drug-interaction-checker/
├── 📂 data/
│   ├── 📂 raw/                    # Raw data files
│   └── 📂 processed/              # Processed datasets
├── 📂 src/
│   ├── 📄 data_processing.py      # Data loading and preprocessing
│   ├── 📄 models.py              # Basic ML models
│   ├── 📄 advanced_models.py     # Advanced ML models and ensemble
│   ├── 📄 feature_engineering.py # Feature engineering pipeline
│   ├── 📄 evaluation.py          # Model evaluation and visualization
│   ├── 📄 web_interface.py       # Flask web application
│   └── 📄 main.py               # Enhanced main application
├── 📂 templates/
│   └── 📄 index.html            # Main web interface
├── 📂 static/
│   ├── 📂 css/
│   │   └── 📄 style.css         # Enhanced custom styling
│   └── 📂 js/
│       └── 📄 main.js           # Interactive JavaScript
├── 📂 notebooks/
│   └── 📄 eda_analysis.ipynb    # Exploratory data analysis
├── 📂 models/                   # Trained model files (*.pkl)
├── 📂 plots/                    # Generated visualizations
├── 📄 requirements.txt          # Python dependencies
├── 📄 LICENSE                   # MIT License
└── 📄 README.md                # This file
```

##  Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/drug-interaction-checker.git
cd drug-interaction-checker
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
# Option 1: Web Interface (Recommended)
python3 src/web_interface.py
# Visit: http://localhost:5001

# Option 2: Command Line Interface
python3 src/main.py

# Option 3: Jupyter Notebook
jupyter notebook notebooks/eda_analysis.ipynb
```

##  Usage Guide

###  Web Interface Features

#### **Home Section**
- **Single Drug Check**: Enter two drug names for instant analysis
- **Batch Processing**: Upload CSV files or enter multiple pairs
- **Real-time Results**: Interactive results with confidence scores
- **Alternative Suggestions**: Safer drug alternatives for high-severity interactions

#### **About Section**
- **Project Overview**: Educational purpose and technical details
- **ML Models Explanation**: Understanding different algorithms
- **Technical Stack**: Technologies used in the project
- **Medical Disclaimer**: Important safety information

#### **Models Section**
- **Performance Comparison**: Detailed metrics for all models
- **Dataset Statistics**: Training data information
- **Severity Distribution**: Visual breakdown of interaction levels

#### **Drug Database Section**
- **Complete Drug List**: All available drugs in the system
- **Drug Classes**: Organized by therapeutic categories
- **Usage Statistics**: Most studied drug combinations

#### **Help Section**
- **Getting Started**: Step-by-step usage guide
- **Keyboard Shortcuts**: Power user features
- **Batch Processing**: CSV upload instructions
- **Understanding Results**: Interpreting predictions and confidence scores

###  CSV Batch Processing
1. **Download Sample**: Get a template CSV file
2. **Prepare Data**: Format as `drug1,drug2` (one pair per line)
3. **Upload File**: Drag and drop or browse for your CSV
4. **Preview Data**: Review parsed drug pairs
5. **Process Batch**: Analyze all combinations at once



##  Model Performance

| Model Category | Best F1-Score | Accuracy | Features |
|----------------|---------------|----------|----------|
| **Basic Models** | ~0.57 | ~0.66 | Fast, interpretable |
| **Advanced Models** | ~0.75+ | ~0.80+ | High accuracy, complex |
| **Ensemble** | ~0.78+ | ~0.82+ | Best performance |

### Performance Metrics
- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate
- **Recall**: Sensitivity to interactions
- **F1-Score**: Balanced performance measure
- **Confidence**: Prediction certainty (0-100%)



##  Advanced Features

###  Feature Engineering
- **Drug Embeddings**: Vector representations based on interaction patterns
- **Statistical Features**: Group statistics and aggregations
- **Interaction Features**: Drug pair frequencies and class combinations
- **Categorical Encoding**: Label encoding for drug classes

###  Model Optimization
- **Hyperparameter Tuning**: Grid search with 5-fold cross-validation
- **Feature Selection**: Automated selection of most important features
- **Data Balancing**: SMOTE and other techniques for class imbalance
- **Ensemble Methods**: Voting classifier combining best models



##  Educational Objectives

### Data Science Concepts
- ✅ Data preprocessing and cleaning
- ✅ Exploratory data analysis (EDA)
- ✅ Feature engineering and selection
- ✅ Data visualization techniques
- ✅ Statistical analysis and interpretation

### Machine Learning Techniques
- ✅ Supervised learning algorithms
- ✅ Model comparison and selection
- ✅ Hyperparameter optimization
- ✅ Ensemble methods
- ✅ Cross-validation strategies
- ✅ Performance evaluation metrics

### Software Engineering Practices
- ✅ Modular code architecture
- ✅ API design and development
- ✅ Web application development
- ✅ User interface design
- ✅ Documentation and testing





##  Limitations & Disclaimers

### ⚠️ Educational Purpose Only
- This is a **learning project** for academic purposes
- **Not validated** for clinical use
- **Simplified data model** for educational clarity
- **Synthetic dataset** (not real clinical data)

###  Technical Limitations
- Training dataset uses 15 drugs for demonstration (but can predict on any drug name)
- Simplified interaction modeling
- No real-time clinical integration
- Basic severity classification

###  Safety Warnings
- **NEVER use for medical decisions**
- **Always consult healthcare professionals**
- **Not a substitute for clinical expertise**
- **Educational demonstration only**

## 🤝 Contributing

This educational project welcomes contributions for learning purposes:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comments and documentation
- Include tests for new features
- Update README for significant changes

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- **Scikit-learn** community for excellent ML tools
- **Flask** team for the web framework
- **Bootstrap** for responsive design components
- **Font Awesome** for beautiful icons
- **Educational institutions** promoting open-source learning

##  Support & Documentation

### Getting Help
-  Check the **Help section** in the web interface
-  RExamine the **Jupyter notebooks** for analysis examples
-  Teast with the **interactive web interface**



**⚠️ Remember**: This project is for educational purposes only. Always consult qualified healthcare professionals for medical advice and drug interaction concerns.
