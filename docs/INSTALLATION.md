# Installation Guide

## Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/tanzilaaaaa/DrugSafe-ML.git
cd DrugSafe-ML
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 5. Run the Application
```bash
python src/main.py
```

The application will be available at `http://localhost:5001`

## Troubleshooting

If you encounter issues during installation, ensure:
- Python version is 3.8 or higher
- Virtual environment is activated
- All dependencies are installed correctly
