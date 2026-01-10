// Main JavaScript for Drug Interaction Checker with Enhanced Interactivity

// Session tracking variables
let sessionStats = {
    totalPredictions: 0,
    interactionsFound: 0,
    startTime: new Date()
};

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the application
    initializeApp();
    
    // Set up event listeners
    setupEventListeners();
    
    // Load initial data
    loadModelInfo();
    loadQuickStats();
    loadDrugList();
    
    // Add interactive enhancements
    addInteractiveEnhancements();
    
    // Initialize session tracking
    initializeSessionTracking();
});

function initializeSessionTracking() {
    // Load session stats from localStorage if available
    const savedStats = localStorage.getItem('drugCheckerSession');
    if (savedStats) {
        const parsed = JSON.parse(savedStats);
        // Only restore if from today
        const today = new Date().toDateString();
        const savedDate = new Date(parsed.startTime).toDateString();
        if (today === savedDate) {
            sessionStats = parsed;
        }
    }
    
    // Update display
    updateSessionStats();
}

function updateSessionStats() {
    const totalElement = document.getElementById('totalPredictions');
    const interactionsElement = document.getElementById('interactionsFound');
    
    if (totalElement) {
        totalElement.textContent = sessionStats.totalPredictions;
        totalElement.classList.add('fade-in');
    }
    
    if (interactionsElement) {
        interactionsElement.textContent = sessionStats.interactionsFound;
        interactionsElement.classList.add('fade-in');
    }
    
    // Save to localStorage
    localStorage.setItem('drugCheckerSession', JSON.stringify(sessionStats));
}

function incrementPredictionStats(hasInteraction) {
    sessionStats.totalPredictions++;
    if (hasInteraction) {
        sessionStats.interactionsFound++;
    }
    updateSessionStats();
    
    // Add visual feedback
    const totalElement = document.getElementById('totalPredictions');
    const interactionsElement = document.getElementById('interactionsFound');
    
    if (totalElement) {
        totalElement.style.transform = 'scale(1.2)';
        totalElement.style.color = '#28a745';
        setTimeout(() => {
            totalElement.style.transform = 'scale(1)';
            totalElement.style.color = '';
        }, 500);
    }
    
    if (hasInteraction && interactionsElement) {
        interactionsElement.style.transform = 'scale(1.2)';
        interactionsElement.style.color = '#dc3545';
        setTimeout(() => {
            interactionsElement.style.transform = 'scale(1)';
            interactionsElement.style.color = '';
        }, 500);
    }
}

function initializeApp() {
    console.log('Drug Interaction Checker initialized with enhanced interactivity');
    
    // Add loading animation to page
    document.body.style.opacity = '0';
    setTimeout(() => {
        document.body.style.transition = 'opacity 0.5s ease-in-out';
        document.body.style.opacity = '1';
    }, 100);
}

// Section Management
function showSection(sectionName) {
    // Hide all sections
    const sections = ['main', 'about', 'models', 'database', 'help'];
    sections.forEach(section => {
        const element = document.getElementById(`${section}-section`);
        if (element) {
            element.style.display = 'none';
        }
    });
    
    // Show selected section
    const targetSection = document.getElementById(`${sectionName}-section`);
    if (targetSection) {
        targetSection.style.display = 'block';
        targetSection.classList.add('fade-in');
        
        // Load section-specific content
        loadSectionContent(sectionName);
    }
    
    // Update navigation active state
    updateNavigation(sectionName);
}

function updateNavigation(activeSection) {
    // Remove active class from all nav links
    document.querySelectorAll('.navbar-nav .nav-link').forEach(link => {
        link.classList.remove('active');
    });
    
    // Add active class to current section (simplified approach)
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    const sectionMap = { 'main': 0, 'about': 1, 'models': 2, 'database': 3, 'help': 4 };
    if (sectionMap[activeSection] !== undefined) {
        navLinks[sectionMap[activeSection]].classList.add('active');
    }
}

function loadSectionContent(sectionName) {
    switch(sectionName) {
        case 'models':
            loadModelPerformance();
            break;
        case 'database':
            loadDatabaseExplorer();
            break;
        case 'main':
            // Reload main section data
            loadModelInfo();
            loadQuickStats();
            loadDrugList();
            break;
    }
}

async function loadModelPerformance() {
    const content = document.getElementById('modelPerformanceContent');
    
    try {
        const response = await fetch('/api/statistics');
        const data = await response.json();
        
        if (response.ok && data.model_performance) {
            let html = `
                <div class="row mb-4">
                    <div class="col-12">
                        <h5><i class="fas fa-trophy me-2 text-warning"></i>Model Comparison</h5>
                        <p class="text-muted">Performance metrics for all trained models</p>
                    </div>
                </div>
            `;
            
            if (data.model_performance.interaction) {
                html += `
                    <div class="table-responsive">
                        <table class="table table-striped">
                            <thead class="table-dark">
                                <tr>
                                    <th>Model</th>
                                    <th>Accuracy</th>
                                    <th>Precision</th>
                                    <th>Recall</th>
                                    <th>F1-Score</th>
                                    <th>Performance</th>
                                </tr>
                            </thead>
                            <tbody>
                `;
                
                Object.entries(data.model_performance.interaction).forEach(([model, metrics]) => {
                    const f1Score = metrics.f1_score * 100;
                    const performanceClass = f1Score >= 80 ? 'success' : f1Score >= 60 ? 'warning' : 'danger';
                    
                    html += `
                        <tr>
                            <td><strong>${model}</strong></td>
                            <td>${(metrics.accuracy * 100).toFixed(1)}%</td>
                            <td>${(metrics.precision * 100).toFixed(1)}%</td>
                            <td>${(metrics.recall * 100).toFixed(1)}%</td>
                            <td>${f1Score.toFixed(1)}%</td>
                            <td><span class="badge bg-${performanceClass}">${f1Score >= 80 ? 'Excellent' : f1Score >= 60 ? 'Good' : 'Fair'}</span></td>
                        </tr>
                    `;
                });
                
                html += `
                            </tbody>
                        </table>
                    </div>
                `;
            }
            
            // Add dataset information
            if (data.dataset) {
                html += `
                    <div class="row mt-4">
                        <div class="col-md-6">
                            <div class="card bg-light">
                                <div class="card-body">
                                    <h6><i class="fas fa-database me-2"></i>Dataset Overview</h6>
                                    <ul class="list-unstyled">
                                        <li><strong>Total Records:</strong> ${data.dataset.total_records}</li>
                                        <li><strong>Interaction Rate:</strong> ${(data.dataset.interaction_rate * 100).toFixed(1)}%</li>
                                        <li><strong>Unique Drugs:</strong> ${data.dataset.unique_drugs}</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card bg-light">
                                <div class="card-body">
                                    <h6><i class="fas fa-chart-pie me-2"></i>Severity Distribution</h6>
                                    <div class="severity-chart">
                `;
                
                if (data.dataset.severity_distribution) {
                    Object.entries(data.dataset.severity_distribution).forEach(([severity, count]) => {
                        const percentage = (count / data.dataset.total_records * 100).toFixed(1);
                        html += `
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <span class="${getSeverityClass(severity)}">${severity}</span>
                                <div class="flex-grow-1 mx-3">
                                    <div class="progress" style="height: 10px;">
                                        <div class="progress-bar ${getSeverityClass(severity)}" style="width: ${percentage}%"></div>
                                    </div>
                                </div>
                                <span class="text-muted">${count} (${percentage}%)</span>
                            </div>
                        `;
                    });
                }
                
                html += `
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            }
            
            content.innerHTML = html;
        } else {
            content.innerHTML = '<div class="alert alert-warning">Model performance data not available</div>';
        }
    } catch (error) {
        console.error('Error loading model performance:', error);
        content.innerHTML = '<div class="alert alert-danger">Error loading model performance data</div>';
    }
}

async function loadDatabaseExplorer() {
    const content = document.getElementById('databaseContent');
    
    try {
        const [drugResponse, statsResponse] = await Promise.all([
            fetch('/api/drug_list'),
            fetch('/api/statistics')
        ]);
        
        const drugData = await drugResponse.json();
        const statsData = await statsResponse.json();
        
        if (drugResponse.ok && drugData) {
            let html = `
                <div class="row mb-4">
                    <div class="col-md-4">
                        <div class="stat-card text-center">
                            <div class="stat-number">${drugData.total_drugs}</div>
                            <div class="stat-label">Total Drugs</div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="stat-card text-center">
                            <div class="stat-number">${drugData.total_classes}</div>
                            <div class="stat-label">Drug Classes</div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="stat-card text-center">
                            <div class="stat-number">${statsData.dataset ? statsData.dataset.total_records : 'N/A'}</div>
                            <div class="stat-label">Interactions Studied</div>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5><i class="fas fa-pills me-2"></i>Available Drugs</h5>
                            </div>
                            <div class="card-body" style="max-height: 400px; overflow-y: auto;">
                                <div class="row">
            `;
            
            // Display drugs in a grid
            drugData.drugs.forEach((drug, index) => {
                if (index % 2 === 0) html += '<div class="col-md-6">';
                html += `<div class="drug-badge-interactive mb-2 me-1">${drug}</div>`;
                if (index % 2 === 1 || index === drugData.drugs.length - 1) html += '</div>';
            });
            
            html += `
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5><i class="fas fa-layer-group me-2"></i>Drug Classes</h5>
                            </div>
                            <div class="card-body">
                                <div class="mb-3">
                                    ${drugData.drug_classes.map(drugClass => 
                                        `<span class="badge bg-primary text-white me-1 mb-2" style="font-size: 0.9rem; padding: 0.5rem 0.8rem;">${drugClass}</span>`
                                    ).join('')}
                                </div>
            `;
            
            // Add most common drugs if available
            if (statsData.dataset && statsData.dataset.most_common_drugs) {
                html += `
                    <h6><i class="fas fa-star me-2"></i>Most Studied Drugs</h6>
                    <div class="list-group">
                `;
                
                Object.entries(statsData.dataset.most_common_drugs).slice(0, 5).forEach(([drug, count]) => {
                    html += `
                        <div class="list-group-item d-flex justify-content-between align-items-center">
                            <span>${drug}</span>
                            <span class="badge bg-primary rounded-pill">${count} studies</span>
                        </div>
                    `;
                });
                
                html += '</div>';
            }
            
            html += `
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            content.innerHTML = html;
        } else {
            content.innerHTML = '<div class="alert alert-warning">Database information not available</div>';
        }
    } catch (error) {
        console.error('Error loading database explorer:', error);
        content.innerHTML = '<div class="alert alert-danger">Error loading database information</div>';
    }
}

function addInteractiveEnhancements() {
    // Add hover effects to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
    
    // Add typing animation to input fields
    const inputs = document.querySelectorAll('.form-control');
    inputs.forEach(input => {
        input.addEventListener('focus', function() {
            this.style.transform = 'scale(1.02)';
            this.style.boxShadow = '0 0 20px rgba(102, 126, 234, 0.3)';
        });
        
        input.addEventListener('blur', function() {
            this.style.transform = 'scale(1)';
            this.style.boxShadow = '';
        });
        
        // Add real-time validation
        input.addEventListener('input', function() {
            if (this.value.length > 0) {
                this.classList.add('is-valid');
                this.classList.remove('is-invalid');
            } else {
                this.classList.remove('is-valid', 'is-invalid');
            }
            
            // Update progress bar
            updateProgress();
        });
        
        // Remove validation on clear
        input.addEventListener('keydown', function(e) {
            if (e.key === 'Delete' || e.key === 'Backspace') {
                if (this.value.length <= 1) {
                    this.classList.remove('is-valid', 'is-invalid');
                }
            }
        });
    });
    
    // Add button click animations
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            // Create ripple effect
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.width = ripple.style.height = size + 'px';
            ripple.style.left = x + 'px';
            ripple.style.top = y + 'px';
            ripple.classList.add('ripple');
            
            this.appendChild(ripple);
            
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });
    
    // Add auto-complete functionality
    setupAutoComplete();
    
    // Add keyboard shortcuts
    setupKeyboardShortcuts();
    
    // Add progress indicators
    setupProgressIndicators();
}

function setupAutoComplete() {
    const drug1Input = document.getElementById('drug1');
    const drug2Input = document.getElementById('drug2');
    
    // Create datalist for autocomplete
    const datalist = document.createElement('datalist');
    datalist.id = 'drugOptions';
    document.body.appendChild(datalist);
    
    drug1Input.setAttribute('list', 'drugOptions');
    drug2Input.setAttribute('list', 'drugOptions');
    
    // Populate with common drugs
    const commonDrugs = [
        'Warfarin', 'Aspirin', 'Metformin', 'Lisinopril', 'Atorvastatin',
        'Omeprazole', 'Levothyroxine', 'Gabapentin', 'Ibuprofen', 'Acetaminophen',
        'Simvastatin', 'Hydrochlorothiazide', 'Losartan', 'Amlodipine', 'Metoprolol'
    ];
    
    commonDrugs.forEach(drug => {
        const option = document.createElement('option');
        option.value = drug;
        datalist.appendChild(option);
    });
}

function setupKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to submit form
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            const form = document.getElementById('interactionForm');
            if (form) {
                form.dispatchEvent(new Event('submit'));
            }
        }
        
        // Escape to clear form
        if (e.key === 'Escape') {
            e.preventDefault();
            clearForm();
        }
        
        // Ctrl/Cmd + K to focus on first input
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const firstInput = document.getElementById('drug1');
            if (firstInput) {
                firstInput.focus();
                firstInput.select();
            }
        }
        
        // Ctrl/Cmd + L to clear form (alternative)
        if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
            e.preventDefault();
            clearForm();
        }
    });
}

function setupProgressIndicators() {
    // Add progress bar for form completion
    const progressBar = document.createElement('div');
    progressBar.className = 'progress mb-3';
    progressBar.style.height = '4px';
    progressBar.innerHTML = '<div class="progress-bar" style="width: 0%"></div>';
    
    const form = document.getElementById('interactionForm');
    form.insertBefore(progressBar, form.firstChild);
    
    const inputs = form.querySelectorAll('input[required]');
    
    function updateProgress() {
        const filledInputs = Array.from(inputs).filter(input => input.value.trim() !== '');
        const progress = (filledInputs.length / inputs.length) * 100;
        
        const progressBarFill = progressBar.querySelector('.progress-bar');
        progressBarFill.style.width = progress + '%';
        
        if (progress === 100) {
            progressBarFill.classList.add('bg-success');
        } else {
            progressBarFill.classList.remove('bg-success');
        }
    }
    
    inputs.forEach(input => {
        input.addEventListener('input', updateProgress);
    });
    
    // Make updateProgress globally available
    window.updateProgress = updateProgress;
}

function setupEventListeners() {
    // Main form submission
    document.getElementById('interactionForm').addEventListener('submit', handleFormSubmission);
    
    // Clear button
    document.getElementById('clearBtn').addEventListener('click', clearForm);
    
    // Batch processing
    document.getElementById('batchProcessBtn').addEventListener('click', handleBatchProcessing);
    
    // CSV file handling
    document.getElementById('csvFile').addEventListener('change', handleCsvFileSelect);
    document.getElementById('loadCsvBtn').addEventListener('click', loadCsvFile);
    document.getElementById('processCsvBtn').addEventListener('click', processCsvData);
    document.getElementById('downloadSampleBtn').addEventListener('click', downloadSampleCsv);
}

// CSV File Handling Functions
let csvData = [];

function handleCsvFileSelect(event) {
    const file = event.target.files[0];
    const loadBtn = document.getElementById('loadCsvBtn');
    
    if (file) {
        loadBtn.disabled = false;
        loadBtn.innerHTML = '<i class="fas fa-upload me-2"></i>Load CSV';
    } else {
        loadBtn.disabled = true;
        document.getElementById('csvPreview').style.display = 'none';
        document.getElementById('processCsvBtn').disabled = true;
    }
}

function loadCsvFile() {
    const fileInput = document.getElementById('csvFile');
    const file = fileInput.files[0];
    
    if (!file) {
        showNotification('Please select a CSV file first', 'warning');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const csvText = e.target.result;
            const lines = csvText.split('\n').filter(line => line.trim());
            
            // Parse CSV data
            csvData = [];
            const invalidLines = [];
            
            lines.forEach((line, index) => {
                const trimmedLine = line.trim();
                if (trimmedLine) {
                    const parts = trimmedLine.split(',').map(part => part.trim());
                    if (parts.length >= 2 && parts[0] && parts[1]) {
                        csvData.push({ drug1: parts[0], drug2: parts[1] });
                    } else {
                        invalidLines.push(index + 1);
                    }
                }
            });
            
            // Show preview
            displayCsvPreview(csvData, invalidLines);
            
            if (csvData.length > 0) {
                document.getElementById('processCsvBtn').disabled = false;
                showNotification(`Successfully loaded ${csvData.length} drug pairs from CSV`, 'success');
            } else {
                showNotification('No valid drug pairs found in the CSV file', 'warning');
            }
            
        } catch (error) {
            console.error('Error reading CSV:', error);
            showNotification('Error reading CSV file. Please check the format.', 'error');
        }
    };
    
    reader.readAsText(file);
}

function displayCsvPreview(data, invalidLines) {
    const preview = document.getElementById('csvPreview');
    const content = document.getElementById('csvPreviewContent');
    
    let html = `
        <div class="mb-3">
            <strong>File Summary:</strong>
            <ul class="list-unstyled mb-0">
                <li><i class="fas fa-check-circle text-success me-2"></i>Valid pairs: ${data.length}</li>
                ${invalidLines.length > 0 ? `<li><i class="fas fa-exclamation-triangle text-warning me-2"></i>Invalid lines: ${invalidLines.length} (lines: ${invalidLines.join(', ')})</li>` : ''}
            </ul>
        </div>
    `;
    
    if (data.length > 0) {
        html += `
            <div class="table-responsive" style="max-height: 200px; overflow-y: auto;">
                <table class="table table-sm table-striped">
                    <thead class="table-dark">
                        <tr>
                            <th>#</th>
                            <th>Drug 1</th>
                            <th>Drug 2</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        data.slice(0, 10).forEach((pair, index) => {
            html += `
                <tr>
                    <td>${index + 1}</td>
                    <td><span class="badge bg-light text-dark">${pair.drug1}</span></td>
                    <td><span class="badge bg-light text-dark">${pair.drug2}</span></td>
                </tr>
            `;
        });
        
        if (data.length > 10) {
            html += `
                <tr>
                    <td colspan="3" class="text-center text-muted">
                        <i class="fas fa-ellipsis-h me-2"></i>
                        and ${data.length - 10} more pairs...
                    </td>
                </tr>
            `;
        }
        
        html += `
                    </tbody>
                </table>
            </div>
        `;
    }
    
    content.innerHTML = html;
    preview.style.display = 'block';
}

async function processCsvData() {
    if (csvData.length === 0) {
        showNotification('No CSV data to process', 'warning');
        return;
    }
    
    // Show loading with enhanced overlay
    const loadingOverlay = showLoadingOverlay(`Processing ${csvData.length} drug pairs from CSV...`);
    document.getElementById('processCsvBtn').disabled = true;
    document.getElementById('processCsvBtn').innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
    
    try {
        const response = await fetch('/api/batch_predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                drug_pairs: csvData
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayBatchResults(data);
            
            // Update session statistics for CSV processing
            if (data.results) {
                data.results.forEach(result => {
                    if (!result.error) {
                        incrementPredictionStats(result.interaction);
                    }
                });
            }
            
            showNotification(`CSV processing completed! Analyzed ${data.total_pairs} drug pairs.`, 'success');
        } else {
            showNotification(data.error || 'CSV processing failed', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('Network error during CSV processing', 'error');
    } finally {
        hideLoadingOverlay(loadingOverlay);
        document.getElementById('processCsvBtn').disabled = false;
        document.getElementById('processCsvBtn').innerHTML = '<i class="fas fa-play me-2"></i>Process CSV Data';
    }
}

function downloadSampleCsv() {
    const sampleData = [
        'drug1,drug2',
        'Warfarin,Aspirin',
        'Metformin,Lisinopril',
        'Atorvastatin,Omeprazole',
        'Levothyroxine,Gabapentin',
        'Ibuprofen,Warfarin',
        'Simvastatin,Amlodipine',
        'Hydrochlorothiazide,Losartan'
    ].join('\n');
    
    const blob = new Blob([sampleData], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'drug_pairs_sample.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
    
    showNotification('Sample CSV file downloaded successfully!', 'success');
}

async function handleFormSubmission(event) {
    event.preventDefault();
    
    const drug1 = document.getElementById('drug1').value.trim();
    const drug2 = document.getElementById('drug2').value.trim();
    
    if (!drug1 || !drug2) {
        showNotification('Please enter both drug names', 'warning');
        return;
    }
    
    if (drug1.toLowerCase() === drug2.toLowerCase()) {
        showNotification('Please enter two different drugs', 'warning');
        return;
    }
    
    // Show enhanced loading overlay
    const loadingOverlay = showLoadingOverlay('Analyzing Drug Interaction...');
    hideResults();
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                drug1: drug1,
                drug2: drug2
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Add a small delay for better UX
            setTimeout(() => {
                displayResults(data);
                
                // Update session statistics
                incrementPredictionStats(data.interaction);
                
                showNotification('Analysis completed successfully!', 'success');
            }, 500);
        } else {
            showNotification(data.error || 'An error occurred during analysis', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('Network error. Please check your connection and try again.', 'error');
    } finally {
        setTimeout(() => {
            hideLoadingOverlay(loadingOverlay);
        }, 500);
    }
}

function displayResults(data) {
    const resultsContent = document.getElementById('resultsContent');
    
    // Determine interaction status and styling
    const interactionClass = data.interaction ? 'interaction-positive' : 'interaction-negative';
    const interactionIcon = data.interaction ? 'fas fa-exclamation-triangle text-danger icon-bounce' : 'fas fa-check-circle text-success';
    const interactionText = data.interaction ? 'INTERACTION DETECTED' : 'NO INTERACTION DETECTED';
    
    // Build confidence bar with enhanced styling
    const confidencePercent = (data.interaction_confidence * 100).toFixed(1);
    const confidenceClass = getConfidenceClass(data.interaction_confidence);
    
    let html = `
        <div class="alert ${interactionClass} fade-in" role="alert">
            <div class="d-flex align-items-center mb-3">
                <i class="${interactionIcon} me-2 fs-4"></i>
                <strong class="fs-5">${interactionText}</strong>
            </div>
            <div class="row">
                <div class="col-md-6 mb-3">
                    <strong>Drug Combination:</strong><br>
                    <div class="mt-2">
                        <span class="badge bg-primary me-2 p-2 drug-badge-interactive">${data.drug1}</span>
                        <i class="fas fa-plus mx-2 text-muted"></i>
                        <span class="badge bg-primary p-2 drug-badge-interactive">${data.drug2}</span>
                    </div>
                </div>
                <div class="col-md-6">
                    <strong>Prediction Confidence:</strong>
                    <div class="confidence-meter mt-2">
                        <div class="confidence-meter-fill ${confidenceClass}" style="width: ${confidencePercent}%"></div>
                    </div>
                    <div class="d-flex justify-content-between mt-1">
                        <small class="text-muted">Confidence</small>
                        <small class="fw-bold">${confidencePercent}%</small>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Add severity information if interaction exists
    if (data.interaction && data.severity) {
        const severityClass = getSeverityClass(data.severity);
        const severityIcon = getSeverityIcon(data.severity);
        const severityConfidence = (data.severity_confidence * 100).toFixed(1);
        
        html += `
            <div class="card mt-3 fade-in">
                <div class="card-body">
                    <h6 class="card-title">
                        <i class="fas fa-thermometer-half me-2"></i>Severity Assessment
                    </h6>
                    <div class="row align-items-center">
                        <div class="col-md-6">
                            <div class="severity-indicator ${severityClass}">
                                <i class="${severityIcon} me-2"></i>
                                <span>${data.severity.toUpperCase()} SEVERITY</span>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="mt-2">
                                <strong>Assessment Confidence:</strong>
                                <div class="confidence-meter mt-1">
                                    <div class="confidence-meter-fill ${getConfidenceClass(data.severity_confidence)}" 
                                         style="width: ${severityConfidence}%"></div>
                                </div>
                                <small class="text-muted">${severityConfidence}% confident</small>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Add alternatives for high severity with enhanced styling
        if (data.alternatives) {
            html += `
                <div class="alternatives-section fade-in">
                    <h6 class="mb-3">
                        <i class="fas fa-lightbulb me-2 text-warning"></i>
                        <strong>Recommended Safer Alternatives</strong>
                    </h6>
                    <div class="row">
                        <div class="col-md-6 mb-3">
                            <div class="p-3 bg-light rounded">
                                <strong class="text-primary">Instead of ${data.drug1}:</strong>
                                <div class="mt-2">
                                    ${data.alternatives.drug1_alternatives.map(alt => 
                                        `<span class="alternative-drug me-1 mb-1" onclick="selectDrug('${alt}', 'drug1')">${alt}</span>`
                                    ).join('')}
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6 mb-3">
                            <div class="p-3 bg-light rounded">
                                <strong class="text-primary">Instead of ${data.drug2}:</strong>
                                <div class="mt-2">
                                    ${data.alternatives.drug2_alternatives.map(alt => 
                                        `<span class="alternative-drug me-1 mb-1" onclick="selectDrug('${alt}', 'drug2')">${alt}</span>`
                                    ).join('')}
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="alert alert-info mt-3">
                        <i class="fas fa-info-circle me-2"></i>
                        <strong>Click on any alternative drug to automatically fill the form and recheck.</strong>
                    </div>
                </div>
            `;
        }
    }
    
    // Add detailed analysis section
    html += `
        <div class="card mt-3 fade-in">
            <div class="card-body">
                <h6 class="card-title">
                    <i class="fas fa-chart-line me-2"></i>Analysis Details
                </h6>
                <div class="row">
                    <div class="col-md-4">
                        <div class="text-center p-3 bg-light rounded">
                            <i class="fas fa-brain fs-3 text-primary mb-2"></i>
                            <div class="fw-bold">ML Prediction</div>
                            <small class="text-muted">AI-powered analysis</small>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="text-center p-3 bg-light rounded">
                            <i class="fas fa-database fs-3 text-success mb-2"></i>
                            <div class="fw-bold">Evidence-Based</div>
                            <small class="text-muted">Clinical data trained</small>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="text-center p-3 bg-light rounded">
                            <i class="fas fa-shield-alt fs-3 text-warning mb-2"></i>
                            <div class="fw-bold">Educational Only</div>
                            <small class="text-muted">Not for medical use</small>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Add timestamp with enhanced styling
    const timestamp = new Date(data.timestamp).toLocaleString();
    html += `
        <div class="mt-3 text-center">
            <small class="text-muted">
                <i class="fas fa-clock me-1"></i>
                Analysis completed on ${timestamp}
            </small>
        </div>
    `;
    
    resultsContent.innerHTML = html;
    showResults();
    
    // Add click handlers for interactive elements
    addResultInteractivity();
}

function addResultInteractivity() {
    // Add click handlers for drug badges
    document.querySelectorAll('.drug-badge-interactive').forEach(badge => {
        badge.addEventListener('click', function() {
            const drugName = this.textContent;
            showNotification(`Clicked on ${drugName}`, 'info');
        });
    });
    
    // Add hover effects for confidence bars
    document.querySelectorAll('.confidence-meter').forEach(meter => {
        meter.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.02)';
        });
        
        meter.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1)';
        });
    });
}

function selectDrug(drugName, inputId) {
    document.getElementById(inputId).value = drugName;
    showNotification(`Selected ${drugName} as ${inputId === 'drug1' ? 'first' : 'second'} drug`, 'success');
    
    // Add visual feedback
    const input = document.getElementById(inputId);
    input.style.background = 'linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%)';
    setTimeout(() => {
        input.style.background = '';
    }, 1000);
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <i class="fas fa-${getNotificationIcon(type)} me-2"></i>
        ${message}
    `;
    
    document.body.appendChild(notification);
    
    // Show notification
    setTimeout(() => {
        notification.classList.add('show');
    }, 100);
    
    // Hide notification after 3 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 3000);
}

function getNotificationIcon(type) {
    switch (type) {
        case 'success': return 'check-circle';
        case 'error': return 'exclamation-circle';
        case 'warning': return 'exclamation-triangle';
        default: return 'info-circle';
    }
}

function showLoadingOverlay(message = 'Processing...') {
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.innerHTML = `
        <div class="loading-content">
            <div class="spinner-border text-primary mb-3" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <h5>${message}</h5>
            <p class="text-muted mb-0">Please wait while we analyze the drug interaction</p>
        </div>
    `;
    
    document.body.appendChild(overlay);
    return overlay;
}

function hideLoadingOverlay(overlay) {
    if (overlay) {
        overlay.style.opacity = '0';
        setTimeout(() => {
            overlay.remove();
        }, 300);
    }
}

async function handleBatchProcessing() {
    const batchInput = document.getElementById('batchInput').value.trim();
    
    if (!batchInput) {
        showNotification('Please enter drug pairs for batch processing', 'warning');
        return;
    }
    
    // Parse input
    const lines = batchInput.split('\n').filter(line => line.trim());
    const drugPairs = [];
    
    for (const line of lines) {
        const parts = line.split(',').map(part => part.trim());
        if (parts.length === 2 && parts[0] && parts[1]) {
            drugPairs.push({ drug1: parts[0], drug2: parts[1] });
        }
    }
    
    if (drugPairs.length === 0) {
        showNotification('No valid drug pairs found. Use format: Drug1,Drug2', 'warning');
        return;
    }
    
    // Show loading with enhanced overlay
    const loadingOverlay = showLoadingOverlay('Processing Batch Analysis...');
    document.getElementById('batchProcessBtn').disabled = true;
    document.getElementById('batchProcessBtn').innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
    
    try {
        const response = await fetch('/api/batch_predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                drug_pairs: drugPairs
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayBatchResults(data);
            
            // Update session statistics for batch processing
            if (data.results) {
                data.results.forEach(result => {
                    if (!result.error) {
                        incrementPredictionStats(result.interaction);
                    }
                });
            }
            
            showNotification(`Batch processing completed! Analyzed ${data.total_pairs} drug pairs.`, 'success');
        } else {
            showNotification(data.error || 'Batch processing failed', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('Network error during batch processing', 'error');
    } finally {
        hideLoadingOverlay(loadingOverlay);
        document.getElementById('batchProcessBtn').disabled = false;
        document.getElementById('batchProcessBtn').innerHTML = '<i class="fas fa-play me-2"></i>Process Batch';
    }
}

function displayBatchResults(data) {
    const batchResultsContent = document.getElementById('batchResultsContent');
    
    let html = `
        <div class="mb-3">
            <strong>Batch Summary:</strong>
            <span class="badge bg-primary ms-2">${data.total_pairs} total pairs</span>
            <span class="badge bg-success ms-1">${data.successful_predictions} successful</span>
            <span class="badge bg-danger ms-1">${data.total_pairs - data.successful_predictions} failed</span>
        </div>
        
        <div class="table-responsive">
            <table class="table table-sm batch-table">
                <thead>
                    <tr>
                        <th>Drug 1</th>
                        <th>Drug 2</th>
                        <th>Interaction</th>
                        <th>Confidence</th>
                        <th>Severity</th>
                    </tr>
                </thead>
                <tbody>
    `;
    
    for (const result of data.results) {
        if (result.error) {
            html += `
                <tr class="table-danger">
                    <td colspan="5">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        Error: ${result.error}
                    </td>
                </tr>
            `;
        } else {
            const interactionBadge = result.interaction ? 
                '<span class="badge bg-danger">Yes</span>' : 
                '<span class="badge bg-success">No</span>';
            
            const confidence = (result.interaction_confidence * 100).toFixed(1) + '%';
            const severity = result.severity || 'N/A';
            
            html += `
                <tr>
                    <td><span class="badge bg-light text-dark">${result.drug1}</span></td>
                    <td><span class="badge bg-light text-dark">${result.drug2}</span></td>
                    <td>${interactionBadge}</td>
                    <td>${confidence}</td>
                    <td><span class="${getSeverityClass(severity)}">${severity}</span></td>
                </tr>
            `;
        }
    }
    
    html += `
                </tbody>
            </table>
        </div>
    `;
    
    batchResultsContent.innerHTML = html;
    document.getElementById('batchResults').style.display = 'block';
}

async function loadModelInfo() {
    try {
        const response = await fetch('/api/model_info');
        const data = await response.json();
        
        if (response.ok) {
            displayModelInfo(data);
        } else {
            document.getElementById('modelInfo').innerHTML = 
                '<div class="alert alert-warning">Failed to load model information</div>';
        }
    } catch (error) {
        console.error('Error loading model info:', error);
        document.getElementById('modelInfo').innerHTML = 
            '<div class="alert alert-danger">Network error</div>';
    }
}

function displayModelInfo(data) {
    const modelInfo = document.getElementById('modelInfo');
    
    let html = `
        <div class="mb-3">
            <div class="d-flex align-items-center">
                <i class="fas fa-check-circle text-success me-2"></i>
                <strong>Models Status: Active</strong>
            </div>
            <small class="text-muted">All ML models loaded and ready for predictions</small>
        </div>
        
        <div class="mb-3">
            <h6><i class="fas fa-brain me-2"></i>Available Models:</h6>
            <div class="mb-2">
                <strong>Interaction Detection:</strong><br>
                ${data.available_models.interaction_models.map(model => 
                    `<span class="badge bg-primary me-1 mb-1">${model}</span>`
                ).join('')}
            </div>
            <div class="mb-2">
                <strong>Severity Classification:</strong><br>
                ${data.available_models.severity_models.map(model => 
                    `<span class="badge bg-secondary me-1 mb-1">${model}</span>`
                ).join('')}
            </div>
        </div>
        
        <div class="card bg-light mb-3">
            <div class="card-body p-3">
                <h6 class="card-title mb-2">
                    <i class="fas fa-database me-2 text-info"></i>Training Dataset
                </h6>
                <div class="row text-center">
                    <div class="col-6">
                        <div class="stat-number text-primary">${data.dataset_info.total_records}</div>
                        <div class="stat-label">Drug Pairs Studied</div>
                    </div>
                    <div class="col-6">
                        <div class="stat-number text-warning">${(data.dataset_info.interaction_rate * 100).toFixed(1)}%</div>
                        <div class="stat-label">Had Interactions</div>
                    </div>
                </div>
                <small class="text-muted d-block mt-2">
                    <i class="fas fa-info-circle me-1"></i>
                    These are static values from the training data used to build the ML models.
                </small>
            </div>
        </div>
        
        <div class="card bg-light">
            <div class="card-body p-3">
                <h6 class="card-title mb-2">
                    <i class="fas fa-chart-line me-2 text-success"></i>Live Predictions
                </h6>
                <div class="row text-center">
                    <div class="col-6">
                        <div class="stat-number text-success" id="totalPredictions">0</div>
                        <div class="stat-label">Predictions Made</div>
                    </div>
                    <div class="col-6">
                        <div class="stat-number text-danger" id="interactionsFound">0</div>
                        <div class="stat-label">Interactions Found</div>
                    </div>
                </div>
                <small class="text-muted d-block mt-2">
                    <i class="fas fa-clock me-1"></i>
                    Live counters updated with each prediction in this session.
                </small>
            </div>
        </div>
    `;
    
    modelInfo.innerHTML = html;
}

async function loadQuickStats() {
    try {
        const response = await fetch('/api/statistics');
        const data = await response.json();
        
        if (response.ok) {
            displayQuickStats(data);
        } else {
            document.getElementById('quickStats').innerHTML = 
                '<div class="alert alert-warning">Failed to load statistics</div>';
        }
    } catch (error) {
        console.error('Error loading stats:', error);
        document.getElementById('quickStats').innerHTML = 
            '<div class="alert alert-danger">Network error</div>';
    }
}

function displayQuickStats(data) {
    const quickStats = document.getElementById('quickStats');
    
    if (data.model_performance && data.model_performance.interaction) {
        const bestModel = Object.entries(data.model_performance.interaction)
            .reduce((best, [name, metrics]) => 
                metrics.f1_score > best.score ? {name, score: metrics.f1_score} : best,
                {name: '', score: 0}
            );
        
        let html = `
            <div class="mb-3">
                <h6>Best Performing Model:</h6>
                <div class="d-flex justify-content-between">
                    <span>${bestModel.name}</span>
                    <span class="badge bg-success">${(bestModel.score * 100).toFixed(1)}% F1</span>
                </div>
            </div>
        `;
        
        if (data.dataset && data.dataset.severity_distribution) {
            html += `
                <div class="mb-3">
                    <h6>Severity Distribution:</h6>
                    ${Object.entries(data.dataset.severity_distribution).map(([severity, count]) => `
                        <div class="d-flex justify-content-between">
                            <span class="${getSeverityClass(severity)}">${severity}</span>
                            <span>${count}</span>
                        </div>
                    `).join('')}
                </div>
            `;
        }
        
        quickStats.innerHTML = html;
    } else {
        quickStats.innerHTML = '<div class="alert alert-info">Statistics not available</div>';
    }
}

async function loadDrugList() {
    try {
        const response = await fetch('/api/drug_list');
        const data = await response.json();
        
        if (response.ok) {
            displayDrugList(data);
        } else {
            document.getElementById('drugList').innerHTML = 
                '<div class="alert alert-warning">Failed to load drug list</div>';
        }
    } catch (error) {
        console.error('Error loading drug list:', error);
        document.getElementById('drugList').innerHTML = 
            '<div class="alert alert-danger">Network error</div>';
    }
}

function displayDrugList(data) {
    const drugList = document.getElementById('drugList');
    
    let html = `
        <div class="mb-3">
            <strong>${data.total_drugs}</strong> drugs available
        </div>
        <div class="mb-3" style="max-height: 200px; overflow-y: auto;">
            ${data.drugs.slice(0, 20).map(drug => 
                `<span class="drug-badge badge bg-light text-dark">${drug}</span>`
            ).join('')}
            ${data.drugs.length > 20 ? '<div class="mt-2"><small class="text-muted">...and more</small></div>' : ''}
        </div>
        <div class="mb-3">
            <strong>Drug Classes:</strong><br>
            <div class="mt-2">
                ${data.drug_classes.map(drugClass => 
                    `<span class="badge bg-primary text-white me-1 mb-1" style="font-size: 0.8rem; padding: 0.4rem 0.6rem;">${drugClass}</span>`
                ).join('')}
            </div>
        </div>
    `;
    
    drugList.innerHTML = html;
}

// Utility functions
function showLoading(show) {
    document.getElementById('loadingSpinner').style.display = show ? 'block' : 'none';
}

function showResults() {
    const results = document.getElementById('results');
    results.style.display = 'block';
    results.classList.add('fade-in');
}

function hideResults() {
    document.getElementById('results').style.display = 'none';
}

function clearForm() {
    // Clear input values
    document.getElementById('drug1').value = '';
    document.getElementById('drug2').value = '';
    
    // Remove validation classes
    const inputs = document.querySelectorAll('#interactionForm .form-control');
    inputs.forEach(input => {
        input.classList.remove('is-valid', 'is-invalid');
        input.style.background = '';
        input.style.transform = '';
        input.style.boxShadow = '';
    });
    
    // Hide results
    hideResults();
    
    // Reset progress bar
    const progressBar = document.querySelector('#interactionForm .progress-bar');
    if (progressBar) {
        progressBar.style.width = '0%';
        progressBar.classList.remove('bg-success');
    }
    
    // Clear any batch results
    document.getElementById('batchResults').style.display = 'none';
    document.getElementById('batchInput').value = '';
    
    // Show notification
    showNotification('Form cleared successfully', 'info');
}

function showAlert(message, type) {
    // Create and show bootstrap alert
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at top of container
    const container = document.querySelector('.container');
    container.insertBefore(alertDiv, container.firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

function getConfidenceClass(confidence) {
    if (confidence >= 0.8) return 'confidence-high';
    if (confidence >= 0.6) return 'confidence-medium';
    return 'confidence-low';
}

function getSeverityClass(severity) {
    switch (severity?.toLowerCase()) {
        case 'high': return 'severity-high';
        case 'moderate': return 'severity-moderate';
        case 'low': return 'severity-low';
        default: return '';
    }
}

function getSeverityIcon(severity) {
    switch (severity?.toLowerCase()) {
        case 'high': return 'fas fa-exclamation-triangle text-danger';
        case 'moderate': return 'fas fa-exclamation-circle text-warning';
        case 'low': return 'fas fa-info-circle text-info';
        default: return 'fas fa-question-circle';
    }
}