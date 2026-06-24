let dashboardData = null;

function openTab(evt, tabName) {
    const contents = document.getElementsByClassName("tab-content");
    for (let content of contents) content.classList.remove("active");

    const buttons = document.getElementsByClassName("nav-btn");
    for (let btn of buttons) btn.classList.remove("active");

    const target = document.getElementById(tabName);
    if (target) target.classList.add("active");
    
    // Update Nav
    const activeBtn = evt ? evt.currentTarget : document.getElementById('nav-' + tabName);
    if (activeBtn) activeBtn.classList.add("active");

    // Title mapping
    const titleMap = { 'dashboard': 'Overview', 'analytics': 'Return Metrics & Segments', 'tools': 'Risk Prediction Tools' };
    document.getElementById('view-title').innerText = titleMap[tabName] || tabName;

    // Trigger Resize for Plotly
    window.dispatchEvent(new Event('resize'));
}

async function loadDashboard() {
    try {
        const response = await fetch('/segment');
        dashboardData = await response.json();

        const zeroState = document.getElementById('zero-state');
        const dashContainer = document.getElementById('dashboard-container');

        if (dashboardData.error === "empty_state") {
            zeroState.style.display = 'flex';
            dashContainer.style.display = 'none';
            return;
        }

        // Show dashboard
        zeroState.style.display = 'none';
        dashContainer.style.display = 'block';

        // Fetch and display metrics
        const metricsResponse = await fetch('/metrics');
        const metrics = await metricsResponse.json();
        
        document.getElementById('stat-total').innerText = metrics.total_customers.toLocaleString();
        document.getElementById('stat-recency').innerText = metrics.avg_recency.toFixed(0) + 'd';
        document.getElementById('stat-frequency').innerText = metrics.avg_frequency.toFixed(1);
        document.getElementById('stat-monetary').innerText = '₹' + metrics.avg_monetary.toLocaleString('en-IN', { maximumFractionDigits: 0 });
        
        // Reverse Logistics metrics
        document.getElementById('stat-return-rate').innerText = metrics.avg_return_rate.toFixed(1) + '%';
        document.getElementById('stat-restocking').innerText = '₹' + metrics.restocking_cost.toLocaleString('en-IN', { maximumFractionDigits: 0 });
        
        renderCharts();
        renderCustomerTable();
    } catch (err) {
        console.error('Core Engine Error:', err);
    }
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    }[char]));
}

function formatCurrency(value) {
    const isNegative = value < 0;
    const absVal = Math.abs(value);
    const formatted = '₹' + Number(absVal || 0).toLocaleString('en-IN', { maximumFractionDigits: 0 });
    return isNegative ? '-' + formatted : formatted;
}

function renderCustomerTable(filter = '') {
    const tbody = document.getElementById('customer-table-body');
    if (!tbody || !dashboardData || !dashboardData.data) return;

    const query = filter.trim().toLowerCase();
    const rows = dashboardData.data
        .filter((customer) => {
            const haystack = `${customer.CustomerID} ${customer.Persona}`.toLowerCase();
            return haystack.includes(query);
        })
        .sort((a, b) => b.Monetary - a.Monetary)
        .slice(0, 100);

    if (!rows.length) {
        tbody.innerHTML = '<tr><td colspan="7">No matching customers found.</td></tr>';
        return;
    }

    tbody.innerHTML = rows.map((customer) => `
        <tr>
            <td>${escapeHtml(customer.CustomerID)}</td>
            <td><span class="persona-pill" data-persona="${escapeHtml(customer.Persona)}">${escapeHtml(customer.Persona)}</span></td>
            <td>${Number(customer.Recency).toFixed(0)} days</td>
            <td>${Number(customer.Frequency).toFixed(0)}</td>
            <td>${formatCurrency(customer.Monetary)}</td>
            <td style="font-weight: 700; color: ${customer.ReturnRate > 0.3 ? '#ef4444' : 'inherit'}">
                ${(Number(customer.ReturnRate || 0) * 100).toFixed(1)}%
            </td>
            <td><a class="table-link" href="/customer/${encodeURIComponent(customer.CustomerID)}">View</a></td>
        </tr>
    `).join('');
}

function renderCharts() {
    if (!dashboardData || !dashboardData.data) return;
    
    const isDark = document.body.classList.contains('dark-theme');
    const paperBg = 'rgba(0,0,0,0)';
    const textColor = isDark ? '#f8fafc' : '#0f172a';
    const gridColor = isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)';

    // Uniform persona color mapping across all charts and UI components
    const colors = {
        "VIP Buyer": '#10b981',       // Emerald
        "Low-Value Buyer": '#f59e0b',   // Amber
        "Serial Returner": '#ef4444',   // Rose
        "Unusual Activity Outlier": '#a78bfa' // Purple
    };

    const traces = [];
    const personas = ["VIP Buyer", "Low-Value Buyer", "Serial Returner", "Unusual Activity Outlier"];

    personas.forEach(pName => {
        const pData = dashboardData.data.filter(item => item.Persona === pName);
        if (pData.length === 0) return;
        
        traces.push({
            x: pData.map(item => item.Recency),
            y: pData.map(item => item.Frequency),
            z: pData.map(item => item.Monetary),
            customdata: pData.map(item => item.CustomerID),
            hoverinfo: 'text',
            text: pData.map(item => `ID: ${item.CustomerID}<br>Persona: ${item.Persona}<br>Return Rate: ${(Number(item.ReturnRate || 0)*100).toFixed(1)}%`),
            mode: 'markers',
            marker: { size: 4, color: colors[pName] || '#64748b', opacity: 0.8 },
            type: 'scatter3d',
            name: pName
        });
    });

    const layout3d = {
        scene: {
            xaxis: { title: 'Recency (Days)', color: textColor, gridcolor: gridColor },
            yaxis: { title: 'Frequency', color: textColor, gridcolor: gridColor },
            zaxis: { title: 'Monetary (₹)', color: textColor, gridcolor: gridColor },
            bgcolor: paperBg
        },
        margin: { l: 0, r: 0, b: 0, t: 0 },
        paper_bgcolor: paperBg,
        font: { family: 'Outfit, sans-serif', color: textColor },
        legend: { orientation: 'h', y: 0.95, font: { size: 12 } }
    };

    Plotly.newPlot('chart-3d', traces, layout3d, { responsive: true, displayModeBar: false });

    // Add click event listener to the 3D chart
    const chart3d = document.getElementById('chart-3d');
    chart3d.on('plotly_click', function(data) {
        if (data.points.length > 0) {
            const point = data.points[0];
            const customerId = point.customdata;
            if (customerId) {
                window.location.href = `/customer/${customerId}`;
            }
        }
    });

    loadAnalyticsCharts(isDark, textColor, paperBg, colors);
}

async function loadAnalyticsCharts(isDark, textColor, paperBg, colors) {
    if (!dashboardData || !dashboardData.data) return;

    const segmentData = dashboardData.data;
    const personas = ["VIP Buyer", "Low-Value Buyer", "Serial Returner", "Unusual Activity Outlier"];

    const metrics = personas.map(pName => {
        const pData = segmentData.filter(d => d.Persona === pName);
        const count = pData.length;
        const avgReturnRate = count > 0 
            ? (pData.reduce((sum, d) => sum + (d.ReturnRate || 0), 0) / count) * 100 
            : 0;
        return {
            Persona: pName,
            Count: count,
            ReturnRate: avgReturnRate,
            Color: colors[pName] || '#64748b'
        };
    }).filter(m => m.Count > 0);

    // Pie Chart
    const pieData = [{
        values: metrics.map(m => m.Count),
        labels: metrics.map(m => m.Persona),
        type: 'pie',
        marker: { colors: metrics.map(m => m.Color), line: { color: isDark ? '#020617' : '#fff', width: 2 } },
        hole: 0.5,
        textinfo: 'percent+label'
    }];
    
    Plotly.newPlot('chart-pie', pieData, {
        paper_bgcolor: paperBg, 
        font: { family: 'Outfit', color: textColor },
        margin: { t: 40, b: 0, l: 0, r: 0 },
        showlegend: false
    }, { responsive: true, displayModeBar: false });

    // Bar Chart (constrain range 0-100%)
    const barData = [{
        x: metrics.map(m => m.Persona),
        y: metrics.map(m => m.ReturnRate),
        type: 'bar',
        marker: { color: metrics.map(m => m.Color), line: { width: 0 } },
        text: metrics.map(m => m.ReturnRate.toFixed(1) + '%'),
        textposition: 'auto'
    }];
    
    Plotly.newPlot('chart-bar', barData, {
        paper_bgcolor: paperBg, 
        plot_bgcolor: paperBg,
        font: { family: 'Outfit', color: textColor },
        yaxis: { 
            title: 'Avg Return Rate (%)', 
            gridcolor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
            range: [0, 100]
        },
        xaxis: { gridcolor: 'transparent' },
        margin: { t: 40, b: 40, l: 60, r: 20 }
    }, { responsive: true, displayModeBar: false });
}

// Predictor Logic
document.getElementById('predictor-form')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const resDiv = document.getElementById('prediction-result');
    const resPersona = document.getElementById('res-persona');
    const resAdvice = document.getElementById('res-advice');

    const data = {
        recency: document.getElementById('pred-recency').value,
        frequency: document.getElementById('pred-frequency').value,
        monetary: document.getElementById('pred-monetary').value,
        return_rate: document.getElementById('pred-return-rate').value
    };

    try {
        const resp = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const result = await resp.json();

        if (result.error) throw new Error(result.error);

        const adviceMap = {
            "VIP Buyer": "High-value customer. Keep engaged with reward points and exclusive collections.",
            "Serial Returner": "Frequent returns. Charge restocking fees and verify sizing profiles.",
            "Low-Value Buyer": "Low spend and return levels. Nurture with bundle offers to grow net basket size.",
            "Unusual Activity Outlier": "Atypical behavior patterns. Recommend auditing account activity manually."
        };

        resPersona.innerText = result.persona;
        resAdvice.innerText = adviceMap[result.persona] || "This customer segment requires further analysis to determine the best shipping/returns policy.";
        resDiv.style.display = 'block';
    } catch (err) { alert(err.message); }
});

// Churn Predictor Logic (Modified to Return Risk Predictor)
document.getElementById('churn-predictor-form')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const resDiv = document.getElementById('churn-prediction-result');
    const resPrediction = document.getElementById('churn-res-prediction');
    const resAdvice = document.getElementById('churn-res-advice');

    const data = {
        frequency: document.getElementById('churn-frequency').value,
        monetary: document.getElementById('churn-monetary').value
    };

    try {
        const resp = await fetch('/predict_churn', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const result = await resp.json();

        if (result.error) throw new Error(result.error);

        const probability = (result.churn_probability * 100).toFixed(1);
        if (result.churn_prediction === 1) {
            resPrediction.innerText = `High Return Risk (${probability}%)`;
            resAdvice.innerText = "This customer profile is highly likely to return purchases. Restrict free return shipping for this order.";
            resDiv.className = "insight-card glass cluster-lost"; // Uses red warning styling
        } else {
            resPrediction.innerText = `Low Return Risk (${probability}%)`;
            resAdvice.innerText = "This customer profile is low risk. standard return and shipping policies apply.";
            resDiv.className = "insight-card glass cluster-champions"; // Uses green styling
        }
        
        resDiv.style.display = 'block';
    } catch (err) { 
        alert(err.message); 
    }
});

function setUploadStatus(message, type = 'success') {
    document.querySelectorAll('.upload-status').forEach((status) => {
        status.innerText = message;
        status.className = `upload-status status-msg status-${type}`;
        status.hidden = false;
    });
}

async function uploadCsv(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.csv')) {
        setUploadStatus('Please choose a CSV file.', 'error');
        return;
    }

    const zeroState = document.getElementById('zero-state');
    const dashContainer = document.getElementById('dashboard-container');

    const formData = new FormData();
    formData.append('file', file);

    setUploadStatus('Processing data...', 'success');

    try {
        const resp = await fetch('/upload', { method: 'POST', body: formData });
        const result = await resp.json();

        if (result.error) throw new Error(result.error);

        setUploadStatus('Upload successful! Reloading...', 'success');
        setTimeout(() => {
            loadDashboard();
            openTab(null, 'dashboard');
        }, 1500);

    } catch (err) {
        setUploadStatus(err.message, 'error');
    }
}

// Drag & Drop
document.addEventListener('DOMContentLoaded', () => {
    const dropZones = ['drop-zone', 'drop-zone-sidebar'];
    
    dropZones.forEach(id => {
        const zone = document.getElementById(id);
        if (!zone) return;

        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            zone.classList.add('dragover');
        });

        zone.addEventListener('dragleave', () => {
            zone.classList.remove('dragover');
        });

        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            uploadCsv(file);
        });

        zone.addEventListener('click', () => {
            const input = document.getElementById('csv-upload');
            if (input) input.click();
        });
    });

    const fileInput = document.getElementById('csv-upload');
    fileInput?.addEventListener('change', (e) => {
        const file = e.target.files[0];
        uploadCsv(file);
    });

    // Theme Toggle
    const themeToggle = document.getElementById('theme-toggle');
    if (localStorage.getItem('theme') === 'dark') {
        document.body.className = 'dark-theme';
        if (themeToggle) themeToggle.checked = true;
    }

    themeToggle?.addEventListener('change', (e) => {
        const theme = e.target.checked ? 'dark' : 'light';
        document.body.className = theme + '-theme';
        localStorage.setItem('theme', theme);
        renderCharts();
    });

    // Search
    document.getElementById('customer-search')?.addEventListener('input', (e) => {
        renderCustomerTable(e.target.value);
    });

    loadDashboard();
});
