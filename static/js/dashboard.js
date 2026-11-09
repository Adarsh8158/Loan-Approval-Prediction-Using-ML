/**
 * Bank Admin Dashboard - Interactive Charts, Dynamic Modals & Real-time Prediction
 * Handles Chart.js initialization, application state, ML predictions, and all modal dialogs.
 */

document.addEventListener("DOMContentLoaded", function () {
    // -------------------------------------------------------------
    // 1. Initial State (matching reference image)
    // -------------------------------------------------------------
    const dashboardState = {
        totalLoans: 15,
        totalDisbursed: 0,
        pendingReview: 3,
        approvedLoans: 6,
        disbursedLoans: 1,
        rejectedLoans: 8
    };

    // Calculate initial approval rate
    function updateMetricsUI() {
        const approvalRate = Math.round((dashboardState.approvedLoans / dashboardState.totalLoans) * 100) || 40;
        document.getElementById("kpiTotalLoans").innerText = dashboardState.totalLoans;
        document.getElementById("kpiTotalDisbursed").innerText = "₹" + dashboardState.totalDisbursed.toLocaleString("en-IN");
        document.getElementById("kpiPendingReview").innerText = dashboardState.pendingReview;
        document.getElementById("kpiApprovalRate").innerText = approvalRate + "%";
    }

    // -------------------------------------------------------------
    // 2. Loan Status Distribution (Donut Chart)
    // -------------------------------------------------------------
    const donutCtx = document.getElementById("donutChart").getContext("2d");
    const statusDonutChart = new Chart(donutCtx, {
        type: "doughnut",
        data: {
            labels: ["Disbursed", "Approved", "Pending", "Rejected"],
            datasets: [{
                data: [
                    dashboardState.disbursedLoans,
                    dashboardState.approvedLoans,
                    dashboardState.pendingReview,
                    dashboardState.rejectedLoans
                ],
                backgroundColor: ["#06b6d4", "#10b981", "#f59e0b", "#f43f5e"],
                borderColor: "#0f172a",
                borderWidth: 3,
                hoverOffset: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: "72%",
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "rgba(15, 23, 42, 0.95)",
                    titleColor: "#ffffff",
                    bodyColor: "#94a3b8",
                    borderColor: "rgba(255, 255, 255, 0.1)",
                    borderWidth: 1,
                    padding: 10
                }
            }
        }
    });

    // -------------------------------------------------------------
    // 3. Monthly Trends (Line Chart)
    // -------------------------------------------------------------
    const lineCtx = document.getElementById("trendChart").getContext("2d");
    const trendChart = new Chart(lineCtx, {
        type: "line",
        data: {
            labels: ["Oct", "Nov", "Dec", "Jan"],
            datasets: [
                {
                    label: "Applications",
                    data: [3.0, 5.2, 4.1, 1.8],
                    borderColor: "#14b8a6",
                    backgroundColor: "rgba(20, 184, 166, 0.08)",
                    borderWidth: 2.5,
                    tension: 0.45,
                    fill: false,
                    pointBackgroundColor: "#14b8a6",
                    pointBorderColor: "#ffffff",
                    pointRadius: 4,
                    pointHoverRadius: 6
                },
                {
                    label: "Approved",
                    data: [2.0, 4.0, 3.8, 1.0],
                    borderColor: "#8b5cf6",
                    backgroundColor: "rgba(139, 92, 246, 0.08)",
                    borderWidth: 2.5,
                    tension: 0.45,
                    fill: false,
                    pointBackgroundColor: "#8b5cf6",
                    pointBorderColor: "#ffffff",
                    pointRadius: 4,
                    pointHoverRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: "rgba(255, 255, 255, 0.04)" },
                    ticks: { color: "#64748b", font: { size: 11 } }
                },
                y: {
                    min: 0,
                    max: 8,
                    ticks: { stepSize: 2, color: "#64748b", font: { size: 11 } },
                    grid: { color: "rgba(255, 255, 255, 0.05)" }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "rgba(15, 23, 42, 0.95)",
                    borderColor: "rgba(255, 255, 255, 0.1)",
                    borderWidth: 1,
                    padding: 10
                }
            }
        }
    });

    // -------------------------------------------------------------
    // 4. Toast Notification Manager
    // -------------------------------------------------------------
    window.showToast = function (message, title = "System Notification") {
        let toast = document.getElementById("welcomeToast");
        if (!toast) {
            toast = document.createElement("div");
            toast.id = "welcomeToast";
            toast.className = "toast-box";
            document.body.appendChild(toast);
        }
        toast.innerHTML = `<h4>${title}</h4><p>${message}</p>`;
        toast.style.display = "flex";
        toast.style.opacity = "1";
        toast.style.transform = "translateY(0)";

        setTimeout(() => {
            toast.style.transition = "opacity 0.6s ease, transform 0.6s ease";
            toast.style.opacity = "0";
            toast.style.transform = "translateY(20px)";
        }, 4500);
    };

    // Auto dismiss initial toast
    setTimeout(() => {
        const toast = document.getElementById("welcomeToast");
        if (toast) {
            toast.style.transition = "opacity 0.6s ease, transform 0.6s ease";
            toast.style.opacity = "0";
            toast.style.transform = "translateY(20px)";
        }
    }, 6000);

    // -------------------------------------------------------------
    // 5. Universal Modal Manager
    // -------------------------------------------------------------
    window.openModal = function (modalId) {
        const target = document.getElementById(modalId);
        if (target) {
            target.classList.add("active");
        }
    };

    window.closeModal = function (modalId) {
        const target = document.getElementById(modalId);
        if (target) {
            target.classList.remove("active");
        }
    };

    // Attach close listener to all elements with data-close
    document.querySelectorAll("[data-close]").forEach(el => {
        el.addEventListener("click", () => {
            const modalId = el.getAttribute("data-close");
            window.closeModal(modalId);
        });
    });

    // Close when clicking outside modal card
    document.querySelectorAll(".modal-overlay").forEach(overlay => {
        overlay.addEventListener("click", (e) => {
            if (e.target === overlay) {
                overlay.classList.remove("active");
            }
        });
    });

    // -------------------------------------------------------------
    // 6. Connect All Navigation & Dashboard Icons
    // -------------------------------------------------------------
    // Sidebar items
    const navDashboard = document.getElementById("navDashboard");
    if (navDashboard) {
        navDashboard.addEventListener("click", () => {
            window.scrollTo({ top: 0, behavior: "smooth" });
            document.querySelectorAll(".nav-link").forEach(l => l.classList.remove("active"));
            navDashboard.classList.add("active");
        });
    }

    const connectClick = (id, modalId) => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener("click", (e) => {
                e.preventDefault();
                window.openModal(modalId);
            });
        }
    };

    connectClick("navLoanApplications", "predictionModal");
    connectClick("btnOpenPredictor", "predictionModal");
    connectClick("cardTotalLoans", "predictionModal");

    connectClick("navDisbursement", "disbursementModal");
    connectClick("cardTotalDisbursed", "disbursementModal");
    connectClick("cardPendingReview", "disbursementModal");

    connectClick("navEmiTracking", "emiModal");
    connectClick("navDocuments", "documentsModal");

    connectClick("navNotifications", "notificationsModal");
    connectClick("topbarBell", "notificationsModal");

    connectClick("navCustomers", "customersModal");

    connectClick("navReports", "reportsModal");
    connectClick("cardApprovalRate", "reportsModal");

    connectClick("navSupportTickets", "supportModal");

    connectClick("adminProfileCard", "profileModal");
    connectClick("topbarUser", "profileModal");

    // Pill Badges
    const badgeActive = document.getElementById("badgeActiveStatus");
    if (badgeActive) {
        badgeActive.addEventListener("click", () => {
            window.showToast("All financial APIs and automated verification services are online.", "System Status: Active");
        });
    }

    const badgeAi = document.getElementById("badgeAiStatus");
    if (badgeAi) {
        badgeAi.addEventListener("click", () => {
            window.showToast("Random Forest Classifier model loaded in memory (Latency < 25ms).", "AI Engine: Ready");
        });
    }

    const adminPill = document.getElementById("topbarAdminPill");
    if (adminPill) {
        adminPill.addEventListener("click", () => {
            window.showToast("Logged in with superuser privileges (Full read/write/audit permissions).", "Admin Authorization");
        });
    }

    // Sign out button
    const btnSignOut = document.getElementById("btnSignOut");
    if (btnSignOut) {
        btnSignOut.addEventListener("click", () => {
            if (confirm("Are you sure you want to end your session?")) {
                window.showToast("You have been signed out. Refresh the page to log back in.", "Session Terminated");
            }
        });
    }

    // Mobile menu toggle
    const menuToggle = document.getElementById("menuToggle");
    const sidebar = document.querySelector(".sidebar");
    if (menuToggle && sidebar) {
        menuToggle.addEventListener("click", () => {
            sidebar.classList.toggle("mobile-open");
        });
    }

    // -------------------------------------------------------------
    // 7. Interactive EMI Modeler Sliders
    // -------------------------------------------------------------
    const rngP = document.getElementById("rngPrincipal");
    const rngT = document.getElementById("rngTenure");
    const rngR = document.getElementById("rngRate");

    const lblP = document.getElementById("lblCalcPrincipal");
    const lblT = document.getElementById("lblCalcTenure");
    const lblR = document.getElementById("lblCalcRate");
    const emiDisplay = document.getElementById("calcMonthlyEmiDisplay");

    function calculateInteractiveEmi() {
        if (!rngP || !rngT || !rngR) return;
        const P = parseFloat(rngP.value);
        const T = parseFloat(rngT.value);
        const R = parseFloat(rngR.value);

        lblP.innerText = "₹" + P.toLocaleString("en-IN");
        lblT.innerText = T + " Years";
        lblR.innerText = R.toFixed(2) + "%";

        const monthlyRate = (R / 12) / 100;
        const n = T * 12;
        const emi = (P * monthlyRate * Math.pow(1 + monthlyRate, n)) / (Math.pow(1 + monthlyRate, n) - 1);
        emiDisplay.innerText = "₹" + Math.round(emi).toLocaleString("en-IN");
    }

    if (rngP && rngT && rngR) {
        [rngP, rngT, rngR].forEach(slider => {
            slider.addEventListener("input", calculateInteractiveEmi);
        });
        calculateInteractiveEmi();
    }

    // -------------------------------------------------------------
    // 8. Interactive Disbursement Processing
    // -------------------------------------------------------------
    window.processDisburse = function (button, amount) {
        button.disabled = true;
        button.innerText = "Disbursing...";

        setTimeout(() => {
            button.innerText = "Disbursed ✓";
            button.style.background = "#059669";
            dashboardState.totalDisbursed += amount;
            dashboardState.disbursedLoans += 1;
            if (dashboardState.pendingReview > 0) dashboardState.pendingReview -= 1;
            updateMetricsUI();

            // Update donut chart
            statusDonutChart.data.datasets[0].data = [
                dashboardState.disbursedLoans,
                dashboardState.approvedLoans,
                dashboardState.pendingReview,
                dashboardState.rejectedLoans
            ];
            statusDonutChart.update();

            window.showToast(`₹${amount.toLocaleString('en-IN')} electronic payment initiated.`, "Funds Disbursed");
        }, 600);
    };

    // -------------------------------------------------------------
    // 9. Support Ticket Form Submission
    // -------------------------------------------------------------
    window.handleSupportSubmit = function (e) {
        e.preventDefault();
        window.closeModal("supportModal");
        window.showToast("Support ticket #TCK-9281 logged. Our support team will review within 2 hours.", "Ticket Submitted");
    };

    // -------------------------------------------------------------
    // 10. ML Loan Prediction Form Submission
    // -------------------------------------------------------------
    const predictForm = document.getElementById("loanPredictionForm");
    const resultBox = document.getElementById("modalResult");

    if (predictForm) {
        predictForm.addEventListener("submit", async function (e) {
            e.preventDefault();
            const submitBtn = predictForm.querySelector("button[type='submit']");
            submitBtn.disabled = true;
            submitBtn.innerText = "Analyzing Risk Profile...";

            const formData = new FormData(predictForm);
            const payload = Object.fromEntries(formData.entries());

            try {
                const response = await fetch("/api/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();

                if (!response.ok || data.error) {
                    alert("Error: " + (data.error || "Failed to make prediction."));
                    return;
                }

                // Display Results
                resultBox.style.display = "block";
                resultBox.className = "modal-result " + (data.is_approved ? "approved" : "rejected");
                resultBox.innerHTML = `
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h3 style="font-size:16px;">Decision: Loan ${data.status}</h3>
                        <span style="font-size:12px; font-weight:700;">Confidence: ${data.confidence}%</span>
                    </div>
                    <p style="font-size:12px; margin-top:6px; opacity:0.9;">
                        CIBIL Rating: <strong>${data.cibil_rating}</strong> | 
                        Estimated Monthly EMI: <strong>₹${Number(data.monthly_emi).toLocaleString('en-IN')}</strong>
                    </p>
                `;

                // Update local dashboard counters
                dashboardState.totalLoans += 1;
                if (data.is_approved) {
                    dashboardState.approvedLoans += 1;
                } else {
                    dashboardState.rejectedLoans += 1;
                }
                if (dashboardState.pendingReview > 0) {
                    dashboardState.pendingReview -= 1;
                }

                updateMetricsUI();

                // Update Donut chart
                statusDonutChart.data.datasets[0].data = [
                    dashboardState.disbursedLoans,
                    dashboardState.approvedLoans,
                    dashboardState.pendingReview,
                    dashboardState.rejectedLoans
                ];
                statusDonutChart.update();

                // Add row to recent applications table
                const tableBody = document.getElementById("applicationsTableBody");
                if (tableBody) {
                    const tr = document.createElement("tr");
                    const badgeClass = data.is_approved ? "status-approved" : "status-rejected";
                    tr.innerHTML = `
                        <td>#LN-${Math.floor(1000 + Math.random() * 9000)}</td>
                        <td>Applicant (Score: ${payload.cibil_score})</td>
                        <td>₹${Number(payload.loan_amount).toLocaleString('en-IN')}</td>
                        <td>${payload.loan_term} Yrs</td>
                        <td><span class="badge-status ${badgeClass}">${data.status}</span></td>
                        <td>₹${Number(data.monthly_emi).toLocaleString('en-IN')}</td>
                    `;
                    tableBody.prepend(tr);
                }

                window.showToast(`Loan application assessed: ${data.status} (${data.confidence}% confidence).`, "Prediction Complete");

            } catch (err) {
                console.error("Prediction request failed:", err);
                alert("Could not connect to prediction server.");
            } finally {
                submitBtn.disabled = false;
                submitBtn.innerText = "Evaluate Loan Application →";
            }
        });
    }

    // Initial sync
    updateMetricsUI();
});
