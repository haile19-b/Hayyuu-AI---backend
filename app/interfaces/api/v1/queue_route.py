import logging
from datetime import datetime
from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from arq.jobs import Job, JobDef, JobResult
from app.core import queue
from app.core.env import settings

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/queue", tags=["Queue Dashboard"])

def serialize_datetime(dt) -> str:
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt)

def format_job_def(job: JobDef) -> dict:
    return {
        "job_id": job.job_id,
        "function": job.function,
        "args": [str(arg) for arg in job.args],
        "enqueue_time": serialize_datetime(job.enqueue_time)
    }

def format_job_result(job: JobResult) -> dict:
    return {
        "job_id": job.job_id,
        "function": job.function,
        "args": [str(arg) for arg in job.args],
        "success": job.success,
        "result": str(job.result) if job.result else None,
        "enqueue_time": serialize_datetime(job.enqueue_time),
        "start_time": serialize_datetime(job.start_time),
        "finish_time": serialize_datetime(job.finish_time)
    }

@router.get("/status")
async def get_queue_status() -> dict:
    """Fetch status of all queued, active, and completed/failed jobs from Redis."""
    if queue.redis_pool is None:
        await queue.connect_redis()
        
    queued = []
    completed = []
    
    # 1. Fetch currently queued/active jobs
    try:
        queued_jobs = await queue.redis_pool.queued_jobs()
        for j in queued_jobs:
            try:
                job = Job(j.job_id, redis=queue.redis_pool)
                status = await job.status()
                q_info = format_job_def(j)
                q_info["status"] = str(status)
                queued.append(q_info)
            except Exception as job_err:
                # Silently skip expired job keys in Redis queue index
                pass
    except Exception as e:
        logger.error(f"Error fetching queued jobs list: {e}")
        
    # 2. Fetch recently completed/failed jobs
    try:
        job_results = await queue.redis_pool.all_job_results()
        for r in job_results:
            completed.append(format_job_result(r))
    except Exception as e:
        logger.error(f"Error fetching completed jobs: {e}")
        
    return {
        "success": True,
        "queued": queued,
        "completed": completed
    }

@router.post("/test-trigger")
async def trigger_test_job() -> dict:
    """Convenience endpoint to trigger a dummy background job for testing."""
    await queue.enqueue_document_analysis(
        document_id="test-doc-id-dashboard",
        project_id="test-proj-id-dashboard"
    )
    return {"success": True, "message": "Test job enqueued!"}

@router.get("", response_class=HTMLResponse)
async def get_dashboard() -> str:
    """Serve a beautiful, premium, glassmorphic UI dashboard to monitor the queue."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ARQ Task Queue Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-color: #080c14;
                --card-bg: rgba(17, 24, 39, 0.55);
                --border-color: rgba(255, 255, 255, 0.06);
                --primary: #6366f1;
                --primary-glow: rgba(99, 102, 241, 0.4);
                --accent: #a855f7;
                --success: #10b981;
                --danger: #f43f5e;
                --text-main: #f3f4f6;
                --text-muted: #9ca3af;
            }
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Outfit', sans-serif;
            }
            
            body {
                background-color: var(--bg-color);
                color: var(--text-main);
                min-height: 100vh;
                padding: 3rem 2rem;
                background-image: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.12) 0%, transparent 45%),
                                  radial-gradient(circle at 90% 80%, rgba(168, 85, 247, 0.12) 0%, transparent 45%);
                background-attachment: fixed;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            
            header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 3rem;
                border-bottom: 1px solid var(--border-color);
                padding-bottom: 1.5rem;
            }
            
            .logo-area h1 {
                font-size: 2.2rem;
                font-weight: 700;
                background: linear-gradient(135deg, var(--text-main) 30%, #93c5fd 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                letter-spacing: -0.02em;
            }
            
            .logo-area p {
                color: var(--text-muted);
                margin-top: 0.25rem;
                font-size: 0.95rem;
            }
            
            .actions-area {
                display: flex;
                gap: 1rem;
            }
            
            button {
                background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
                color: white;
                border: none;
                padding: 0.75rem 1.5rem;
                border-radius: 12px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                box-shadow: 0 4px 12px var(--primary-glow);
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 18px var(--primary-glow);
            }
            
            button:active {
                transform: translateY(0);
            }
            
            .btn-secondary {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid var(--border-color);
                color: var(--text-main);
                box-shadow: none;
            }
            
            .btn-secondary:hover {
                background: rgba(255, 255, 255, 0.1);
                box-shadow: none;
            }
            
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 1.5rem;
                margin-bottom: 3rem;
            }
            
            .metric-card {
                background: var(--card-bg);
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                border: 1px solid var(--border-color);
                padding: 1.5rem;
                border-radius: 20px;
                display: flex;
                flex-direction: column;
                position: relative;
                overflow: hidden;
            }
            
            .metric-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 4px;
                height: 100%;
                background: var(--primary);
            }
            
            .metric-card.queued::before { background: var(--accent); }
            .metric-card.success::before { background: var(--success); }
            .metric-card.failed::before { background: var(--danger); }
            
            .metric-label {
                color: var(--text-muted);
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                font-weight: 500;
            }
            
            .metric-value {
                font-size: 2.5rem;
                font-weight: 700;
                margin-top: 0.5rem;
                color: var(--text-main);
            }
            
            .section-title {
                font-size: 1.4rem;
                font-weight: 600;
                margin-bottom: 1.5rem;
                color: var(--text-main);
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }
            
            .section-badge {
                font-size: 0.8rem;
                background: rgba(255, 255, 255, 0.08);
                padding: 0.25rem 0.6rem;
                border-radius: 8px;
                color: var(--text-muted);
            }
            
            .dashboard-sections {
                display: grid;
                grid-template-columns: 1fr;
                gap: 3rem;
            }
            
            .table-container {
                background: var(--card-bg);
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                border: 1px solid var(--border-color);
                border-radius: 24px;
                overflow: hidden;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            }
            
            table {
                width: 100%;
                border-collapse: collapse;
                text-align: left;
            }
            
            th {
                background: rgba(255, 255, 255, 0.02);
                padding: 1.2rem 1.5rem;
                font-weight: 600;
                font-size: 0.9rem;
                color: var(--text-muted);
                border-bottom: 1px solid var(--border-color);
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            
            td {
                padding: 1.2rem 1.5rem;
                border-bottom: 1px solid rgba(255, 255, 255, 0.03);
                font-size: 0.95rem;
                vertical-align: middle;
            }
            
            tr:last-child td {
                border-bottom: none;
            }
            
            tr:hover td {
                background: rgba(255, 255, 255, 0.01);
            }
            
            .badge {
                padding: 0.35rem 0.75rem;
                border-radius: 10px;
                font-weight: 600;
                font-size: 0.8rem;
                text-transform: uppercase;
                display: inline-block;
                letter-spacing: 0.03em;
            }
            
            .badge-queued {
                background: rgba(168, 85, 247, 0.15);
                color: #d8b4fe;
                border: 1px solid rgba(168, 85, 247, 0.3);
            }
            
            .badge-running {
                background: rgba(99, 102, 241, 0.15);
                color: #c7d2fe;
                border: 1px solid rgba(99, 102, 241, 0.3);
            }
            
            .badge-success {
                background: rgba(16, 185, 129, 0.15);
                color: #a7f3d0;
                border: 1px solid rgba(16, 185, 129, 0.3);
            }
            
            .badge-failed {
                background: rgba(244, 63, 94, 0.15);
                color: #fecdd3;
                border: 1px solid rgba(244, 63, 94, 0.3);
            }
            
            .code-text {
                font-family: monospace;
                background: rgba(0, 0, 0, 0.3);
                padding: 0.25rem 0.5rem;
                border-radius: 6px;
                font-size: 0.85rem;
                border: 1px solid rgba(255, 255, 255, 0.03);
            }
            
            .empty-state {
                padding: 4rem 2rem;
                text-align: center;
                color: var(--text-muted);
            }
            
            .empty-state svg {
                width: 48px;
                height: 48px;
                stroke: var(--text-muted);
                margin-bottom: 1rem;
                opacity: 0.5;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <div class="logo-area">
                    <h1>ARQ Queue Monitor</h1>
                    <p>Real-time Async Task Queue Dashboard</p>
                </div>
                <div class="actions-area">
                    <button class="btn-secondary" onclick="refreshDashboard()">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
                        Refresh
                    </button>
                    <button onclick="triggerTestJob()">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14M5 12h14"/></svg>
                        Test Trigger Job
                    </button>
                </div>
            </header>

            <div class="metrics-grid">
                <div class="metric-card active">
                    <span class="metric-label">Active / Running</span>
                    <span class="metric-value" id="metric-active">0</span>
                </div>
                <div class="metric-card queued">
                    <span class="metric-label">Queued / Scheduled</span>
                    <span class="metric-value" id="metric-queued">0</span>
                </div>
                <div class="metric-card success">
                    <span class="metric-label">Completed successfully</span>
                    <span class="metric-value" id="metric-success">0</span>
                </div>
                <div class="metric-card failed">
                    <span class="metric-label">Failed execution</span>
                    <span class="metric-value" id="metric-failed">0</span>
                </div>
            </div>

            <div class="dashboard-sections">
                <div>
                    <h2 class="section-title">
                        Active & Queued Tasks
                        <span class="section-badge" id="badge-queued-count">0</span>
                    </h2>
                    <div class="table-container">
                        <table id="active-tasks-table">
                            <thead>
                                <tr>
                                    <th>Job ID</th>
                                    <th>Task Name</th>
                                    <th>Arguments</th>
                                    <th>Enqueued At</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody id="active-tasks-body">
                                <tr>
                                    <td colspan="5" class="empty-state">
                                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                                        <p>No active or queued jobs found</p>
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <div>
                    <h2 class="section-title">
                        Completed Job History
                        <span class="section-badge" id="badge-completed-count">0</span>
                    </h2>
                    <div class="table-container">
                        <table id="history-tasks-table">
                            <thead>
                                <tr>
                                    <th>Job ID</th>
                                    <th>Task Name</th>
                                    <th>Duration</th>
                                    <th>Execution Status</th>
                                    <th>Result / Error</th>
                                </tr>
                            </thead>
                            <tbody id="history-tasks-body">
                                <tr>
                                    <td colspan="5" class="empty-state">
                                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
                                        <p>No completed jobs found</p>
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <script>
            async function fetchStatus() {
                try {
                    const res = await fetch('/api/v1/queue/status');
                    return await res.json();
                } catch (e) {
                    console.error("Failed to load queue status:", e);
                    return null;
                }
            }

            async function refreshDashboard() {
                const data = await fetchStatus();
                if (!data) return;

                const queued = data.queued || [];
                const completed = data.completed || [];

                // 1. Calculate metric totals
                const runningCount = queued.filter(j => j.status === 'in_progress').length;
                const queuedCount = queued.filter(j => j.status !== 'in_progress').length;
                const successCount = completed.filter(j => j.success).length;
                const failedCount = completed.filter(j => !j.success).length;

                document.getElementById('metric-active').innerText = runningCount;
                document.getElementById('metric-queued').innerText = queuedCount;
                document.getElementById('metric-success').innerText = successCount;
                document.getElementById('metric-failed').innerText = failedCount;

                // 2. Populate active/queued table
                const activeBody = document.getElementById('active-tasks-body');
                document.getElementById('badge-queued-count').innerText = queued.length;
                
                if (queued.length === 0) {
                    activeBody.innerHTML = `
                        <tr>
                            <td colspan="5" class="empty-state">
                                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                                <p>No active or queued jobs found</p>
                            </td>
                        </tr>
                    `;
                } else {
                    activeBody.innerHTML = queued.map(j => {
                        const isRunning = j.status === 'in_progress';
                        const statusClass = isRunning ? 'badge-running' : 'badge-queued';
                        const statusLabel = isRunning ? 'running' : 'queued';
                        const enqueuedTime = new Date(j.enqueue_time).toLocaleTimeString();
                        
                        return `
                            <tr>
                                <td><span class="code-text">${j.job_id.substring(0, 8)}...</span></td>
                                <td><strong>${j.function}</strong></td>
                                <td><span class="code-text">${JSON.stringify(j.args)}</span></td>
                                <td>${enqueuedTime}</td>
                                <td><span class="badge ${statusClass}">${statusLabel}</span></td>
                            </tr>
                        `;
                    }).join('');
                }

                // 3. Populate history table
                const historyBody = document.getElementById('history-tasks-body');
                document.getElementById('badge-completed-count').innerText = completed.length;
                
                if (completed.length === 0) {
                    historyBody.innerHTML = `
                        <tr>
                            <td colspan="5" class="empty-state">
                                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
                                <p>No completed jobs found</p>
                            </td>
                        </tr>
                    `;
                } else {
                    historyBody.innerHTML = completed.map(j => {
                        const durationSec = ((new Date(j.finish_time) - new Date(j.start_time)) / 1000).toFixed(2);
                        const statusClass = j.success ? 'badge-success' : 'badge-failed';
                        const statusLabel = j.success ? 'success' : 'failed';
                        const resultText = j.success ? (j.result || 'None') : (j.result || 'Error occurred');
                        
                        return `
                            <tr>
                                <td><span class="code-text">${j.job_id.substring(0, 8)}...</span></td>
                                <td><strong>${j.function}</strong></td>
                                <td>${durationSec}s</td>
                                <td><span class="badge ${statusClass}">${statusLabel}</span></td>
                                <td><span class="code-text" style="color: ${j.success ? '#a7f3d0' : '#fecdd3'}">${resultText}</span></td>
                            </tr>
                        `;
                    }).join('');
                }
            }

            async function triggerTestJob() {
                try {
                    const res = await fetch('/api/v1/queue/test-trigger', { method: 'POST' });
                    const data = await res.json();
                    if (data.success) {
                        refreshDashboard();
                    }
                } catch (e) {
                    console.error("Failed to trigger job:", e);
                }
            }

            // Auto refresh every 2 seconds
            setInterval(refreshDashboard, 2000);
            window.onload = refreshDashboard;
        </script>
    </body>
    </html>
    """
    return html_content
