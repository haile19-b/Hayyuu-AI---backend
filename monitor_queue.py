import asyncio
import os
import sys
import time
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
import redis.asyncio as aioredis
from app.core.env import settings

console = Console()

async def fetch_queue_stats(redis_client):
    try:
        # ARQ queue is a Redis sorted set
        queue_len = await redis_client.zcard("arq:queue")
        pending_job_ids = await redis_client.zrange("arq:queue", 0, 9)
        
        # All job keys in Redis
        all_job_keys = await redis_client.keys("arq:job:*")
        health_check = await redis_client.get("arq:health-check")
        
        jobs_info = []
        for key in all_job_keys[-10:]:
            raw_id = key.decode().replace("arq:job:", "")
            job_data = await redis_client.hgetall(key)
            decoded = {k.decode(): v.decode(errors="replace") for k, v in job_data.items()} if job_data else {}
            jobs_info.append((raw_id, decoded))

        return {
            "queue_len": queue_len,
            "total_tracked_jobs": len(all_job_keys),
            "pending_ids": [j.decode() for j in pending_job_ids],
            "recent_jobs": jobs_info,
            "connected": True,
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}

def build_display(stats):
    if not stats["connected"]:
        return Panel(f"[bold red]❌ Failed to connect to Redis at {settings.REDIS_URL}: {stats.get('error')}[/]", title="Queue Status")

    table = Table(title="📋 ARQ Task Queue Status", expand=True)
    table.add_column("Job ID", style="cyan", no_wrap=True)
    table.add_column("Function", style="magenta")
    table.add_column("Status / State", style="green")
    table.add_column("Enqueued At", style="yellow")

    if not stats["recent_jobs"]:
        table.add_row("-", "No active or recent jobs", "IDLE (Waiting for tasks)", "-")
    else:
        for job_id, data in stats["recent_jobs"]:
            func_name = data.get("f", "analyze_document_job")
            status = "⏳ QUEUED" if job_id in stats["pending_ids"] else "⚡ RUNNING / COMPLETED"
            enqueued = data.get("enqueue_time", datetime.now().strftime("%H:%M:%S"))
            table.add_row(job_id[:18] + "...", func_name, status, enqueued)

    summary = (
        f"[bold cyan]Redis URL:[/] {settings.REDIS_URL}  |  "
        f"[bold yellow]Pending Queue Length:[/] [bold]{stats['queue_len']}[/]  |  "
        f"[bold green]Tracked Jobs:[/] {stats['total_tracked_jobs']}  |  "
        f"[bold magenta]Time:[/] {datetime.now().strftime('%H:%M:%S')}"
    )
    
    return Panel(table, title="[bold cyan]Hayyuu AI Worker Dashboard[/]", subtitle=summary)

async def main():
    redis_client = aioredis.from_url(settings.REDIS_URL)
    console.print("[green]Connecting to Redis monitor...[/]")
    with Live(console=console, refresh_per_second=2) as live:
        while True:
            stats = await fetch_queue_stats(redis_client)
            live.update(build_display(stats))
            await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Queue monitor stopped.[/]")
