@echo off
cd /d C:\Users\joshm\nba_props_research_tool\nba_props

:loop
python src/ingest_boxscores.py --limit 300
echo Waiting 3 minutes before next run...
timeout /t 180 /nobreak
goto loop
