<#
.SYNOPSIS
    Registers the depangram night run with Task Scheduler: "depangram night", daily at 02:00 (or -At).

.DESCRIPTION
    The task runs the depangram Python with -m engine.night in the depangram folder (the folder this script sits
    in), with no window, for this user only. The Python is -Python when given, else DEPANGRAM_PYTHON from the
    environment, else venv\Scripts\python.exe or .venv\Scripts\python.exe in this folder. -Job names the job
    (else the night takes [job] default from config.toml), and -At the start time. It wakes the computer to run, runs only while you are
    logged on (it needs your git credentials for the push and the publish, and the GPU), starts as soon as it can
    when the start time was missed, never starts a second copy while one is running, and is stopped after 8 hours.

    No admin rights are needed to register it. Waking the computer also needs wake timers allowed in the power
    plan, which this script reads but never changes; when they are off it prints the line to run.

.EXAMPLE
    & ".\night task.ps1"            # register the task, or update it in place
    & ".\night task.ps1" -Job my-doc -At 01:30 -Python C:\path\to\python.exe
    & ".\night task.ps1" -Show      # its state, last and next run, wake timers, and the last night's own log
    & ".\night task.ps1" -Remove    # unregister it
#>
[CmdletBinding()]
param(
    [switch]$Remove,
    [switch]$Show,
    [string]$Job = '',
    [string]$At = '02:00',
    [string]$Python = ''
)

$ErrorActionPreference = 'Stop'
$TaskName = 'depangram night'
$Folder = $PSScriptRoot
if (-not $Python) { $Python = $env:DEPANGRAM_PYTHON }
if (-not $Python) {
    foreach ($venv in 'venv', '.venv') {
        $cand = Join-Path $Folder "$venv\Scripts\python.exe"
        if (Test-Path -LiteralPath $cand) { $Python = $cand; break }
    }
}
$NightArgs = '-m engine.night'
if ($Job) { $NightArgs += " --job `"$Job`"" }
$User = "$env:USERDOMAIN\$env:USERNAME"
# conhost --headless gives the night a console nobody sees, so no window opens at 02:00. pythonw has no stdout at
# all, and every depangram command writes to it. If conhost --headless ever fails here, the plain fallback is
# -Execute $Python -Argument $NightArgs, which works the same but shows a console window while it runs.
$Conhost = Join-Path $env:SystemRoot 'System32\conhost.exe'

function Show-WakeTimers {
    # powercfg /query only reads, so it needs no admin rights. RTCWAKE is "Allow wake timers": 0 off, 1 on,
    # 2 important wake timers only (which leaves a scheduled task asleep).
    $out = powercfg /query SCHEME_CURRENT SUB_SLEEP RTCWAKE 2>$null
    $fix = 'powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP RTCWAKE 1; powercfg /setactive SCHEME_CURRENT'
    $ac = $null
    if ($out) {
        $m = $out | Select-String -Pattern 'Current AC Power Setting Index:\s*0x([0-9a-fA-F]+)'
        if ($m) { $ac = [Convert]::ToInt32($m.Matches[0].Groups[1].Value, 16) }
    }
    if ($null -eq $ac) {
        Write-Host 'Wake timers: could not read the setting. Check it by hand: Control Panel, Power Options,'
        Write-Host '  Change plan settings, Change advanced power settings, Sleep, Allow wake timers: Enable (plugged in).'
    }
    elseif ($ac -eq 1) {
        Write-Host 'Wake timers: allowed on mains power, so the task can wake the computer from sleep.'
    }
    else {
        Write-Host 'Wake timers: not allowed on mains power, so the task waits until the computer is awake.'
        Write-Host "  To allow them, run this (from an admin prompt if it says access is denied): $fix"
    }
}

if ($Remove) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed the task '$TaskName'."
    }
    else {
        Write-Host "There is no task '$TaskName' to remove."
    }
    return
}

if ($Show) {
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $task) {
        Write-Host "The task '$TaskName' is not registered. Run this script without -Show to register it."
        exit 1
    }
    $info = $task | Get-ScheduledTaskInfo
    $result = $info.LastTaskResult
    # the result is conhost's, the launcher's: it exits 0 whatever Python's exit code was, so 0 only says the
    # launcher started and ended. The night's own log below says how the night went.
    $meaning = switch ($result) {
        0 { "the launcher started and ended; that says nothing about how the night went" }
        267009 { 'it is running now' }
        267011 { 'it has never run' }
        267014 { 'it was ended before it finished (by hand, or the 8 hour limit)' }
        default { "the launcher's code, not the night's" }
    }
    Write-Host "Task:        $TaskName"
    Write-Host "State:       $($task.State)"
    Write-Host "Last run:    $($info.LastRunTime)"
    Write-Host "Last result: $result ($meaning)"
    Write-Host "Next run:    $($info.NextRunTime)"
    Write-Host "Runs:        $($task.Actions[0].Execute) $($task.Actions[0].Arguments)"
    Write-Host "In:          $($task.Actions[0].WorkingDirectory)"
    Write-Host "Wakes to run: $($task.Settings.WakeToRun)"
    Show-WakeTimers
    # the runs folder as the Python side finds it: RUNS_DIR, else runs\ in this folder. This script never reads
    # .env, so a RUNS_DIR set only there isn't seen here.
    if ($env:RUNS_DIR) { $runs = $env:RUNS_DIR; $where = 'RUNS_DIR' }
    else { $runs = Join-Path $Folder 'runs'; $where = 'the default; a RUNS_DIR set only in .env is not read here' }
    # the job's own log with -Job, else the newest night.log of any job
    if ($Job) { $log = Join-Path $runs "$Job\night.log" }
    else {
        $found = Get-ChildItem -Path $runs -Filter 'night.log' -Recurse -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        $log = if ($found) { $found.FullName } else { Join-Path $runs '<job>\night.log' }
    }
    Write-Host ''
    Write-Host "The night's own log: $log ($where)"
    if (-not (Test-Path -LiteralPath $log)) {
        Write-Host '  not there yet: the night has never run outside a dry run'
        return
    }
    $lines = @(Get-Content -LiteralPath $log -Encoding UTF8)
    $from = -1
    for ($i = $lines.Count - 1; $i -ge 0; $i--) {
        if ($lines[$i] -match 'the night run of \S+ starts') { $from = $i; break }
    }
    if ($from -lt 0) {
        Write-Host '  no "the night run of ... starts" line in it'
        return
    }
    $last = $lines[$from..($lines.Count - 1)]
    $most = 80
    if ($last.Count -gt $most) {
        Write-Host "  $($lines[$from])"
        Write-Host "  ... $($last.Count - $most - 1) lines left out (the pass's own output); the last $most follow"
        $last = $last[($last.Count - $most)..($last.Count - 1)]
    }
    $last | ForEach-Object { Write-Host "  $_" }
    return
}

if (-not $Python) { throw "No depangram Python found: pass -Python, set DEPANGRAM_PYTHON, or make a venv in $Folder" }
if (-not (Test-Path -LiteralPath $Python)) { throw "The depangram Python isn't there: $Python" }
if (-not (Test-Path -LiteralPath (Join-Path $Folder 'engine\night.py'))) {
    throw "This script has to sit in the depangram folder (engine\night.py next to it): $Folder"
}
if (-not (Test-Path -LiteralPath $Conhost)) { throw "conhost.exe isn't there: $Conhost" }

$action = New-ScheduledTaskAction -Execute $Conhost -Argument "--headless `"$Python`" $NightArgs" `
    -WorkingDirectory $Folder
$trigger = New-ScheduledTaskTrigger -Daily -At $At
$settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 8) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Limited
$description = 'depangram: the night run (main first, the paragraph pass, the finals, the morning submission). ' +
    'Registered by night task.ps1; -Remove unregisters it.'

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Description $description -Force | Out-Null

Write-Host "Registered '$TaskName': daily at $At as $User, only while you are logged on."
Write-Host "  $Conhost --headless `"$Python`" $NightArgs"
Write-Host "  in $Folder"
Write-Host '  Wakes the computer to run, starts late when the start time was missed, no second copy, stopped after 8 hours.'
Show-WakeTimers
Write-Host "Check it any time with: & `"$PSCommandPath`" -Show"
