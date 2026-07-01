# Process and export Persyst Seizure/Spike Detections using Persyst CLI (PSCLI)
# Wrapped in a dynamic scheduler that continuously discovers new RESP folders and manages parallel processing.
#
# Author: Sjors Verschuren
# Date: November 2025

# =========================
# Configuration
# =========================

$baseInputPath = "\\ds\data\HER\her_knf_golf\Wetenschap\newtransport\Sjors\data\preprocessing\eeg\persyst"
$baseOutputPath = "\\ds\data\HER\her_knf_golf\Wetenschap\newtransport\Sjors\data\preprocessing\eeg\persyst\trend_exports"
$logFile = Join-Path -Path $baseInputPath -ChildPath "PSCLI_process_and_export_log.txt"

# Path to PSCLI
$pscliPath = "C:\Program Files (x86)\Persyst\Insight\PSCLI.exe"

# Maximum concurrent jobs --> 12 logical cores, so 10 for safety
$maxConcurrentJobs = 10

# Scheduler settings
$pollSeconds = 10

# Require EDF file to be at least this old before submitting it.
# This reduces the risk of processing a file that is still being copied.
$minFileAgeSeconds = 60

# Exit after this many consecutive idle polls with:
#   - no running jobs
#   - no newly discovered unprocessed folders
#
# With defaults: 6 x 10 sec = 60 sec idle grace period.
$maxIdlePolls = 6

# Export definition
$exportPanel = "SeizureDetails"
$outputPrefix = "event_export"

# =========================
# Setup
# =========================

if (-not (Test-Path $baseInputPath)) {
    throw "Base input path does not exist: $baseInputPath"
}

if (-not (Test-Path $pscliPath)) {
    throw "PSCLI executable not found: $pscliPath"
}

# Clear previous log file
if (Test-Path $logFile) {
    Remove-Item $logFile -Force
}

# Create output dir if not present
if (-not (Test-Path $baseOutputPath)) {
    New-Item -Path $baseOutputPath -ItemType Directory | Out-Null
}

# =========================
# Helper functions
# =========================

function Write-LogOutput {
    param(
        [string]$RespID,
        [string]$Message
    )

    $line = "[${RespID}] $Message"

    # Write to console
    Write-Output $line
    [System.Console]::Out.Flush()

    # Write to log file with retry for concurrent access
    $maxRetries = 5
    $retryCount = 0

    while ($retryCount -lt $maxRetries) {
        try {
            Add-Content -Path $logFile -Value $line
            break
        } catch {
            $retryCount++
            Write-Output "[${RespID}] Log write failed, retrying... ($retryCount/$maxRetries)"
            if ($retryCount -lt $maxRetries) {
                Start-Sleep -Milliseconds 100
            }
        }
    }
}

function Test-FileReady {
    param(
        [string]$Path,
        [int]$MinAgeSeconds
    )

    if (-not (Test-Path $Path)) {
        return $false
    }

    try {
        $item = Get-Item $Path -ErrorAction Stop

        if ($item.Length -le 0) {
            return $false
        }

        $ageSeconds = ((Get-Date) - $item.LastWriteTime).TotalSeconds
        if ($ageSeconds -lt $MinAgeSeconds) {
            return $false
        }

        # Try opening the file for reading.
        # On Windows, this catches many still-copying / locked-file situations.
        $stream = [System.IO.File]::Open(
            $Path,
            [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read,
            [System.IO.FileShare]::ReadWrite
        )
        $stream.Close()

        return $true
    } catch {
        return $false
    }
}

function Get-UnprocessedRespFolders {
    param(
        [string]$BaseInputPath,
        [string]$BaseOutputPath,
        [System.Collections.Generic.HashSet[string]]$SubmittedRespIDs,
        [int]$MinFileAgeSeconds,
        [string]$OutputPrefix
    )

    Get-ChildItem -Path $BaseInputPath -Directory |
        Where-Object { $_.Name -match "^RESP\d{4}$" } |
        ForEach-Object {
            $respID = $_.Name
            $edfPath = Join-Path $_.FullName "$respID.edf"
            $outputFile = Join-Path $BaseOutputPath "${OutputPrefix}_${respID}.csv"

            # Skip if already submitted in this run
            if ($SubmittedRespIDs.Contains($respID)) {
                return
            }

            # Skip if output already exists
            if (Test-Path $outputFile) {
                return
            }

            # Skip if no ready EDF yet. This allows folders to appear before
            # their EDF copy has completed; they will be picked up on a later poll.
            if (-not (Test-FileReady -Path $edfPath -MinAgeSeconds $MinFileAgeSeconds)) {
                return
            }

            $edfItem = Get-Item $edfPath

            [PSCustomObject]@{
                Folder       = $_
                RespID       = $respID
                EdfPath      = $edfPath
                OutputFile   = $outputFile
                EdfSizeBytes = $edfItem.Length
                EdfSizeGB    = [math]::Round($edfItem.Length / 1GB, 2)
            }
        } |
        Sort-Object EdfSizeBytes -Descending
}

# =========================
# Start logging
# =========================

Write-LogOutput "SYSTEM" "========= Start Persyst batch processing ============"
Write-LogOutput "SYSTEM" "Input path: $baseInputPath"
Write-LogOutput "SYSTEM" "Output path: $baseOutputPath"
Write-LogOutput "SYSTEM" "PSCLI path: $pscliPath"
Write-LogOutput "SYSTEM" "Running max $maxConcurrentJobs jobs in parallel"
Write-LogOutput "SYSTEM" "Folder list will be refreshed dynamically while running"
Write-LogOutput "SYSTEM" "Polling every $pollSeconds seconds"
Write-LogOutput "SYSTEM" "Minimum EDF file age before submission: $minFileAgeSeconds seconds"
Write-LogOutput "SYSTEM" "Idle exit after $maxIdlePolls idle polls"
Write-LogOutput "SYSTEM" "====================================================="

# =========================
# Dynamic job scheduler
# =========================

$submittedRespIDs = [System.Collections.Generic.HashSet[string]]::new()
$jobs = @()
$idlePolls = 0

while ($true) {

    # Collect completed jobs, print their output, then remove them.
    $completedJobs = $jobs | Where-Object { $_.State -in @("Completed", "Failed", "Stopped") }

    foreach ($job in $completedJobs) {
        try {
            Receive-Job $job
        } catch {
            Write-LogOutput "SYSTEM" "Failed to receive job output: $_"
        }

        try {
            Remove-Job $job -Force
        } catch {
            Write-LogOutput "SYSTEM" "Failed to remove completed job: $_"
        }
    }

    # Keep only currently running jobs in the active list.
    $jobs = @($jobs | Where-Object { $_.State -in @("NotStarted", "Running") })

    # Fill available slots. Before each submission, refresh the folder list.
    while (($jobs | Where-Object { $_.State -in @("NotStarted", "Running") }).Count -lt $maxConcurrentJobs) {

        $candidate = Get-UnprocessedRespFolders `
            -BaseInputPath $baseInputPath `
            -BaseOutputPath $baseOutputPath `
            -SubmittedRespIDs $submittedRespIDs `
            -MinFileAgeSeconds $minFileAgeSeconds `
            -OutputPrefix $outputPrefix |
            Select-Object -First 1

        if ($null -eq $candidate) {
            break
        }

        $folder = $candidate.Folder
        $respID = $candidate.RespID

        [void]$submittedRespIDs.Add($respID)

        Write-LogOutput "SYSTEM" ("Submitting job: {0} | EDF size: {1} GB" -f $respID, $candidate.EdfSizeGB)

        $job = Start-ThreadJob -ThrottleLimit $maxConcurrentJobs -ScriptBlock {
            param(
                [string]$FolderPath,
                [string]$OutputPath,
                [string]$PscliPath,
                [string]$LogPath,
                [string]$ExportPanel,
                [string]$OutputPrefix
            )

            function Write-LogMessage {
                param(
                    [string]$RespID,
                    [string]$Message
                )

                $line = "[${RespID}] $Message"

                Write-Output $line
                [System.Console]::Out.Flush()

                $maxRetries = 5
                $retryCount = 0

                while ($retryCount -lt $maxRetries) {
                    try {
                        Add-Content -Path $LogPath -Value $line
                        break
                    } catch {
                        $retryCount++
                        Write-Output "[${RespID}] Log write failed, retrying... ($retryCount/$maxRetries)"
                        if ($retryCount -lt $maxRetries) {
                            Start-Sleep -Milliseconds 100
                        }
                    }
                }
            }

            function Invoke-PSCLIProcess {
                param(
                    [string]$FolderPath,
                    [string]$PscliPath
                )

                $respID = Split-Path $FolderPath -Leaf
                $inputFile = Join-Path -Path $FolderPath -ChildPath "${respID}.edf"

                if (-not (Test-Path $inputFile)) {
                    Write-LogMessage $respID "No .edf file found for $respID, skipping processing..."
                    return
                }

                Write-LogMessage $respID "Processing $respID..."

                try {
                    # Safer than Invoke-Expression: call executable directly with arguments.
                    $output = & $PscliPath `
                        "/SourceFile=$inputFile" `
                        "/FileType=EDF90" `
                        "/Process" 2>&1

                    foreach ($line in $output) {
                        Write-LogMessage $respID $line
                    }

                    Write-LogMessage $respID "Finished processing $respID."
                } catch {
                    Write-LogMessage $respID "Error processing ${respID}: $_"
                }
            }

            function Export-PSCLI {
                param(
                    [string]$FolderPath,
                    [string]$OutputPath,
                    [string]$PscliPath,
                    [string]$ExportPanel,
                    [string]$OutputPrefix
                )

                $respID = Split-Path $FolderPath -Leaf
                $inputFile = Join-Path -Path $FolderPath -ChildPath "${respID}.lay"
                $outputFile = Join-Path -Path $OutputPath -ChildPath "${OutputPrefix}_${respID}.csv"

                if (-not (Test-Path $inputFile)) {
                    Write-LogMessage $respID "No .lay file found for $respID, skipping export..."
                    return
                }

                Write-LogMessage $respID "Exporting $respID to $outputFile..."

                try {
                    # Safer than Invoke-Expression: call executable directly with arguments.
                    $output = & $PscliPath `
                        "/SourceFile=$inputFile" `
                        "/ExportCSV" `
                        "/OutputFile=$outputFile" `
                        "/Panel=$ExportPanel" 2>&1

                    foreach ($line in $output) {
                        Write-LogMessage $respID $line
                    }

                    if (Test-Path $outputFile) {
                        Write-LogMessage $respID "Finished export $respID."
                    } else {
                        Write-LogMessage $respID "Export command completed but output CSV was not found: $outputFile"
                    }
                } catch {
                    Write-LogMessage $respID "Error exporting ${respID}: $_"
                }
            }

            $respID = Split-Path $FolderPath -Leaf

            Write-LogMessage $respID "---------- START ----------"
            Invoke-PSCLIProcess -FolderPath $FolderPath -PscliPath $PscliPath
            Export-PSCLI `
                -FolderPath $FolderPath `
                -OutputPath $OutputPath `
                -PscliPath $PscliPath `
                -ExportPanel $ExportPanel `
                -OutputPrefix $OutputPrefix
            Write-LogMessage $respID "---------- DONE -----------"

        } -ArgumentList `
            $folder.FullName, `
            $baseOutputPath, `
            $pscliPath, `
            $logFile, `
            $exportPanel, `
            $outputPrefix

        $jobs += $job
    }

    $activeCount = ($jobs | Where-Object { $_.State -in @("NotStarted", "Running") }).Count

    # Check whether any unprocessed ready candidates currently remain.
    $remainingCandidateCount = (
        Get-UnprocessedRespFolders `
            -BaseInputPath $baseInputPath `
            -BaseOutputPath $baseOutputPath `
            -SubmittedRespIDs $submittedRespIDs `
            -MinFileAgeSeconds $minFileAgeSeconds `
            -OutputPrefix $outputPrefix |
        Measure-Object
    ).Count

    if ($activeCount -eq 0 -and $remainingCandidateCount -eq 0) {
        $idlePolls++

        Write-LogOutput "SYSTEM" (
            "Idle poll {0}/{1}: no running jobs and no ready unprocessed folders" -f `
            $idlePolls, $maxIdlePolls
        )

        if ($idlePolls -ge $maxIdlePolls) {
            break
        }
    } else {
        $idlePolls = 0
    }

    Start-Sleep -Seconds $pollSeconds
}

# One final receive/remove pass, just in case anything completed between loop condition and exit.
$completedJobs = $jobs | Where-Object { $_.State -in @("Completed", "Failed", "Stopped") }
foreach ($job in $completedJobs) {
    try {
        Receive-Job $job
    } catch {
        Write-LogOutput "SYSTEM" "Failed to receive final job output: $_"
    }

    try {
        Remove-Job $job -Force
    } catch {
        Write-LogOutput "SYSTEM" "Failed to remove final job: $_"
    }
}

Write-LogOutput "SYSTEM" "All processing and exports completed. Log saved to $logFile"
