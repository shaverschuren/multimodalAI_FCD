# Process and export Persyst Seizure/Spike Detections using Persyst CLI (PSCLI)
# Use some parallel processing to speed things up. Running 5 threads on this machine.
#
# Author: Sjors Verschuren
# Date: November 2025

# Define the base paths
$baseInputPath = "C:\Users\sversch6\Documents\persyst"
$baseOutputPath = "C:\Users\sversch6\Documents\persyst\detection_exports"
$logFile = Join-Path -Path $baseInputPath -ChildPath "PSCLI_process_and_export_log.txt"

# Clear previous log file
if (Test-Path $logFile) { Remove-Item $logFile }

# Create output dir if not present
if (-not (Test-Path $baseOutputPath)) {
    New-Item -Path $baseOutputPath -ItemType Directory | Out-Null
}

# Path to PSCLI
$pscliPath = "C:\'Program Files (x86)'\Persyst\Insight\PSCLI.exe"

# Get all RESP folders
$respFolders = Get-ChildItem -Path $baseInputPath -Directory | Where-Object { ${_}.Name -match "^RESP\d{4}$" }
$nRespFolders = $respFolders.Count

# Maximum concurrent jobs --> 6 cores so 5 for safety
$maxConcurrentJobs = 5
$jobs = @()

# Helper function to log both to console and file
function Write-LogOutput {
    param([string]${respID}, [string]$message)
    $line = "[${respID}] $message"
    
    # Write to console
    Write-Output $line
    [System.Console]::Out.Flush()
    # Write to log file (with retry for concurrent access)
    $maxRetries = 5
    $retryCount = 0
    while ($retryCount -lt $maxRetries) {
        try {
            Add-Content -Path $l -Value $line
            break
        } catch {
            $retryCount++
            Write-Output "[${respID}] Log write failed, retrying... ($retryCount/$maxRetries)"
            if ($retryCount -lt $maxRetries) {
                Start-Sleep -Milliseconds 100
            }
        }
    }
}

# Write startup info
Write-LogOutput "SYSTEM" "========= Start Persyst batch processing ============"
Write-LogOutput "SYSTEM" "Found data for $nRespFolders patients"
Write-LogOutput "SYSTEM" "Running $maxConcurrentJobs jobs in parallel"
Write-LogOutput "SYSTEM" "====================================================="

foreach ($folder in $respFolders) {
    # Wait if we already have maxConcurrentJobs running
    while (($jobs | Where-Object { ${_}.State -eq 'Running' }).Count -ge $maxConcurrentJobs) {
        Start-Sleep -Seconds 1
    }

    # Start the job
    $jobs += Start-ThreadJob -ScriptBlock {
        param($f, $o, $p, $l)

        # Function to log from inside the job
        function Write-LogMessage {
            param([string]${respID}, [string]$message)
            $line = "[${respID}] $message"
            
            # Write to console
            Write-Output $line
            [System.Console]::Out.Flush()
            # Write to log file (with retry for concurrent access)
            $maxRetries = 5
            $retryCount = 0
            while ($retryCount -lt $maxRetries) {
                try {
                    Add-Content -Path $l -Value $line
                    break
                } catch {
                    $retryCount++
                    Write-Output "[${respID}] Log write failed, retrying... ($retryCount/$maxRetries)"
                    if ($retryCount -lt $maxRetries) {
                        Start-Sleep -Milliseconds 100
                    }
                }
            }
        }

        # ---- Process Function ----
        function Invoke-PSCLIProcess {
            param([string]$folderPath, [string]$pscliPath)

            ${respID} = Split-Path $folderPath -Leaf
            $inputFile = Join-Path -Path $folderPath -ChildPath "${respID}.edf"

            if (-not (Test-Path $inputFile)) {
                Write-LogMessage ${respID} "No .edf file found for ${respID}, skipping processing..."
                return
            }

            Write-LogMessage ${respID} "Processing ${respID}..."
            $cmd = "$pscliPath /SourceFile=`"$inputFile`" /FileType=`"EDF90`" /Process"

            try {
                $output = Invoke-Expression $cmd 2>&1
                foreach ($o in $output) {
                    Write-LogMessage ${respID} $o
                }
                Write-LogMessage ${respID} "Finished processing ${respID}.`n"
            } catch {
                Write-LogMessage ${respID} "Error processing ${respID}: ${_}`n"
            }
        }

        # ---- Export Function ----
        function Export-PSCLI {
            param([string]$folderPath, [string]$outputPath, [string]$pscliPath)

            ${respID} = Split-Path $folderPath -Leaf
            $inputFile = Join-Path -Path $folderPath -ChildPath "${respID}.lay"
            $outputFile = Join-Path -Path $outputPath -ChildPath "event_export_${respID}.csv"

            if (-not (Test-Path $inputFile)) {
                Write-LogMessage ${respID} "No .lay file found for ${respID}, skipping export..."
                return
            }

            Write-LogMessage ${respID} "Exporting ${respID}..."
            $cmd = "$pscliPath /SourceFile=`"$inputFile`" /ExportCSV /OutputFile=`"$outputFile`" /Panel=`"SeizureDetails`""

            try {
                $output = Invoke-Expression $cmd 2>&1
                foreach ($o in $output) {
                    Write-LogMessage ${respID} $o
                }
                Write-LogMessage ${respID} "Finished export ${respID}.`n"
            } catch {
                Write-LogMessage ${respID} "Error export ${respID}: ${_}`n"
            }
        }

        # Run both functions
        ${respID} = Split-Path $f -Leaf
        Write-LogMessage ${respID} "---------- START ----------"
        Invoke-PSCLIProcess -folderPath $f -pscliPath $p
        Export-PSCLI -folderPath $f -outputPath $o -pscliPath $p
        Write-LogMessage ${respID} "---------- DONE -----------"

    } -ArgumentList $folder.FullName, $baseOutputPath, $pscliPath, $logFile
}

# Wait for all jobs to complete and receive output
$jobs | ForEach-Object {
    Wait-Job ${_}
    Receive-Job ${_}
    Remove-Job ${_}
}

Write-LogOutput " SYSTEM " "All processing and exports completed. Log saved to $logFile"
