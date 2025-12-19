# <gcl_re.ps1>
#
# Runs the sequence of helper scripts used to setup and deploy Open-Sora Serving on GCP.
# This wrapper can dry-run, resume from a particular step, and control error behaviour.
#
# Usage examples:
#   .\gcl_re.ps1 -DryRun
#   .\gcl_re.ps1 -StartStep 2 -EndStep 5
#   .\gcl_re.ps1 -ContinueOnError

[CmdletBinding()]
param(
	[switch]$DryRun,
	[switch]$ContinueOnError,
	[int]$StartStep = 1,
	[int]$EndStep = 0
)

function Write-Log {
	param(
		[string]$Message,
		[ValidateSet('Info','Success','Warning','Error')]
		[string]$Level = 'Info'
	)
	switch ($Level) {
		'Success' { Write-Host "[✔] $Message" -ForegroundColor Green }
		'Warning' { Write-Host "[!] $Message" -ForegroundColor Yellow }
		'Error'   { Write-Host "[X] $Message" -ForegroundColor Red }
		default   { Write-Host "[-] $Message" -ForegroundColor Cyan }
	}
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$steps = @(
	@{ File = 'myconfig.ps1'; Optional = $false },
	@{ File = 'build.ps1'; Optional = $false },
	@{ File = 'upload.ps1'; Optional = $false },
	@{ File = 'model_id.ps1'; Optional = $false },
	@{ File = 'endpoint.ps1'; Optional = $true  }, # optional; uncommented depending on environment
	@{ File = 'deploy.ps1'; Optional = $false },
	@{ File = 'auth.ps1'; Optional = $false },
	@{ File = 'reqt.ps1'; Optional = $false },
	@{ File = 'api.ps1'; Optional = $false }
)

$TotalSteps = $steps.Count
if ($EndStep -eq 0) { $EndStep = $TotalSteps }

if ($StartStep -lt 1 -or $StartStep -gt $TotalSteps) {
	Write-Log "Invalid StartStep ($StartStep). Must be 1..$TotalSteps" 'Error'
	exit 2
}
if ($EndStep -lt $StartStep -or $EndStep -gt $TotalSteps) {
	Write-Log "Invalid EndStep ($EndStep). Must be StartStep..$TotalSteps" 'Error'
	exit 2
}

if (-not $ContinueOnError) { $ErrorActionPreference = 'Stop' }

Write-Log "Running steps $StartStep through $EndStep (total: $TotalSteps). DryRun: $DryRun. ContinueOnError: $ContinueOnError" 'Info'

for ($i = $StartStep; $i -le $EndStep; $i++) {
	$s = $steps[$i - 1]
	$filePath = Join-Path -Path $ScriptDir -ChildPath $s.File
	$stepName = $s.File
	if (-not (Test-Path $filePath)) {
		if ($s.Optional) {
				    Write-Log "Step $($i): $stepName missing but optional - skipping" 'Warning'
			continue
		}
		else {
				    Write-Log "Step $($i): $stepName not found at path: $filePath" 'Error'
			if (-not $ContinueOnError) { exit 3 }
			continue
		}
	}

	Write-Log "Step $($i)/$($TotalSteps): $stepName" 'Info'
	if ($DryRun) {
		Write-Log "DryRun: Would execute: $filePath" 'Info'
		continue
	}

	try {
		# dot-source the config script so it can set environment variables and export functions
		Write-Host "Executing $filePath ..." -ForegroundColor DarkCyan
		. $filePath
		Write-Log "Step $($i): $stepName completed" 'Success'
	}
	catch {
		Write-Log "Error running $($stepName): $($_.Exception.Message)" 'Error'
		if (-not $ContinueOnError) { exit 4 }
		Write-Log "Continuing to next step due to -ContinueOnError" 'Warning'
	}
}

Write-Log "Sequence finished (steps $StartStep-$EndStep)." 'Success'

exit 0

# End of gcl_re.ps1