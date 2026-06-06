#!/usr/bin/env pwsh
#Requires -Version 5.1
<#
.SYNOPSIS
    casr installer — Cross Agent Session Resumer (Windows)

.DESCRIPTION
    Downloads and installs the casr binary for Windows.
    Supports --version, --dest, --easy-mode, --verify, --from-source, --uninstall.

.EXAMPLE
    irm https://raw.githubusercontent.com/quangdang46/cross_agent_session_resumer/main/install.ps1 | iex

.EXAMPLE
    .\install.ps1 -Version v0.2.2 -Dest C:\Tools
#>

[CmdletBinding()]
param(
    [string]$Version = "",
    [string]$Dest = "",
    [switch]$EasyMode,
    [switch]$Verify,
    [switch]$FromSource,
    [switch]$Quiet,
    [switch]$Uninstall,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# ── Config ──────────────────────────────────────────────────────────────────────
$BINARY_NAME = "casr.exe"
$OWNER = "quangdang46"
$REPO = "cross_agent_session_resumer"
$MAX_RETRIES = 3
$DOWNLOAD_TIMEOUT = 120

if (-not $Dest) {
    $Dest = Join-Path $env:USERPROFILE ".local\bin"
}

# ── Logging ─────────────────────────────────────────────────────────────────────
function Log-Info    { if (-not $Quiet) { Write-Host "[casr] $args" } }
function Log-Warn    { Write-Host "[casr] WARN: $args" -ForegroundColor Yellow }
function Log-Success { if (-not $Quiet) { Write-Host "✓ $args" -ForegroundColor Green } }
function Die         { Write-Host "ERROR: $args" -ForegroundColor Red; exit 1 }

# ── Help ────────────────────────────────────────────────────────────────────────
function Show-Help {
    @"

casr installer — Cross Agent Session Resumer (Windows)

Usage:
  irm https://raw.githubusercontent.com/$OWNER/$REPO/main/install.ps1 | iex
  .\install.ps1 [OPTIONS]

Options:
  -Version vX.Y.Z   Install specific version (default: latest)
  -Dest DIR          Install to DIR (default: ~/.local/bin)
  -EasyMode          Auto-update PATH in user environment
  -Verify            Run self-test after install
  -FromSource        Build from source instead of downloading binary
  -Quiet             Suppress non-error output
  -Uninstall         Remove casr and clean up PATH
  -Help              Show this help

"@
    exit 0
}

if ($Help) { Show-Help }

# ── Uninstall ───────────────────────────────────────────────────────────────────
function Do-Uninstall {
    $binPath = Join-Path $Dest $BINARY_NAME
    if (Test-Path $binPath) {
        Remove-Item $binPath -Force
        Log-Success "Removed $binPath"
    }

    # Remove PATH entry
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($userPath -like "*$Dest*") {
        $newPath = ($userPath -split ";" | Where-Object { $_ -ne $Dest }) -join ";"
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
        Log-Success "Removed $Dest from PATH"
    }

    Log-Success "casr uninstalled"
    exit 0
}

if ($Uninstall) { Do-Uninstall }

# ── Platform ────────────────────────────────────────────────────────────────────
function Get-Platform {
    $arch = $env:PROCESSOR_ARCHITECTURE
    switch ($arch) {
        "AMD64"  { return "x86_64" }
        "ARM64"  { return "aarch64" }
        default  { Die "Unsupported architecture: $arch" }
    }
}

# ── Version resolution ──────────────────────────────────────────────────────────
function Resolve-Version {
    if ($Version) { return $Version }

    try {
        $headers = @{ "Accept" = "application/vnd.github.v3+json" }
        $release = Invoke-RestMethod -Uri "https://api.github.com/repos/$OWNER/$REPO/releases/latest" `
            -Headers $headers -TimeoutSec 30
        $v = $release.tag_name
        if ($v -match "^v\d") { return $v }
    } catch {
        Log-Warn "GitHub API failed, trying redirect..."
    }

    try {
        $resp = Invoke-WebRequest -Uri "https://github.com/$OWNER/$REPO/releases/latest" `
            -MaximumRedirection 0 -ErrorAction SilentlyContinue
    } catch {
        if ($_.Exception.Response.StatusCode -eq 302) {
            $location = $_.Exception.Response.Headers.Location.ToString()
            if ($location -match "/tag/(v[\d.]+)") { return $Matches[1] }
        }
    }

    Die "Could not resolve version. Use -Version vX.Y.Z"
}

# ── Download ────────────────────────────────────────────────────────────────────
function Download-File {
    param([string]$Url, [string]$DestPath)

    $partial = "${DestPath}.part"
    for ($attempt = 1; $attempt -le $MAX_RETRIES; $attempt++) {
        try {
            $wc = New-Object System.Net.WebClient
            $wc.DownloadFile($Url, $partial)
            Move-Item $partial $DestPath -Force
            return $true
        } catch {
            if ($attempt -lt $MAX_RETRIES) {
                Log-Warn "Retry $attempt/$MAX_RETRIES..."
                Start-Sleep -Seconds 3
            }
        }
    }
    return $false
}

# ── Build from source ───────────────────────────────────────────────────────────
function Build-FromSource {
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        Die "Rust/cargo not found. Install: https://rustup.rs"
    }

    $tmpDir = Join-Path $env:TEMP "casr-build-$(Get-Random)"
    New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null

    Log-Info "Cloning repository..."
    git clone --depth 1 "https://github.com/$OWNER/$REPO.git" "$tmpDir\src" 2>&1 | Out-Null

    Log-Info "Building from source (this may take a few minutes)..."
    $env:CARGO_TARGET_DIR = "$tmpDir\target"
    Push-Location "$tmpDir\src"
    cargo build --release 2>&1 | Out-Null
    Pop-Location

    $binSrc = "$tmpDir\target\release\$BINARY_NAME"
    if (-not (Test-Path $binSrc)) { Die "Build failed: $binSrc not found" }

    New-Item -ItemType Directory -Path $Dest -Force | Out-Null
    $binDst = Join-Path $Dest $BINARY_NAME
    Copy-Item $binSrc $binDst -Force

    Remove-Item $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
    Log-Success "Built from source"
}

# ── Main ────────────────────────────────────────────────────────────────────────
function Main {
    Log-Info "casr installer for Windows"

    $platform = Get-Platform
    Log-Info "Platform: windows-$platform | Dest: $Dest"

    New-Item -ItemType Directory -Path $Dest -Force | Out-Null

    if ($FromSource) {
        Build-FromSource
    } else {
        $ver = Resolve-Version
        Log-Info "Version: $ver"

        $archive = "casr-${ver}-windows-${platform}.zip"
        $url = "https://github.com/$OWNER/$REPO/releases/download/$ver/$archive"
        $tmpDir = Join-Path $env:TEMP "casr-install-$(Get-Random)"
        New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null

        $archivePath = Join-Path $tmpDir $archive
        Log-Info "Downloading $url..."

        if (Download-File $url $archivePath) {
            # Verify checksum
            $shaUrl = "${url}.sha256"
            $shaPath = Join-Path $tmpDir "checksum.sha256"
            if (Download-File $shaUrl $shaPath) {
                $expected = (Get-Content $shaPath -Raw).Split(" ")[0].Trim()
                $actual = (Get-FileHash -Algorithm SHA256 $archivePath).Hash.ToLower()
                if ($expected -ne $actual) { Die "Checksum mismatch" }
                Log-Info "Checksum verified"
            }

            # Extract
            $extractDir = Join-Path $tmpDir "extract"
            Expand-Archive -Path $archivePath -DestinationPath $extractDir -Force
            $binSrc = Get-ChildItem -Path $extractDir -Recurse -Filter $BINARY_NAME | Select-Object -First 1
            if (-not $binSrc) { Die "Binary not found after extraction" }

            $binDst = Join-Path $Dest $BINARY_NAME
            Copy-Item $binSrc.FullName $binDst -Force
            Log-Success "Installed $binDst"
        } else {
            Log-Warn "Binary download failed — building from source..."
            Build-FromSource
        }

        Remove-Item $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
    }

    # PATH update
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($userPath -notlike "*$Dest*") {
        if ($EasyMode) {
            $newPath = "$Dest;$userPath"
            [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
            $env:Path = "$Dest;$env:Path"
            Log-Success "Added $Dest to PATH (restart terminal to persist)"
        } else {
            Log-Warn "Add to PATH: `$env:Path = `"$Dest;`$env:Path`""
        }
    }

    # Verify
    if ($Verify) {
        $binPath = Join-Path $Dest $BINARY_NAME
        $verOutput = & $binPath --version 2>&1
        Log-Info "Verify: $verOutput"
    }

    Write-Host ""
    Write-Host "✓ casr installed → $Dest\$BINARY_NAME" -ForegroundColor Green
    $binPath = Join-Path $Dest $BINARY_NAME
    $verStr = & $binPath --version 2>&1
    Write-Host "  $verStr"
    Write-Host ""
    Write-Host "  Usage: casr --help"
    Write-Host ""
}

Main
