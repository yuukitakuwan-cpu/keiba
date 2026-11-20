# 毎週の自動処理をタスクスケジューラに登録します。
# 使い方：PowerShell でこのフォルダを開き、次を実行
#   powershell -ExecutionPolicy Bypass -File .\register_tasks.ps1
# 削除するとき：
#   Unregister-ScheduledTask -TaskName "keiba-*" -Confirm:$false

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun `
    -ExecutionTimeLimit (New-TimeSpan -Hours 6) -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries

# 土曜・日曜 8:40 に、その日の全レースを予想
$t1 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday,Sunday -At 8:40
$a1 = New-ScheduledTaskAction -Execute "$here\run_auto_predict.bat"
Register-ScheduledTask -TaskName "keiba-auto-predict" -Trigger $t1 -Action $a1 -Settings $settings `
    -Description "競馬：土日の全レースを自動予想" -Force

# 日曜 21:10 に、答え合わせと再学習
$t2 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 21:10
$a2 = New-ScheduledTaskAction -Execute "$here\run_weekly.bat"
Register-ScheduledTask -TaskName "keiba-weekly" -Trigger $t2 -Action $a2 -Settings $settings `
    -Description "競馬：結果の取り込み・答え合わせ・再学習" -Force

Write-Host "登録しました。タスクスケジューラの『タスク スケジューラ ライブラリ』で keiba-* を確認できます。"
