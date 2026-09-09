@echo off
setlocal

REM ------------------------------------------------------------
REM Copy selected v24x core output files for Soft Spaces article
REM Destination: Soft_Spaces_article_work\02_v24x_output_files
REM ------------------------------------------------------------

set "DEST=Soft_Spaces_article_work\02_v24x_output_files"

if not exist "%DEST%" (
    mkdir "%DEST%"
)

copy /Y "v24_3_q6_interference.txt" "%DEST%\"
copy /Y "v24_3_run_7qubits_3x5000.txt" "%DEST%\"

copy /Y "v24_5_6qubits_structure_dynamics_boosted.txt" "%DEST%\"
copy /Y "v24_5_7qubits_structure_dynamics_boosted.txt" "%DEST%\"

copy /Y "v24_6_8qubits_structure_dynamics_probe.txt" "%DEST%\"
copy /Y "v24_7_8qubits_contrast_focus_probe.txt" "%DEST%\"

copy /Y "v24_8_merge_8qubits_kernel_zoom.txt" "%DEST%\"
copy /Y "v24_8_1_8qubits_kernel_zoom_evidence.txt" "%DEST%\"

copy /Y "v24_9_2_8qubits_repeat1.txt" "%DEST%\"
copy /Y "v24_9_2_8qubits_repeat2.txt" "%DEST%\"
copy /Y "v24_9_2_8qubits_repeat3.txt" "%DEST%\"
copy /Y "v24_9_3_8qubits_repeat_merge.txt" "%DEST%\"
copy /Y "v24_9_4_8qubits_priority_report.txt" "%DEST%\"

copy /Y "v24_12_REALonly_repeat_merge.txt" "%DEST%\"
copy /Y "v24_13_repeat_merge.txt" "%DEST%\"
copy /Y "v24_14_repeat_merge.txt" "%DEST%\"

copy /Y "v24_17_v24_14_local_ridge_report.txt" "%DEST%\"
copy /Y "v24_18_explanatory_ridge_report.txt" "%DEST%\"
copy /Y "v24_20_article_ready_mini_section.txt" "%DEST%\"
copy /Y "v24_21_local_lindblad_geometry_memo.txt" "%DEST%\"

echo.
echo Done. Selected files copied to "%DEST%".
echo.

dir "%DEST%"

pause
endlocal