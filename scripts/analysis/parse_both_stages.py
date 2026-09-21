import re
import sys

# Usage: python parse_both_stages.py <log_file> <job_name> [output_file]
if len(sys.argv) < 3:
    print("Usage: python parse_both_stages.py <log_file> <job_name> [output_file]")
    sys.exit(1)

log_file = sys.argv[1]
job_name = sys.argv[2]
output_file = sys.argv[3] if len(sys.argv) > 3 else None

try:
    with open(log_file, 'r') as f:
        content = f.read()
except FileNotFoundError:
    print(f"Error: Log file not found at {log_file}")
    sys.exit(1)

output_lines = []
output_lines.append("\n" + "=" * 80)
output_lines.append(f"JOB RESULTS: {job_name}")
output_lines.append("=" * 80)

# ========== STAGE 1 (Link Prediction) ==========
stage1_match = re.search(r'\[Stage 1.*?\] class_scale=1\.0.*?Reloading best Stage 1 checkpoint', content, re.DOTALL)
if stage1_match:
    stage1_text = stage1_match.group(0)
    
    # Extract Stage 1 progress lines (AUC metrics)
    s1_progress = re.findall(r'(\d+)%\|[^|]+\|\s*(\d+)/500.*?best_val=([\d.]+).*?loss=([\d.]+).*?test_auc=([\d.]+).*?train_auc=([\d.]+).*?val_auc=([\d.]+)', stage1_text)
    
    if s1_progress:
        output_lines.append("\nStage 1 (Link Prediction) - Per-Epoch Metrics")
        output_lines.append("-" * 80)
        output_lines.append(f"{'Epoch':>6} {'Progress':>8} {'Train Loss':>12} {'Train AUC':>10} {'Val AUC':>10} {'Test AUC':>10} {'Best Val':>10}")
        output_lines.append("-" * 80)

        
        for percent, epoch, best_val, loss, test_auc, train_auc, val_auc in s1_progress:
            output_lines.append(f"{epoch:>6} {percent:>7}% {loss:>12} {train_auc:>10} {val_auc:>10} {test_auc:>10} {best_val:>10}")

# ========== STAGE 2 (Edge Weight Regression) ==========
stage2_match = re.search(r'\[Stage 2.*?\] class_scale=0\.0.*?Done\.', content, re.DOTALL)
if stage2_match:
    stage2_text = stage2_match.group(0)
    
    # Extract Stage 2 progress lines (MAE metrics)
    s2_progress = re.findall(r'(\d+)%\|[^|]+\|\s*(\d+)/500.*?best_val=([\d.]+).*?loss=([\d.]+).*?test_mae=([\d.]+).*?val_mae=([\d.]+)', stage2_text)
    
    if s2_progress:
        output_lines.append("\n\nStage 2 (Edge Weight Regression) - Per-Epoch Metrics")
        output_lines.append("-" * 80)
        output_lines.append(f"{'Epoch':>6} {'Progress':>8} {'Train Loss':>12} {'Val MAE':>10} {'Test MAE':>10} {'Best Val':>10}")
        output_lines.append("-" * 80)
        
        for percent, epoch, best_val, loss, test_mae, val_mae in s2_progress:
            output_lines.append(f"{epoch:>6} {percent:>7}% {loss:>12} {val_mae:>10} {test_mae:>10} {best_val:>10}")
    
    # Extract LR reductions
    output_lines.append("\nLearning Rate Reductions (Stage 2):")
    output_lines.append("-" * 80)
    lr_reductions = re.findall(r'Epoch (\d+): reducing learning rate.*?to ([\d.e-]+)', stage2_text)
    for epoch, new_lr in lr_reductions:
        output_lines.append(f"Epoch {epoch:>3}: LR reduced to {new_lr}")
    
    # Extract final results
    final_match = re.search(r"stage2: \{([^}]+)\}", content)
    if final_match:
        output_lines.append("\nFinal Metrics (Stage 2):")
        output_lines.append("-" * 80)
        output_lines.append(final_match.group(1))

# Write to file or stdout
final_output = "\n".join(output_lines)
if output_file:
    with open(output_file, 'a') as f:
        f.write(final_output + "\n")
    print(f"Results appended to {output_file}")
else:
    print(final_output)
