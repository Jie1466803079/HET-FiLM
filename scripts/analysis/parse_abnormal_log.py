import re
import sys

# Usage: python parse_abnormal_log.py <log_file> <output_file>
if len(sys.argv) < 3:
    print("Usage: python parse_abnormal_log.py <log_file> <output_file>")
    sys.exit(1)

log_file = sys.argv[1]
output_file = sys.argv[2]

try:
    with open(log_file, 'r') as f:
        content = f.read()
except FileNotFoundError:
    print(f"Error: Log file not found at {log_file}")
    sys.exit(1)

output_lines = []
output_lines.append("\n" + "=" * 80)
output_lines.append(f"JOB RESULTS: Abnormal Return Prediction")
output_lines.append("=" * 80)

# Extract progress lines
# Format: 14%|...| 14/100 ... loss=0.0972 ... test=0.3099, train=0.1991, val=0.2015
progress_lines = re.findall(r'(\d+)/100.*?loss=([\d.]+).*?test=([\d.]+).*?train=([\d.]+).*?val=([\d.]+)', content)

output_lines.append("Per-Epoch Metrics (MAE)")
output_lines.append("-" * 80)
output_lines.append(f"{'Epoch':>6} {'Train MAE':>12} {'Val MAE':>10} {'Test MAE':>10} {'Loss':>10}")
output_lines.append("-" * 80)

for epoch, loss, test, train, val in progress_lines:
    output_lines.append(f"{epoch:>6} {train:>12} {val:>10} {test:>10} {loss:>10}")

# Extract Comprehensive Evaluation
eval_match = re.search(r"COMPREHENSIVE EVALUATION.*", content, re.DOTALL)
if eval_match:
    output_lines.append("\nFinal Metrics:")
    output_lines.append("-" * 80)
    output_lines.append(eval_match.group(0))

final_output = "\n".join(output_lines)
with open(output_file, 'a') as f:
    f.write(final_output + "\n")
print(f"Results appended to {output_file}")
