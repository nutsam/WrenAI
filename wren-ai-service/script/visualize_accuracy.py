import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import font_manager
import pandas as pd
import os
import glob
import sys
from datetime import datetime

# Set font for better display
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def list_csv_files(directory_path):
    """列出所有可用的 CSV 檔案"""
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    return sorted(csv_files, key=os.path.getctime, reverse=True)

def select_csv_file(directory_path):
    """讓使用者選擇 CSV 檔案"""
    files = list_csv_files(directory_path)
    
    if not files:
        print("❌ 沒有找到任何 CSV 檔案！")
        sys.exit(1)
        
    print("\n📁 可用的 CSV 檔案：")
    for i, file_path in enumerate(files, 1):
        file_time = datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{i:2}. {os.path.basename(file_path)} - {file_time}")
    
    while True:
        try:
            choice = int(input(f"\n請選擇檔案 (1-{len(files)}): "))
            if 1 <= choice <= len(files):
                selected_file = files[choice - 1]
                print(f"✅ 已選擇檔案: {os.path.basename(selected_file)}")
                df = pd.read_csv(selected_file)
                return df, selected_file
            else:
                print(f"❌ 請輸入 1 到 {len(files)} 之間的數字")
        except ValueError:
            print("❌ 請輸入有效的數字")
        except KeyboardInterrupt:
            print("\n👋 程式已取消")
            sys.exit(0)

def analyze_data(df):
    """Analyze the data and create summary statistics by table count"""
    # Group by table_count and calculate statistics
    summary = df.groupby('table_count').agg({
        'score': ['count', 'sum', 'mean'],
        'success': ['sum', 'mean']
    }).round(3)
    
    # Flatten column names
    summary.columns = ['total_cases', 'total_score', 'avg_score', 'success_cases', 'success_rate']
    summary['success_rate'] = summary['success_rate'] * 100  # Convert to percentage
    summary = summary.reset_index()
    
    # Create readable table count labels
    summary['table_count_label'] = summary['table_count'].apply(lambda x: f'{x} Table{"s" if x > 1 else ""}')
    
    return summary

# Load data from the analysis directory
analysis_dir = "/Users/samtsai/WrenAI/wren-ai-service/outputs/analysis"

print("="*80)
print("WrenAI Accuracy Visualization Tool")
print("="*80)
print("📊 這個工具會從分析目錄中的 CSV 檔案視覺化準確率指標")
print("🔍 請從下列檔案中選擇一個進行視覺化分析")

# Use interactive file selection
df_raw, file_path = select_csv_file(analysis_dir)
print(f"📈 成功載入 {len(df_raw)} 筆記錄，來自檔案: {os.path.basename(file_path)}")

# Analyze the data
df = analyze_data(df_raw)

# Set chart style
plt.style.use('default')
fig = plt.figure(figsize=(18, 6))

# Create 1x3 subplot layout
gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)

# 1. Success Rate Bar Chart
ax1 = fig.add_subplot(gs[0, 0])
colors = ['#2E8B57', '#4169E1', '#FF6347', '#32CD32', '#DC143C'][:len(df)]
bars1 = ax1.bar(df['table_count_label'], df['success_rate'], color=colors, alpha=0.8, edgecolor='black', linewidth=1)
ax1.set_title('Success Rate by Table Count', fontsize=14, fontweight='bold', pad=20)
ax1.set_ylabel('Success Rate (%)', fontsize=12)
ax1.set_ylim(0, 110)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars1, df['success_rate']):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

# 2. Total Cases vs Success Cases Comparison
ax2 = fig.add_subplot(gs[0, 1])
x = np.arange(len(df))
width = 0.35

bars2 = ax2.bar(x - width/2, df['total_cases'], width, label='Total Cases', 
                color='lightblue', alpha=0.8, edgecolor='black')
bars3 = ax2.bar(x + width/2, df['success_cases'], width, label='Success Cases', 
                color='lightgreen', alpha=0.8, edgecolor='black')

ax2.set_title('Total Cases vs Success Cases', fontsize=14, fontweight='bold', pad=20)
ax2.set_ylabel('Number of Cases', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(df['table_count_label'])
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                 f'{int(height)}', ha='center', va='bottom', fontweight='bold')

# 3. Average Score Line Chart
ax3 = fig.add_subplot(gs[0, 2])
line = ax3.plot(df['table_count_label'], df['avg_score'], marker='o', linewidth=3, 
                markersize=8, color='purple', markerfacecolor='orange', 
                markeredgecolor='black', markeredgewidth=2)
ax3.set_title('Average Score Trend', fontsize=14, fontweight='bold', pad=20)
ax3.set_ylabel('Average Score', fontsize=12)
ax3.set_ylim(0, 1.1)
ax3.grid(True, alpha=0.3)

# Add value labels on points
for i, value in enumerate(df['avg_score']):
    ax3.text(i, value + 0.05, f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

# Display charts
plt.tight_layout()
plt.show()

# Create detailed data table
print("\n" + "="*80)
print("📊 詳細測試結果資料表")
print("="*80)
display_df = df[['table_count_label', 'total_cases', 'success_cases', 'success_rate', 'avg_score']].copy()
display_df.columns = ['表格數量', '總測試案例', '成功案例', '成功率 (%)', '平均分數']
print(display_df.to_string(index=False))
print("\n")

# Calculate additional statistical metrics
print("📈 額外統計分析:")
print(f"• 分析檔案: {os.path.basename(file_path)}")
print(f"• 總測試案例數: {df_raw.shape[0]}")
print(f"• 整體成功率: {(df_raw['success'].sum() / len(df_raw) * 100):.1f}%")
print(f"• 整體平均分數: {df_raw['score'].mean():.3f}")
print(f"• 最高成功率: {df['success_rate'].max():.1f}% ({df.loc[df['success_rate'].idxmax(), 'table_count_label']})")
print(f"• 最低成功率: {df['success_rate'].min():.1f}% ({df.loc[df['success_rate'].idxmin(), 'table_count_label']})")
print(f"• 成功率標準差: {df['success_rate'].std():.1f}%")
print(f"• 最高平均分數: {df['avg_score'].max():.3f} ({df.loc[df['avg_score'].idxmax(), 'table_count_label']})")
print(f"• 最低平均分數: {df['avg_score'].min():.3f} ({df.loc[df['avg_score'].idxmin(), 'table_count_label']})")

# Show distribution by table count
print(f"\n🔢 表格數量分布:")
for _, row in df.iterrows():
    print(f"• {row['table_count_label']}: {int(row['total_cases'])} 個案例 ({row['success_rate']:.1f}% 成功)")
    
# Show error analysis if errors exist
error_cases = df_raw[df_raw['error'].notna()]
if not error_cases.empty:
    print(f"\n❌ 錯誤分析:")
    print(f"• 有錯誤的案例數: {len(error_cases)}")
    print(f"• 錯誤率: {(len(error_cases) / len(df_raw) * 100):.1f}%")