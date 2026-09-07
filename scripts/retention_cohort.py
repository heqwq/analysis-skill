import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 模拟数据生成（50个玩家，30天登录记录）
np.random.seed(42)
n_users = 50
n_days = 30

register_dates = pd.date_range('2026-01-01', periods=n_days)
user_ids = range(1, n_users + 1)

records = []
for user_id in user_ids:
    register_dt = np.random.choice(register_dates, 1)[0]
    # 模拟登录行为：注册后登录天数逐渐减少
    login_days = max(1, int(np.random.exponential(5)) + 1)
    active_dates = pd.date_range(register_dt, periods=min(login_days, n_days))
    active_dates = [d for d in active_dates if d <= register_dates[-1]]
    for active_dt in active_dates:
        records.append({
            'role_id': f'user_{user_id:03d}',
            'register_dt': register_dt.strftime('%Y-%m-%d'),
            'active_dt': active_dt.strftime('%Y-%m-%d')
        })

df = pd.DataFrame(records)

# --- 数据清洗 ---
df['register_dt'] = pd.to_datetime(df['register_dt'])
df['active_dt'] = pd.to_datetime(df['active_dt'])

# 计算注册后第几天登录（cohort index）
df['cohort_index'] = (df['active_dt'] - df['register_dt']).dt.days

# Cohort 留存计算
cohort_data = df.groupby(['register_dt', 'cohort_index'])['role_id'].nunique().reset_index()
cohort_pivot = cohort_data.pivot_table(
    index='register_dt',
    columns='cohort_index',
    values='role_id'
)

cohort_size = cohort_pivot.iloc[:, 0]  # 第0天=注册当天人数
retention = cohort_pivot.divide(cohort_size, axis=0)

# --- 可视化 ---
plt.figure(figsize=(12, 8))
sns.heatmap(retention, annot=True, fmt='.0%', cmap='Blues',
            cbar_kws={'label': 'Retention Rate'})
plt.title('Cohort Retention Heatmap\n(模拟数据)', fontsize=14)
plt.xlabel('Days After Registration', fontsize=12)
plt.ylabel('Registration Cohort', fontsize=12)
plt.tight_layout()
plt.savefig('cohort_heatmap.png', dpi=150)
plt.show()

print("✅ Cohort 热力图已生成: cohort_heatmap.png")
print(f"   数据概览: {len(df)} 条登录记录, {df['role_id'].nunique()} 个玩家")
