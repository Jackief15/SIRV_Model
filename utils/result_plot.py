import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_double_plot(data, col='x', title=None,
                         max_outer=None,         # 只畫前幾個外層 (gamma, beta)，例如 (3,3)
                         downsample=1,           # 內層 (eta, C) 每隔幾格取一次
                         cmap='jet'):
    # ---------- 1) 取唯一值 (避免排序開銷過多，必要時才排序) ----------
    outer_x = np.sort(data['beta'].unique())                  # β 升序
    outer_y = np.sort(data['gamma'].unique())[::-1]           # γ 降序
    inner_x = np.sort(data['C'].unique())                     # C
    inner_y = np.sort(data['eta'].unique())                   # η

    B, G, Cn, En = len(outer_x), len(outer_y), len(inner_x), len(inner_y)

    # ---------- 2) 把 (β,γ,η,C) 映射到 4D array 位置，避開 pivot ----------
    # 建類別編碼（O(n)）
    bi = pd.Categorical(data['beta'], categories=outer_x, ordered=True).codes
    gi = pd.Categorical(data['gamma'], categories=outer_y, ordered=True).codes
    ci = pd.Categorical(data['C'],    categories=inner_x, ordered=True).codes
    ei = pd.Categorical(data['eta'],  categories=inner_y, ordered=True).codes

    arr = np.full((G, B, En, Cn), np.nan, dtype=float)
    vals = data[col].to_numpy()
    arr[gi, bi, ei, ci] = vals   # 直接填值

    # ---------- 3) 色階範圍 ----------
    if col == 'payoff_D':
        vmin, vmax = -1.0, 0.0
    elif col == 'payoff_C':
        vmin, vmax = -1.0, 1.0
    else:
        vmin, vmax = 0.0, 1.0

    # ---------- 4) 限縮繪製區域 + 降採樣 ----------
    g_lim = G if not max_outer else min(G, max_outer[0])
    b_lim = B if not max_outer else min(B, max_outer[1])
    # 內層降採樣（為了看 layout/字體很夠用）
    arr_view = arr[:g_lim, :b_lim, ::downsample, ::downsample]
    inner_y_ticks = inner_y[::downsample]
    inner_x_ticks = inner_x[::downsample]

    # ---------- 5) 畫圖：用 imshow（快），關掉多餘 ticks ----------
    fig, axes = plt.subplots(nrows=g_lim, ncols=b_lim,
                             figsize=(20, 20), sharex=False, sharey=False)
    if g_lim == 1 and b_lim == 1:
        axes = np.array([[axes]])
    elif g_lim == 1:
        axes = axes[np.newaxis, :]
    elif b_lim == 1:
        axes = axes[:, np.newaxis]

    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in range(g_lim):
        for j in range(b_lim):
            ax = axes[i, j]
            # 注意：原本你用 [::-1] 讓 η 反序，本質是把 row 上下翻轉。
            # imshow 用 origin='upper' 就能符合「上方是大的 η」的直覺，不需額外複製數據。
            im = ax.imshow(arr_view[i, j], vmin=vmin, vmax=vmax, cmap=cmap,
                           origin='lower', aspect='auto', interpolation='nearest',
                           rasterized=True)

            # 只有左下角保留 tick（且只標 0.1 / 1 的兩端）
            if i == g_lim - 1 and j == 0:
                ax.set_xticks([0, arr_view.shape[-1]-1])
                ax.set_xticklabels([r"0.1", r"1.0"], fontsize=32)
                ax.set_yticks([0, arr_view.shape[-2]-1])
                ax.set_yticklabels([r"1.0", r"0.1"], fontsize=32)
                ax.set_xlabel('')
                ax.set_ylabel('')
            else:
                ax.set_xticks([0, arr_view.shape[-1]-1])
                ax.set_xticklabels(["", ""], fontsize=32)
                ax.set_yticks([0, arr_view.shape[-2]-1])
                ax.set_yticklabels(["", ""], fontsize=32)

    # ---------- 6) 外層標籤 + 單一 colorbar ----------
    fig.text(0.5, 0.04, r'$\beta$', ha='center', va='center', fontsize=48)
    fig.text(0.04, 0.5, r'$\gamma$', ha='center', va='center',
             rotation=0, fontsize=48)
    fig.text(0.16, 0.08, r'$C$', ha='center', va='center', fontsize=48)
    fig.text(0.1, 0.15, r'$\eta$', ha='center', va='center', fontsize=48)
    fig.text(0.16, 0.04, r'$0.1$', ha='center', va='center', fontsize=48)
    fig.text(0.84, 0.04, r'$1.0$', ha='center', va='center', fontsize=48)
    fig.text(0.06, 0.14, r'$0.1$', ha='center', va='center', fontsize=48)
    fig.text(0.06, 0.84, r'$1.0$', ha='center', va='center', fontsize=48)

    if title is not None:
        fig.text(0.05, 0.9, title, ha='left', va='bottom', fontsize=52)


    # 共享 colorbar（用最後一個 im 的 mappable）
    cbar = fig.colorbar(axes[0, 0].images[0], ax=axes, orientation='vertical',
                        fraction=0.02, aspect=30, pad=0.03)
    cbar.ax.tick_params(labelsize=48)
    
    # === KEY CHANGE: 新增程式碼以繪製指定的外圍刻度 ===
    # ---------- 7) 繪製指定位置的外層刻度線 ---
    outer_tick_len = 0.015  # 外圍刻度線的長度
    outer_pad = 0.02       # 數字標籤與刻度線的間距

    # --- 外部 X 軸 (beta) 的指定刻度 ---
    target_betas = [0.1, 0.4, 0.7, 1.0]
    for beta_val in target_betas:
        # 找到最接近的 beta 值及其索引 j
        j = np.argmin(np.abs(outer_x - beta_val))
        
        # 獲取對應的子圖 ax，並計算其水平中心
        ax = axes[-1, j] # 以最下面一排的 ax 為基準
        pos = ax.get_position()
        x_center = pos.x0 + pos.width / 2
        
        # 在底部畫刻度線和文字
        line_bottom = mlines.Line2D([x_center, x_center], [pos.y0, pos.y0 - outer_tick_len], 
                                    color='black', transform=fig.transFigure)
        fig.add_artist(line_bottom)
        fig.text(x_center, pos.y0 - outer_tick_len - outer_pad, f"{beta_val:.1f}", 
                 ha='center', va='top', fontsize=32, transform=fig.transFigure)
                 
    # --- 外部 Y 軸 (gamma) 的指定刻度 ---
    target_gammas = [0.1, 0.4, 0.7, 1.0]
    for gamma_val in target_gammas:
        # 找到最接近的 gamma 值及其索引 i
        i = np.argmin(np.abs(outer_y - gamma_val))

        # 獲取對應的子圖 ax，並計算其垂直中心
        ax = axes[i, 0] # 以最左邊一排的 ax 為基準
        pos = ax.get_position()
        y_center = pos.y0 + pos.height / 2

        # 在左邊畫刻度線和文字
        line_left = mlines.Line2D([pos.x0, pos.x0 - outer_tick_len], [y_center, y_center], 
                                  color='black', transform=fig.transFigure)
        fig.add_artist(line_left)
        fig.text(pos.x0 - outer_tick_len - outer_pad, y_center, f"{gamma_val:.1f}", 
                 ha='right', va='center', fontsize=32, transform=fig.transFigure)

    plt.show()
    return

def plot_cross_section_with_std(data, dep_var_base, indep_var, fixed_param_sets):
    """
    (ax物件導向版本)
    Plots cross-section data, showing the mean as a line with markers 
    and the standard deviation as a shaded error band.
    """
    # === KEY CHANGE 1: 使用 subplots() 建立 fig 和 ax ===
    fig, ax = plt.subplots(figsize=(25, 20))
    
    mean_col = f"{dep_var_base}_mean"
    std_col = f"{dep_var_base}_std"

    markers = ['o', 'v', 's', 'D', 'h', 'P']
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(fixed_param_sets)))

    for i, fixed_params in enumerate(fixed_param_sets):
        filtered_data = data.copy()
        for param, value in fixed_params.items():
            if param in data.columns:
                filtered_data = filtered_data[filtered_data[param] == value]

        if filtered_data.empty:
            print(f"No data available for {fixed_params}. Skipping.")
            continue

        filtered_data = filtered_data.sort_values(by=indep_var)

        vari = list(fixed_params.items())[-1][0]
        if vari != 'C':
            vari = f"\\{vari}"
        val = list(fixed_params.items())[-1][1]
        label = f"${vari} = {str(val)}$"
        
        # === KEY CHANGE 2: 所有繪圖操作都從 plt. 改為 ax. ===
        # Plot the mean value line
        ax.plot(
            filtered_data[indep_var], 
            filtered_data[mean_col], 
            color=colors[i],
            marker=markers[i % len(markers)], 
            linestyle='-', 
            label=label, 
            lw=4,
            markersize=30,
            mew=1.5,
            mec='black'
        )
        
        # Plot the standard deviation as a shaded area
        ax.fill_between(
            filtered_data[indep_var],
            filtered_data[mean_col] - filtered_data[std_col],
            filtered_data[mean_col] + filtered_data[std_col],
            color=colors[i],
            alpha=0.2
        )
    
    # === KEY CHANGE 3: 所有客製化操作也都從 plt. 改為 ax.set_... ===
    if indep_var == 'C':
        ax.set_xlabel(f'${indep_var}$', fontsize=64, labelpad=30)
    else:
        ax.set_xlabel(f'$\\{indep_var}$', fontsize=64, labelpad=30)
    
    ax.set_ylabel(fr'FOV (${dep_var_base}$)', fontsize=64, labelpad=30)
    ax.legend(fontsize=56, loc='upper left', bbox_to_anchor=(1, 1))
    ax.tick_params(axis='both', which='major', labelsize=56)
    ax.text(-0.05, 1.02, '(d)', fontsize=64, transform=ax.transAxes, ha='left', va='bottom')
    # ax.grid(True, linestyle='--', alpha=0.5)
    
    # fig.tight_layout(rect=[0, 0, 0.85, 1])
    
    # 顯示圖表
    plt.show()
    
    # 讓函數回傳 fig 和 ax 物件，方便後續可能需要的微調
    return fig, ax

def get_agg_data(data):
    grouped_data = data[cols + ['x']].groupby(cols).agg(['mean', 'std'])
    flat_columns = ['_'.join(col).strip() for col in grouped_data.columns.values]
    grouped_data.columns = flat_columns
    grouped_data = grouped_data.reset_index()
    return grouped_data

df = pd.read_csv(p / 'Final_cross_section' / 'MC_sim.csv')

cols = ['C', 'eta', 'beta', 'gamma']

for col in ['C', 'eta', 'beta', 'gamma', 'theta']:
    df[col] = df[col].round(1)

# data1 = df[df['run_id'] <= 10]                              # C=0.2, eta=0.8, gamma=0.1,0.3,0.5,0.7,0.9
# data2 = df[(df['run_id'] > 10) & (df['run_id'] <= 20)]      # C=0.2, eta=0.8, beta=0.1,0.3,0.5,0.7,0.9
# data3 = df[(df['run_id'] > 20) & (df['run_id'] <= 30)]              # beta=0.8, gamma=0.2


plot_cross_section_with_std(
    data=get_agg_data(data1),
    dep_var_base='x',
    indep_var='beta',
    fixed_param_sets=[
        {'C': 0.2, 'eta': 0.8, 'gamma': 0.1},
        {'C': 0.2, 'eta': 0.8, 'gamma': 0.3},
        {'C': 0.2, 'eta': 0.8, 'gamma': 0.5},
        {'C': 0.2, 'eta': 0.8, 'gamma': 0.7},
        {'C': 0.2, 'eta': 0.8, 'gamma': 0.9}
    ]
)