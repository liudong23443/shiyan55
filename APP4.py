import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.font_manager import FontProperties, fontManager
import matplotlib.colors as mcolors
import os
import platform
import warnings
import matplotlib as mpl
import io
import base64
warnings.filterwarnings('ignore')

# 添加中文字体文件（尝试解决Streamlit云环境中的字体问题）
def setup_chinese_fonts():
    # 在Streamlit Cloud环境中下载并安装思源黑体
    import tempfile
    import os
    import urllib.request
    
    # 检查是否在Streamlit Cloud环境中
    if os.getenv('STREAMLIT_SHARING') or os.getenv('STREAMLIT_CLOUD'):
        try:
            # 创建临时目录存放字体文件
            font_dir = tempfile.mkdtemp()
            # 思源黑体下载链接
            font_url = "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"
            font_path = os.path.join(font_dir, "SourceHanSansSC-Regular.otf")
            
            # 下载字体文件
            urllib.request.urlretrieve(font_url, font_path)
            
            # 添加字体文件到matplotlib的字体管理器
            fontManager.addfont(font_path)
            
            # 设置字体缓存
            mpl.font_manager._rebuild()
            
            return ['Source Han Sans SC', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Droid Sans Fallback']
        except Exception as e:
            st.warning(f"下载中文字体失败: {str(e)}，将使用系统默认字体")
            return ['sans-serif']
    else:
        # 本地环境使用系统字体
        return ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']

# 根据操作系统不同设置合适的字体
system = platform.system()
if system == 'Windows':
    font_list = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
else:  # Linux 或 MacOS
    font_list = setup_chinese_fonts()

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = font_list
plt.rcParams['axes.unicode_minus'] = False

# 确保matplotlib的全局字体设置
plt.rcParams.update({'font.family': 'sans-serif'})
# 确保plotly也能显示中文
import plotly.io as pio
pio.templates.default = "simple_white"

# 设置页面配置
st.set_page_config(
    page_title="胃癌术后生存预测",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: system-ui, -apple-system, 'Segoe UI', Roboto, 'Microsoft YaHei', 'SimHei', sans-serif;
        padding: 0.8rem 0;
        border-bottom: 2px solid #E5E7EB;
    }
    .sub-header {
        font-size: 1.2rem;
        color: white;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        font-family: system-ui, -apple-system, 'Segoe UI', Roboto, 'Microsoft YaHei', 'SimHei', sans-serif;
    }
    .description {
        font-size: 1rem;
        color: #4B5563;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        border-left: 4px solid #1E3A8A;
    }
    .section-container {
        padding: 0.8rem;
        background-color: #F9FAFB;
        border-radius: 0.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        margin-bottom: 0.8rem;
        height: 100%;
    }
    .results-container {
        padding: 0.8rem;
        background-color: #F0F9FF;
        border-radius: 0.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        margin-bottom: 0.8rem;
        border: 1px solid #93C5FD;
        height: 100%;
    }
    .metric-card {
        background-color: #F0F9FF;
        padding: 0.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        text-align: center;
    }
    .disclaimer {
        font-size: 0.75rem;
        color: #6B7280;
        text-align: center;
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid #E5E7EB;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        border-radius: 0.3rem;
        border: none;
        margin-top: 0.5rem;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1E40AF;
    }
    /* 改善小型设备上的响应式布局 */
    @media (max-width: 1200px) {
        .main-header {
            font-size: 1.5rem;
        }
        .sub-header {
            font-size: 1.1rem;
        }
    }
    /* 隐藏Streamlit默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    /* 优化指标显示 */
    .stMetric {
        background-color: transparent;
        padding: 5px;
        border-radius: 5px;
    }
    /* 改进分割线 */
    hr {
        margin: 0.8rem 0;
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(0,0,0,0), rgba(0,0,0,0.1), rgba(0,0,0,0));
    }
    /* 仪表盘和SHAP图中的文字加深 */
    .js-plotly-plot .plotly .gtitle {
        font-weight: bold !important;
        fill: #000000 !important;
    }
    .js-plotly-plot .plotly .g-gtitle {
        font-weight: bold !important;
        fill: #000000 !important;
    }
    /* 图表背景 */
    .stPlotlyChart, .stImage {
        background-color: white !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
        font-weight: bold !important;
        color: #1E3A8A !important;
    }
    div[data-testid="stMetricLabel"] {
        font-weight: bold !important;
        font-size: 0.9rem !important;
    }
    /* 紧凑化滑块和单选按钮 */
    div.row-widget.stRadio > div {
        flex-direction: row;
        align-items: center;
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label {
        padding: 0.2rem 0.5rem;
        min-height: auto;
    }
    div.stSlider {
        padding-top: 0.3rem;
        padding-bottom: 0.5rem;
    }
    /* 紧凑化标签 */
    p {
        margin-bottom: 0.3rem;
    }
    div.stMarkdown p {
        margin-bottom: 0.3rem;
    }
    /* 美化进度条区域 */
    .progress-container {
        background-color: #f0f7ff;
        border-radius: 0.3rem;
        padding: 0.4rem;
        margin-bottom: 0.5rem;
        border: 1px solid #dce8fa;
    }
    
    /* 改善左右对齐 */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* 确保滑块组件对齐 */
    .stSlider > div {
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    
    /* 缩小图表外边距 */
    .stPlotlyChart > div, .stImage > img {
        margin: 0 auto !important;
        padding: 0 !important;
    }
    
    /* 使侧边栏更紧凑 */
    section[data-testid="stSidebar"] div.stMarkdown p {
        margin-bottom: 0.2rem;
    }
    
    /* 更紧凑的标题 */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        margin-top: 0.2rem;
        margin-bottom: 0.2rem;
    }
    
    /* 使结果区域更紧凑 */
    .results-container > div {
        margin-bottom: 0.4rem !important;
    }
</style>
""", unsafe_allow_html=True)

# 加载保存的随机森林模型
@st.cache_resource
def load_model():
    try:
        model = joblib.load('rf1.pkl')
        # 添加模型信息
        if hasattr(model, 'n_features_in_'):
            st.session_state['model_n_features'] = model.n_features_in_
            st.session_state['model_feature_names'] = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
        return model
    except Exception as e:
        st.error(f"⚠️ 模型文件 'rf.pkl' 加载错误: {str(e)}。请确保模型文件在正确的位置。")
        return None

model = load_model()

# 侧边栏配置和调试信息
with st.sidebar:
    st.markdown("### 模型信息")
    if model is not None and hasattr(model, 'n_features_in_'):
        st.info(f"模型期望特征数量: {model.n_features_in_}")
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
            st.write("模型期望特征列表:", expected_features)
    
    st.markdown("---")
    st.markdown("### 应用说明")
    st.markdown("""
    本应用基于随机森林算法构建，通过分析胃癌患者的关键临床特征，预测术后三年内的死亡风险。

    **使用方法：**
    1. 在右侧输入患者特征
    2. 点击"开始预测"按钮
    3. 查看预测结果与解释
    """)

# 特征范围定义
feature_ranges = {
    "术中出血量": {"type": "numerical", "min": 0.000, "max": 800.000, "default": 50, 
                                 "description": "手术期间的出血量 (ml)", "unit": "ml"},
    "CEA": {"type": "numerical", "min": 0, "max": 150.000, "default": 8.68, 
           "description": "癌胚抗原水平", "unit": "ng/ml"},
    "白蛋白": {"type": "numerical", "min": 1.0, "max": 80.0, "default": 38.60, 
               "description": "血清白蛋白水平", "unit": "g/L"},
    "TNM分期": {"type": "categorical", "options": [1, 2, 3, 4], "default": 2, 
                 "description": "肿瘤分期", "unit": ""},
    "年龄": {"type": "numerical", "min": 25, "max": 90, "default": 76, 
           "description": "患者年龄", "unit": "岁"},
    "术中肿瘤最大直径": {"type": "numerical", "min": 0.2, "max": 20, "default": 4, 
                          "description": "肿瘤最大直径", "unit": "cm"},
    "淋巴血管侵犯": {"type": "categorical", "options": [0, 1], "default": 1, 
                              "description": "淋巴血管侵犯 (0=否, 1=是)", "unit": ""},
}

# 特征顺序定义 - 确保与模型训练时的顺序一致
if model is not None and hasattr(model, 'feature_names_in_'):
    feature_input_order = list(model.feature_names_in_)
    feature_ranges_ordered = {}
    for feature in feature_input_order:
        if feature in feature_ranges:
            feature_ranges_ordered[feature] = feature_ranges[feature]
        else:
            # 模型需要但UI中没有定义的特征
            with st.sidebar:
                st.warning(f"模型要求特征 '{feature}' 但在UI中未定义")
    
    # 检查UI中定义但模型不需要的特征
    for feature in feature_ranges:
        if feature not in feature_input_order:
            with st.sidebar:
                st.warning(f"UI中定义的特征 '{feature}' 不在模型要求的特征中")
    
    # 使用排序后的特征字典
    feature_ranges = feature_ranges_ordered
else:
    # 如果模型没有feature_names_in_属性，使用原来的顺序
    feature_input_order = list(feature_ranges.keys())

# 应用标题和描述
st.markdown('<h1 class="main-header">胃癌术后三年生存预测模型</h1>', unsafe_allow_html=True)

# 创建两列布局，调整为更合适的比例
col1, col2 = st.columns([3.5, 6.5], gap="small")

with col1:
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">患者特征输入</h2>', unsafe_allow_html=True)
    
    # 动态生成输入项 - 更紧凑布局
    feature_values = {}
    
    for feature in feature_input_order:
        properties = feature_ranges[feature]
        
        # 显示特征描述 - 根据变量类型生成不同的帮助文本
        if properties["type"] == "numerical":
            help_text = f"{properties['description']} ({properties['min']}-{properties['max']} {properties['unit']})"
            
            # 为数值型变量创建滑块 - 使用更紧凑的布局
            value = st.slider(
                label=f"{feature}",
                min_value=float(properties["min"]),
                max_value=float(properties["max"]),
                value=float(properties["default"]),
                step=0.1,
                help=help_text,
                # 使布局更紧凑
            )
        elif properties["type"] == "categorical":
            # 对于分类变量，只使用描述作为帮助文本
            help_text = f"{properties['description']}"
            
            # 为分类变量创建单选按钮
            if feature == "TNM分期":
                options_display = {1: "I期", 2: "II期", 3: "III期", 4: "IV期"}
                value = st.radio(
                    label=f"{feature}",
                    options=properties["options"],
                    format_func=lambda x: options_display[x],
                    help=help_text,
                    horizontal=True
                )
            elif feature == "淋巴血管侵犯":
                options_display = {0: "否", 1: "是"}
                value = st.radio(
                    label=f"{feature}",
                    options=properties["options"],
                    format_func=lambda x: options_display[x],
                    help=help_text,
                    horizontal=True
                )
            else:
                value = st.radio(
                    label=f"{feature}",
                    options=properties["options"],
                    help=help_text,
                    horizontal=True
                )
                
        feature_values[feature] = value
    
    # 预测按钮
    predict_button = st.button("开始预测", help="点击生成预测结果")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if predict_button and model is not None:
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">预测结果</h2>', unsafe_allow_html=True)
        
        # 准备模型输入
        features_df = pd.DataFrame([feature_values])
        
        # 确保特征顺序与模型训练时一致
        if hasattr(model, 'feature_names_in_'):
            # 检查是否所有需要的特征都有值
            missing_features = [f for f in model.feature_names_in_ if f not in features_df.columns]
            if missing_features:
                st.error(f"缺少模型所需的特征: {missing_features}")
                st.stop()
            
            # 按模型训练时的特征顺序重排列特征
            features_df = features_df[model.feature_names_in_]
        
        # 转换为numpy数组
        features_array = features_df.values
        
        with st.spinner("计算预测结果..."):
            try:
                # 模型预测
                predicted_class = model.predict(features_array)[0]
                predicted_proba = model.predict_proba(features_array)[0]
                
                # 提取预测的类别概率
                death_probability = predicted_proba[1] * 100  # 假设1表示死亡类
                survival_probability = 100 - death_probability
                
                # 创建概率显示 - 进一步减小尺寸
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = death_probability,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "", 'font': {'size': 14, 'family': font_list[0] if font_list else 'sans-serif', 'color': 'black', 'weight': 'bold'}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue", 'tickfont': {'color': 'black', 'size': 9}},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 1,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': 'green'},
                            {'range': [30, 70], 'color': 'orange'},
                            {'range': [70, 100], 'color': 'red'}],
                        'threshold': {
                            'line': {'color': "red", 'width': 2},
                            'thickness': 0.6,
                            'value': death_probability}}))
                
                fig.update_layout(
                    height=160,  # 进一步减小高度
                    margin=dict(l=5, r=5, t=5, b=5),  # 减小顶部边距
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font={'family': font_list[0] if font_list else 'sans-serif', 'color': 'black', 'size': 11},
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 创建风险类别显示
                risk_category = "低风险"
                risk_color = "green"
                if death_probability > 30 and death_probability <= 70:
                    risk_category = "中等风险"
                    risk_color = "orange"
                elif death_probability > 70:
                    risk_category = "高风险"
                    risk_color = "red"
                
                # 显示风险类别和概率 - 使用浅色背景代替白色
                st.markdown(f"""
                <div style="text-align: center; margin: -0.2rem 0 0.3rem 0;">
                    <span style="font-size: 1.1rem; font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; color: {risk_color}; font-weight: bold;">
                        {risk_category}
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                # 显示具体概率数值 - 放入浅色背景容器
                st.markdown(f"""
                <div class="progress-container">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.2rem;">
                        <div style="text-align: center; width: 48%;">
                            <div style="font-size: 0.9rem; font-weight: bold; color: #1E3A8A;">三年生存概率</div>
                            <div style="font-size: 1.1rem; font-weight: bold; color: #10B981;">{survival_probability:.1f}%</div>
                        </div>
                        <div style="text-align: center; width: 48%;">
                            <div style="font-size: 0.9rem; font-weight: bold; color: #1E3A8A;">三年死亡风险</div>
                            <div style="font-size: 1.1rem; font-weight: bold; color: #EF4444;">{death_probability:.1f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 添加SHAP可视化部分 - 减小间距
                st.markdown('<hr style="margin:0.3rem 0;">', unsafe_allow_html=True)
                st.markdown('<h2 class="sub-header">预测结果解释</h2>', unsafe_allow_html=True)
                
                try:
                    with st.spinner("正在生成SHAP解释图..."):
                        # 使用最新版本的SHAP API，采用最简洁、最兼容的方式
                        explainer = shap.Explainer(model)
                        
                        # 计算SHAP值
                        shap_values = explainer(features_df)
                        
                        # 提取特征名称和SHAP值
                        feature_names = list(features_df.columns)
                        if hasattr(shap_values, 'values') and len(shap_values.values.shape) > 2:
                            # 多分类情况 - 选择第二个类别(通常是正类/死亡类)
                            shap_vals = shap_values[0, :, 1].values
                        else:
                            # 二分类或回归情况
                            if hasattr(shap_values, 'values'):
                                shap_vals = shap_values[0].values
                            else:
                                shap_vals = shap_values[0]
                        
                        # 创建特征重要性DataFrame
                        importance_df = pd.DataFrame({
                            '特征': feature_names,
                            'SHAP值': shap_vals,
                            '绝对值': np.abs(shap_vals)
                        })
                        
                        # 按绝对值排序并选择前7个特征
                        importance_df = importance_df.sort_values('绝对值', ascending=False).head(7)
                        
                        # 计算基准值
                        base_value = explainer.expected_value
                        if isinstance(base_value, np.ndarray) and len(base_value) > 1:
                            base_value = base_value[1]  # 假设是二分类，取第二个类别
                        elif isinstance(base_value, list) and len(base_value) > 1:
                            base_value = base_value[1]
                            
                        if isinstance(base_value, np.ndarray) and base_value.size == 1:
                            base_value = float(base_value.item())
                        
                        # 格式化为小数点后3位
                        try:
                            base_value_formatted = float(f"{base_value:.3f}")
                        except:
                            base_value_formatted = 0.5  # 默认值
                        
                        # 准备特征标签和值
                        feature_labels_with_values = []
                        for i, row in importance_df.iterrows():
                            feature = row['特征']
                            # 获取该特征的值
                            if feature in feature_values:
                                value = feature_values[feature]
                                # 处理分类特征
                                if feature == "TNM分期":
                                    value_display = f"{int(value)}期"
                                elif feature == "淋巴血管侵犯":
                                    value_display = "是" if value == 1 else "否"
                                else:
                                    value_display = f"{value}"
                                feature_labels_with_values.append(f"{value_display} = {feature}")
                            else:
                                feature_labels_with_values.append(feature)
                        
                        # 我们将仅绘制不带中文标签的简单条形图
                        plt.figure(figsize=(8, 4), dpi=100, facecolor='white')
                        
                        # 按照索引排列
                        y_pos = np.arange(len(feature_labels_with_values))
                        
                        # 绘制水平条形图，使用序号代替中文标签
                        colors = ['#ff4d4d' if x > 0 else '#2196F3' for x in importance_df['SHAP值']]
                        bars = plt.barh(y_pos, importance_df['SHAP值'].values, color=colors, alpha=0.8)
                        
                        # 设置Y轴为数字标签（暂时）
                        plt.yticks(y_pos, range(len(feature_labels_with_values)))
                        
                        # 添加垂直中轴线
                        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
                        
                        # 设置X轴范围
                        max_val = max(abs(importance_df['SHAP值'])) * 1.2
                        plt.xlim(-max_val, max_val)
                        
                        # 添加X轴标签
                        plt.xlabel("SHAP值", fontsize=10)
                        
                        # 美化图表
                        plt.grid(axis='x', linestyle='--', alpha=0.3)
                        plt.box(False)  # 移除边框
                        
                        # 使用紧凑布局
                        plt.tight_layout()
                        
                        # 保存基本图表到内存中
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', dpi=100)
                        plt.close()
                        
                        # 使用PIL打开图像
                        buf.seek(0)
                        img = Image.open(buf)
                        
                        # 创建一个全新的白色画布，更大，以便有足够空间放置文本
                        canvas_width = 800
                        canvas_height = 500
                        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
                        
                        # 将原始条形图粘贴到新画布中间部分
                        graph_paste_x = 250  # 左侧留出足够空间放置特征标签
                        graph_paste_y = 50   # 顶部留出空间放置标题
                        canvas.paste(img, (graph_paste_x, graph_paste_y))
                        
                        # 创建绘图对象
                        draw = ImageDraw.Draw(canvas)
                        
                        # 尝试加载中文字体
                        try:
                            # 尝试加载字体
                            font_path = None
                            # 在Windows上查找常见中文字体
                            if platform.system() == 'Windows':
                                for path in [
                                    "C:\\Windows\\Fonts\\msyh.ttc",    # 微软雅黑
                                    "C:\\Windows\\Fonts\\simhei.ttf",  # 黑体
                                    "C:\\Windows\\Fonts\\simsun.ttc"   # 宋体
                                ]:
                                    if os.path.exists(path):
                                        font_path = path
                                        break
                            else:
                                # 在Linux/MacOS上尝试常见中文字体
                                for path in [
                                    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                                    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                                    "/System/Library/Fonts/STHeiti Light.ttc"
                                ]:
                                    if os.path.exists(path):
                                        font_path = path
                                        break
                            
                            # 如果未找到字体，使用Streamlit提供的默认sans-serif字体
                            if font_path is None:
                                # 创建临时字体文件
                                import tempfile
                                tmp_dir = tempfile.mkdtemp()
                                try:
                                    # 从GitHub下载思源黑体
                                    import urllib.request
                                    font_url = "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"
                                    font_path = os.path.join(tmp_dir, "SourceHanSansSC-Regular.otf")
                                    urllib.request.urlretrieve(font_url, font_path)
                                except:
                                    # 使用系统默认
                                    font_path = "DejaVuSans.ttf"
                            
                            # 加载字体
                            title_font = ImageFont.truetype(font_path, 20)
                            feature_font = ImageFont.truetype(font_path, 15)
                            value_font = ImageFont.truetype(font_path, 13)
                            small_font = ImageFont.truetype(font_path, 12)
                        except Exception as font_error:
                            st.warning(f"加载字体失败: {str(font_error)}，将使用默认字体")
                            # 使用PIL默认字体
                            title_font = ImageFont.load_default()
                            feature_font = ImageFont.load_default()
                            value_font = ImageFont.load_default()
                            small_font = ImageFont.load_default()
                        
                        # 添加标题
                        title_text = "特征对预测的影响"
                        title_width = draw.textlength(title_text, font=title_font)
                        draw.text((canvas_width // 2 - title_width // 2, 15), title_text, fill="black", font=title_font)
                        
                        # 根据条形图在新画布中的位置计算条形图的Y坐标范围
                        # 假设条形图在原始图像中占据了大部分高度
                        graph_height = img.height * 0.7  # 条形图在原始图像中的高度比例
                        bar_area_top = graph_paste_y + img.height * 0.15  # 条形图在原始图像中的顶部位置
                        bar_area_bottom = bar_area_top + graph_height
                        
                        # 计算每个条形的高度和位置
                        num_bars = len(feature_labels_with_values)
                        bar_height = graph_height / num_bars
                        
                        # 添加特征标签到左侧，确保与条形图对齐
                        for i, label in enumerate(feature_labels_with_values):
                            # 计算当前条形的中心Y坐标
                            bar_center_y = bar_area_top + i * bar_height + bar_height / 2
                            # 右对齐文本
                            label_width = draw.textlength(label, font=feature_font)
                            draw.text((graph_paste_x - label_width - 10, bar_center_y - 10), label, fill="black", font=feature_font)
                        
                        # 计算SHAP值的位置和显示
                        center_x = graph_paste_x + img.width / 2  # 图表中心X坐标
                        graph_width = img.width * 0.7  # 估计条形图在原图中的宽度
                        
                        for i, value in enumerate(importance_df['SHAP值']):
                            # 计算当前条形的中心Y坐标
                            bar_center_y = bar_area_top + i * bar_height + bar_height / 2
                            
                            # 为正负值选择不同的位置
                            if value > 0:
                                # 正值在条形图右侧
                                x_position = center_x + (value / max_val) * (graph_width / 2) + 5
                                text = f"+{value:.2f}"
                                draw.text((x_position, bar_center_y - 8), text, fill="black", font=value_font)
                            else:
                                # 负值在条形图左侧
                                x_position = center_x + (value / max_val) * (graph_width / 2) - 5
                                text = f"{value:.2f}"
                                text_width = draw.textlength(text, font=value_font)
                                draw.text((x_position - text_width, bar_center_y - 8), text, fill="black", font=value_font)
                        
                        # 添加X轴标签
                        x_label = "SHAP值"
                        x_label_width = draw.textlength(x_label, font=feature_font)
                        draw.text((center_x - x_label_width / 2, bar_area_bottom + 20), x_label, fill="black", font=feature_font)
                        
                        # 添加基准值文本
                        base_text = f"基准值 f(x) = {base_value_formatted}"
                        draw.text((20, canvas_height - 30), base_text, fill="black", font=small_font)
                        
                        # 添加图例
                        legend_y1 = canvas_height - 60
                        legend_y2 = canvas_height - 40
                        
                        # 红色方块 - 增加风险
                        draw.rectangle([(canvas_width - 180, legend_y1), (canvas_width - 160, legend_y1 + 15)], fill="#ff4d4d")
                        draw.text((canvas_width - 155, legend_y1), "增加风险", fill="black", font=small_font)
                        
                        # 蓝色方块 - 降低风险
                        draw.rectangle([(canvas_width - 180, legend_y2), (canvas_width - 160, legend_y2 + 15)], fill="#2196F3")
                        draw.text((canvas_width - 155, legend_y2), "降低风险", fill="black", font=small_font)
                        
                        # 保存和显示最终图像
                        canvas.save("shap_effect_plot_final.png")
                        st.image("shap_effect_plot_final.png")
                        
                        # 添加简要解释 - 更紧凑，使用浅色背景
                        st.markdown("""
                        <div style="background-color: #f0f7ff; padding: 5px; border-radius: 3px; margin-top: 3px; font-size: 0.8rem; border: 1px solid #dce8fa;">
                          <p style="margin:0"><strong>图表解释:</strong> 红色条表示该特征增加死亡风险，蓝色条表示该特征降低死亡风险。数值表示对预测结果的贡献大小。</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                except Exception as shap_error:
                    st.error(f"生成SHAP图时出错: {str(shap_error)}")
                    st.warning("无法生成SHAP解释图，请联系技术支持。")
                
            except Exception as e:
                st.error(f"预测过程中发生错误: {str(e)}")
                st.warning("请检查输入数据是否与模型期望的特征匹配，或联系开发人员获取支持。")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # 当没有点击预测按钮时，不显示任何内容
        pass

# 添加页脚说明
st.markdown("""
<div class="disclaimer">
    <p>📋 免责声明：本预测工具仅供临床医生参考，不能替代专业医疗判断。预测结果应结合患者的完整临床情况进行综合评估。</p>
    <p>© 2025 | 开发版本 v1.1.0</p>
</div>
""", unsafe_allow_html=True) 