import streamlit as st
import yfinance as yf
import pandas as pd
import concurrent.futures
import datetime
import warnings
import time
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from backtesting import Backtest, Strategy

# 忽略警告
warnings.filterwarnings("ignore")

# ==========================================
# ⚙️ 頁面設定 (必須是第一個 st 指令)
# ==========================================
st.set_page_config(
    page_title="強勢股狙擊手戰情室",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 依賴檢查 (twstock) ---
try:
    import twstock
except ImportError as e:
    if "lxml" in str(e) or "twstock" in str(e):
        st.error("❌ 啟動失敗：缺少 `lxml` 套件")
        st.info("請在 CMD 輸入: pip install lxml")
        st.stop()
    else:
        raise e

# 自訂 CSS
st.markdown("""
<style>
    .stDataFrame {font-size: 1.1rem;}
    [data-testid="stMetricValue"] {font-size: 1.5rem;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🧠 核心策略
# ==========================================
def SMA(array, n):
    return pd.Series(array).rolling(window=n).mean()

class SniperStrategy(Strategy):
    ma_trend_period = 60
    ma_long_period = 240
    ma_base_exit = 20
    ma_fast_exit = 10
    vol_ma_period = 5
    big_candle_pct = 0.03
    min_volume_shares = 2000000
    lookback_window = 10
    use_year_line = True 
    
    def init(self):
        close = pd.Series(self.data.Close)
        volume = pd.Series(self.data.Volume)
        self.ma_trend = self.I(SMA, close, self.ma_trend_period)
        self.ma_base = self.I(SMA, close, self.ma_base_exit)
        self.ma_fast = self.I(SMA, close, self.ma_fast_exit)
        self.vol_ma = self.I(SMA, volume, self.vol_ma_period)
        if self.use_year_line:
            self.ma_long = self.I(SMA, close, self.ma_long_period)
        self.setup_active = False
        self.setup_bar_index = 0
        self.setup_low_price = 0

    def next(self):
        price = self.data.Close[-1]
        prev_high = self.data.High[-2]
        
        if self.position:
            if price < self.setup_low_price:
                self.position.close()
                return
            current_profit_pct = self.position.pl_pct
            exit_line = self.ma_fast[-1] if current_profit_pct > 0.15 else self.ma_base[-1]
            if price < exit_line:
                self.position.close()
            return

        triggered_buy = False
        days_since_setup = len(self.data) - self.setup_bar_index
        
        if self.setup_active:
            if days_since_setup > self.lookback_window:
                self.setup_active = False
            elif price < self.setup_low_price:
                self.setup_active = False
            elif price > prev_high:
                self.buy()
                self.setup_active = False 
                triggered_buy = True
                return 
        
        # Setup 檢查 (若未觸發買進)
        if not triggered_buy:
            if self.data.Volume[-1] < self.min_volume_shares: return
            
            # 趨勢
            is_trend_up = (price > self.ma_trend[-1]) and (self.ma_trend[-1] > self.ma_trend[-2])
            if self.use_year_line and (pd.isna(self.ma_long[-1]) or price < self.ma_long[-1]): return

            # 長紅
            prev_close = self.data.Close[-2]
            open_price = self.data.Open[-1]
            change_pct = (price - prev_close) / prev_close
            is_big = change_pct > self.big_candle_pct
            is_vol = self.data.Volume[-1] > self.vol_ma[-1]
            is_red = price > open_price

            if is_trend_up and is_big and is_vol and is_red:
                self.setup_active = True
                self.setup_bar_index = len(self.data)
                self.setup_low_price = self.data.Low[-1]

# ==========================================
# 🛠️ 輔助函式
# ==========================================
# 🔥 自定義細分產業資料庫
CUSTOM_SECTOR_MAP = {
    '2317': 'AI伺服器', '2382': 'AI伺服器', '3231': 'AI伺服器', '2356': 'AI伺服器', '6669': 'AI伺服器', '2376': 'AI伺服器',
    '3017': '散熱模組', '3324': '散熱模組', '2421': '散熱模組', '3653': '散熱模組',
    '1513': '重電綠能', '1519': '重電綠能', '1503': '重電綠能', '1504': '重電綠能', '1609': '重電綠能',
    '3661': 'IP/ASIC', '3443': 'IP/ASIC', '3035': 'IP/ASIC', '3529': 'IP/ASIC', '6531': 'IP/ASIC',
    '2603': '貨櫃航運', '2609': '貨櫃航運', '2615': '貨櫃航運',
    '2368': 'PCB/CCL', '3037': 'PCB/CCL', '6213': 'PCB/CCL', '6274': 'PCB/CCL',
    '2330': '半導體', '3711': '半導體封測'
}

def get_detailed_sector(code):
    """取得細分產業"""
    if code in CUSTOM_SECTOR_MAP: return CUSTOM_SECTOR_MAP[code]
    try:
        if code in twstock.codes: return twstock.codes[code].group
    except: pass
    return "其他"

@st.cache_data(ttl=3600)
def get_stock_info_map():
    """
    取得上市櫃股票資訊表
    回傳字典: {code: {'name': name, 'symbol': full_symbol}}
    🔥 優化：直接區分 .TW 與 .TWO，避免下載時嘗試錯誤
    """
    try:
        stock_map = {}
        # 上市 (.TW)
        for code, info in twstock.twse.items():
            if len(code) == 4:
                stock_map[code] = {'name': f"{code} {info.name}", 'symbol': f"{code}.TW"}
        # 上櫃 (.TWO)
        for code, info in twstock.tpex.items():
            if len(code) == 4:
                stock_map[code] = {'name': f"{code} {info.name}", 'symbol': f"{code}.TWO"}
        return stock_map
    except:
        return {}

def fetch_history_data(symbol, start_date=None, end_date=None, period="2y"):
    """
    下載數據 (使用 yf.Ticker 增強多執行緒隔離性)
    支援指定日期範圍
    """
    try:
        ticker = yf.Ticker(symbol)
        # 如果有指定日期範圍，優先使用
        if start_date and end_date:
            df = ticker.history(start=start_date, end=end_date)
        else:
            df = ticker.history(period=period)
        
        if df.empty: return None
        
        # 移除時區資訊，避免後續運算報錯
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        return df
    except:
        return None

def get_stock_data_with_realtime(code, symbol, analysis_date_str, start_date=None, end_date=None):
    """
    取得資料並補即時盤
    🔥 優化：直接接收 symbol，不再猜測 .TW/.TWO
    """
    # 若有指定日期範圍，使用日期範圍下載
    if start_date:
        df = fetch_history_data(symbol, start_date=start_date, end_date=end_date)
    else:
        df = fetch_history_data(symbol)
        
    if df is None or df.empty: return None
    
    last_dt = df.index[-1].strftime('%Y-%m-%d')
    today_str = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # 檢查是否需要補即時盤 (僅當分析日為今日且尚未有資料時)
    if analysis_date_str == today_str and last_dt != today_str:
        try:
            realtime = twstock.realtime.get(code)
            if realtime['success'] and realtime['realtime']['latest_trade_price'] != '-':
                rt = realtime['realtime']
                new_row = pd.Series({
                    'Open': float(rt['open']), 'High': float(rt['high']), 
                    'Low': float(rt['low']), 'Close': float(rt['latest_trade_price']), 
                    'Volume': float(rt['accumulate_trade_volume']) * 1000
                }, name=pd.Timestamp(today_str))
                df = pd.concat([df, new_row.to_frame().T])
        except:
            pass
    return df

def analyze_stock(code, stock_name, symbol, analysis_date_str, params):
    """多執行緒分析核心"""
    try:
        # 🔥 優化：減少延遲時間以加快速度，但保留微小隨機避免完全同步
        time.sleep(random.uniform(0.01, 0.05))
        
        df = get_stock_data_with_realtime(code, symbol, analysis_date_str)
        if df is None or len(df) < 250: return None
        
        # 解包參數
        ma_trend = params['ma_trend']
        use_year = params['use_year']
        big_candle = params['big_candle']
        min_vol = params['min_vol']
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        op = df['Open']
        
        # 指標計算
        ma_t = close.rolling(window=ma_trend).mean()
        ma_y = close.rolling(window=240).mean()
        vol_ma = volume.rolling(window=5).mean()
        
        # 定位日期
        df['DateStr'] = df.index.strftime('%Y-%m-%d')
        if analysis_date_str not in df['DateStr'].values: return None
        idx = df.index.get_loc(pd.Timestamp(analysis_date_str))
        
        # 基礎濾網
        if volume.iloc[idx] < min_vol: return None
        if use_year and close.iloc[idx] < ma_y.iloc[idx]: return None
        if not (close.iloc[idx] > ma_t.iloc[idx] and ma_t.iloc[idx] > ma_t.iloc[idx-1]): return None
        
        # 今日 Setup?
        is_setup = (
            (close.iloc[idx] - close.iloc[idx-1]) / close.iloc[idx-1] > big_candle and
            volume.iloc[idx] > vol_ma.iloc[idx] and
            close.iloc[idx] > op.iloc[idx]
        )
        
        # 回溯尋找 Setup
        setup_found = False
        s_low = 0
        s_high = 0 # 長紅高點
        s_date = ""
        setup_idx = -1
        
        for k in range(1, 11):
            b_idx = idx - k
            if b_idx < 0: break
            
            # Setup 條件
            if ((close.iloc[b_idx] - close.iloc[b_idx-1]) / close.iloc[b_idx-1] > big_candle and
                volume.iloc[b_idx] > vol_ma.iloc[b_idx] and
                close.iloc[b_idx] > op.iloc[b_idx]):
                
                # 破底檢查
                broken = False
                for m in range(b_idx+1, idx+1):
                    if close.iloc[m] < low.iloc[b_idx]:
                        broken = True
                        break
                if not broken:
                    setup_found = True
                    setup_idx = b_idx
                    s_low = low.iloc[b_idx]
                    s_high = high.iloc[b_idx]
                    s_date = df.index[b_idx].strftime('%Y-%m-%d')
                    break
        
        c_close = close.iloc[idx]
        if setup_found:
            yest_high = high.iloc[idx-1]
            if close.iloc[idx] > yest_high:
                # 強勢續漲 vs N字突破
                is_strong = False
                if idx == setup_idx + 1: is_strong = True
                else:
                    intermediate_lows = low.iloc[setup_idx+1 : idx]
                    if (intermediate_lows > s_high).all(): is_strong = True
                
                tag = "🚀 強勢續漲" if is_strong else "🎯 N字突破"
                return ("triggered", {"代號": code, "名稱": stock_name, "收盤": f"{c_close:.2f}", "狀態": tag, "訊號日": s_date, "突破價": f"{yest_high:.2f}"})
            else:
                # Watching 分類邏輯
                prev_c_today = close.iloc[idx-1]
                curr_pct = (c_close - prev_c_today) / prev_c_today
                
                status_watch = "👀 整理中"
                # 強勢整理: 股價在長紅K上方整理 漲跌幅<3% 且收盤不跌破長紅K高點
                if c_close >= s_high and abs(curr_pct) < 0.03:
                    status_watch = "💪 強勢整理"
                # 回檔整理: 股價在實體長紅K內 (小於高點) 且未跌破長紅K低點
                elif c_close < s_high and c_close >= s_low:
                    status_watch = "📉 回檔整理"

                return ("watching", {
                    "代號": code, "名稱": stock_name, "收盤": f"{c_close:.2f}", 
                    "狀態": status_watch, "訊號日": s_date, "防守": f"{s_low:.2f}", 
                    "長紅高": f"{s_high:.2f}", "漲跌幅": f"{curr_pct*100:.2f}%"
                })
        elif is_setup:
            # 計算漲幅
            prev_c = close.iloc[idx-1]
            pct_chg = (c_close - prev_c) / prev_c * 100
            stock_group = get_detailed_sector(code)
            return ("new_setup", {
                "代號": code, "名稱": stock_name, "收盤": f"{c_close:.2f}", 
                "狀態": "🔥 剛起漲", "漲幅": f"{pct_chg:+.2f}%", "族群": stock_group
            })
            
    except: return None
    return None

# 🔥 關鍵新增：全展開表格顯示函式
def display_full_table(df):
    """
    動態計算表格高度以顯示所有行 (取消內部捲動)
    因應 CSS 字體放大 (1.1rem)，調整行高計算參數
    """
    if df is not None and not df.empty:
        # 由於您的 CSS 將字體設為 1.1rem，原先的 35px 高度估算會太小導致捲軸出現
        # 這裡將每行高度估算加大至 45px
        # 總高度 = (資料行數 + 1 標題列) * 45px + 緩衝像素
        row_height = 45 
        height = (len(df) + 1) * row_height + 10
        
        st.dataframe(
            df, 
            hide_index=True, 
            use_container_width=True, 
            height=height 
        )
    else:
        st.info("無")

# ==========================================
# 🖥️ 側邊欄與主畫面
# ==========================================
st.sidebar.header("🛡️ 狙擊手策略參數")

analysis_date_input = st.sidebar.date_input("分析基準日", datetime.date.today())
analysis_date_str = analysis_date_input.strftime('%Y-%m-%d')

with st.sidebar.expander("進階參數設定", expanded=False):
    ma_trend = st.number_input("趨勢線 (MA)", value=60)
    use_year = st.checkbox("啟用年線 (240MA) 濾網", value=True)
    big_candle = st.slider("長紅漲幅門檻 (%)", 2.0, 10.0, 3.0, 0.5) / 100
    min_vol = st.number_input("最小成交量 (張)", value=1000) * 1000

params = {'ma_trend': ma_trend, 'use_year': use_year, 'big_candle': big_candle, 'min_vol': min_vol}

tab1, tab2 = st.tabs(["🚀 全台股掃描", "📊 個股 K 線診斷"])

with tab1:
    st.header("全台股強勢股掃描")
    col_mode, col_info = st.columns([1, 2])
    with col_mode:
        scan_scope = st.radio("掃描範圍", ["🔥 熱門股 (約50檔)", "🌏 全市場 (約1800檔)"])
    with col_info:
        st.info(f"📅 基準日: **{analysis_date_str}**")

    if st.button("開始掃描", type="primary"):
        # 🔥 修正：使用正確的函式名稱 get_stock_info_map
        stock_info_map = get_stock_info_map()
        
        if scan_scope.startswith("🔥"):
            scan_codes = ['2330', '2317', '2454', '2603', '1519', '3231', '2382', '3037', '2368', '3035', 
                         '3017', '3324', '1513', '6213', '8069', '3661', '6669', '9958', '6415', '6531',
                         '3532', '2376', '3529', '3443', '2609', '2615', '2002', '2881', '2882', '8038',
                         '2356', '2357', '4938', '4906', '5347', '6274', '2313', '2401', '2449', '3034']
        else:
            scan_codes = list(stock_info_map.keys())

        triggered, new_setup, watching = [], [], []
        
        status = st.empty()
        prog = st.progress(0)
        status.text("🚀 啟動多執行緒引擎 (Max: 20)...")
        
        total = len(scan_codes)
        done = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = {}
            for code in scan_codes:
                if code in stock_info_map:
                    info = stock_info_map[code]
                    futures[executor.submit(analyze_stock, code, info['name'], info['symbol'], analysis_date_str, params)] = code
            
            for future in concurrent.futures.as_completed(futures):
                done += 1
                if done % 10 == 0:
                    prog.progress(done / total)
                    status.text(f"掃描進度: {done}/{total}...")
                
                res = future.result()
                if res:
                    typ, data = res
                    if typ == "triggered": triggered.append(data)
                    elif typ == "new_setup": new_setup.append(data)
                    elif typ == "watching": watching.append(data)
        
        prog.progress(1.0)
        status.success(f"掃描完成！")
        
        # 分類處理 Triggered 名單 (強勢續漲 vs N字突破)
        trigger_strong = [x for x in triggered if "強勢續漲" in x['狀態']]
        trigger_n = [x for x in triggered if "N字突破" in x['狀態']]
        
        # 分類處理 Watching 名單
        watch_strong = [d for d in watching if "強勢整理" in d['狀態']]
        watch_pullback = [d for d in watching if "回檔整理" in d['狀態']]
        
        st.markdown("### 🎯 買點觸發訊號 (Actionable)")
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            st.subheader(f"🚀 強勢續漲 ({len(trigger_strong)})")
            display_full_table(pd.DataFrame(trigger_strong))
            
        with col_t2:
            st.subheader(f"🎯 N字突破 ({len(trigger_n)})")
            display_full_table(pd.DataFrame(trigger_n))
            
        st.divider()
        
        st.markdown("### 👀 市場潛力名單 (Monitoring)")
        
        # 剛起漲 (含族群統計)
        st.subheader(f"🔥 今日剛起漲 ({len(new_setup)})")
        st.caption("符合條件：季線之上第一根爆量實體長紅")
        if new_setup:
            df_new = pd.DataFrame(new_setup)
            # 統計族群分佈
            if "族群" in df_new.columns:
                sector_counts = df_new['族群'].value_counts().reset_index()
                sector_counts.columns = ['族群', '數量']
                top_sectors = [f"{row['族群']}({row['數量']})" for i, row in sector_counts.head(5).iterrows()]
                st.success("📊 熱門族群: " + " | ".join(top_sectors))
            display_full_table(df_new)
        else:
            st.info("無")
        
        st.write("") 

        # 觀察名單分類顯示
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            st.subheader(f"💪 強勢整理 ({len(watch_strong)})")
            st.caption("股價守在長紅高點之上")
            display_full_table(pd.DataFrame(watch_strong))
        
        with col_w2:
            st.subheader(f"📉 回檔整理 ({len(watch_pullback)})")
            st.caption("股價回跌至長紅實體內 (未破底)")
            display_full_table(pd.DataFrame(watch_pullback))

with tab2:
    st.header("📊 個股 K 線診斷 & 回測")
    st.caption("此功能可驗證策略在該股票過去一段時間的表現，確認買賣點邏輯。")
    
    col_in, col_date1, col_date2, col_b = st.columns([2, 2, 2, 1])
    with col_in: 
        stock_input = st.text_input("輸入代號", value="3231")
    
    default_start = datetime.date.today() - datetime.timedelta(days=365)
    default_end = datetime.date.today()
    
    with col_date1:
        start_date = st.date_input("開始日期", default_start)
    with col_date2:
        end_date = st.date_input("結束日期", default_end)
    
    if col_b.button("診斷"):
        try:
            # 取得正確 Symbol
            symbol_try = f"{stock_input}.TW"
            df = get_stock_data_with_realtime(stock_input, symbol_try, analysis_date_str)
            if df is None or df.empty:
                symbol_try = f"{stock_input}.TWO"
                df = get_stock_data_with_realtime(stock_input, symbol_try, analysis_date_str)

            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            download_start = (start_date - datetime.timedelta(days=400)).strftime('%Y-%m-%d')
            
            if df is not None:
                SniperStrategy.ma_trend_period = ma_trend
                SniperStrategy.use_year_line = use_year
                SniperStrategy.big_candle_pct = big_candle
                SniperStrategy.min_volume_shares = min_vol 
                
                bt = Backtest(df, SniperStrategy, cash=1_000_000, commission=0.004, trade_on_close=True)
                stats = bt.run()
                trades = stats['_trades']
                
                m1, m2, m3 = st.columns(3)
                m1.metric("區間報酬率", f"{stats['Return [%]']:.2f}%")
                m2.metric("勝率", f"{stats['Win Rate [%]']:.2f}%")
                m3.metric("交易次數", f"{stats['# Trades']}")
                
                df['MA_Trend'] = df['Close'].rolling(window=ma_trend).mean()
                df['MA_Year'] = df['Close'].rolling(window=240).mean()
                df['MA_Base'] = df['Close'].rolling(window=20).mean()
                
                plot_df = df[df.index >= pd.Timestamp(start_str)].copy()
                
                if plot_df.empty:
                    st.warning("選定區間無資料")
                else:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name='K線', increasing_line_color='red', decreasing_line_color='green'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA_Trend'], line=dict(color='blue'), name=f'{ma_trend}MA'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA_Base'], line=dict(color='orange', width=1.5), name='20MA (Base)'), row=1, col=1)
                    if use_year: fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA_Year'], line=dict(color='purple', dash='dash'), name='240MA'), row=1, col=1)
                    
                    if len(trades) > 0:
                        buy_dates = [t for t in trades['EntryTime'] if t in plot_df.index]
                        buy_prices = [plot_df.loc[t]['Low']*0.96 for t in buy_dates]
                        sell_dates = [t for t in trades['ExitTime'] if t in plot_df.index]
                        sell_prices = [plot_df.loc[t]['High']*1.04 for t in sell_dates]
                        fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', marker=dict(symbol='triangle-up', size=15, color='red'), name='買進'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers', marker=dict(symbol='triangle-down', size=15, color='green'), name='賣出'), row=1, col=1)

                    colors = ['red' if r['Close'] >= r['Open'] else 'green' for i, r in plot_df.iterrows()]
                    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], marker_color=colors, name='成交量'), row=2, col=1)
                    
                    dt_all = pd.date_range(start=plot_df.index[0], end=plot_df.index[-1])
                    dt_obs = [d.strftime("%Y-%m-%d") for d in plot_df.index]
                    dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if d not in dt_obs]
                    fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
                    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template='plotly_white', hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("查無資料")
        except Exception as e:
            st.error(f"發生錯誤：{e}")