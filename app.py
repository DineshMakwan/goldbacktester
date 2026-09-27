import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, time

# Page Config
st.set_page_config(page_title="Pro XAUUSD Multi-Indicator Backtester", layout="wide", page_icon="📈")

# Title & Description
st.title("🥇 Pro XAUUSD & Multi-Asset Backtester")
st.caption("Test multi-indicator confluence strategies with custom risk, session time filters, and detailed trade execution logs.")

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("1. Asset & Timeframe Setup")
ticker = st.sidebar.text_input("Ticker Symbol", value="XAUUSD=X", help="Examples: XAUUSD=X (Gold), GC=F (Gold Futures), RELIANCE.NS, EURUSD=X")
period = st.sidebar.selectbox("Data Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
interval = st.sidebar.selectbox("Timeframe", ["5m", "15m", "30m", "60m", "1d"], index=3)

st.sidebar.header("2. Trading Time Window (UTC/IST Filter)")
enable_time_filter = st.sidebar.checkbox("Enable Time Filter (e.g. London/NY Session)", value=False)
start_time_input = st.sidebar.time_input("Session Start Time", time(13, 0))
end_time_input = st.sidebar.time_input("Session End Time", time(21, 0))

st.sidebar.header("3. Risk & Capital Settings")
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=1000.0, step=100.0)
lot_size = st.sidebar.number_input("Lot Size (1 Lot = $100 per $1 move)", value=0.10, step=0.01)
sl_dollars = st.sidebar.number_input("Stop Loss ($ Price Offset)", value=5.0, step=0.5, help="Example: $5.0 SL means if entry is $2000, SL is $1995 for BUY")
tp_dollars = st.sidebar.number_input("Take Profit ($ Price Offset)", value=10.0, step=0.5, help="Example: $10.0 TP means if entry is $2000, TP is $2010 for BUY")

st.sidebar.header("4. Indicator Confluence Selection (Min 1 Enabled)")
# 1. EMA Crossover
use_ema = st.sidebar.checkbox("1. EMA Crossover", value=True)
fast_ema_p = st.sidebar.number_input("Fast EMA Period", value=9, min_value=1)
slow_ema_p = st.sidebar.number_input("Slow EMA Period", value=21, min_value=1)

# 2. SMA Crossover
use_sma = st.sidebar.checkbox("2. SMA Crossover", value=False)
fast_sma_p = st.sidebar.number_input("Fast SMA Period", value=20, min_value=1)
slow_sma_p = st.sidebar.number_input("Slow SMA Period", value=50, min_value=1)

# 3. RSI
use_rsi = st.sidebar.checkbox("3. RSI Filter", value=True)
rsi_period = st.sidebar.number_input("RSI Period", value=14, min_value=1)
rsi_oversold = st.sidebar.slider("RSI Buy Level (Below)", 10, 50, 45)
rsi_overbought = st.sidebar.slider("RSI Sell Level (Above)", 50, 90, 55)

# 4. MACD
use_macd = st.sidebar.checkbox("4. MACD Crossover", value=False)
macd_fast = st.sidebar.number_input("MACD Fast", value=12)
macd_slow = st.sidebar.number_input("MACD Slow", value=26)
macd_sig = st.sidebar.number_input("MACD Signal", value=9)

# 5. Supertrend
use_st = st.sidebar.checkbox("5. Supertrend Filter", value=False)
st_period = st.sidebar.number_input("Supertrend ATR Period", value=10)
st_mult = st.sidebar.number_input("Supertrend Multiplier", value=3.0, step=0.1)

# 6. Bollinger Bands
use_bb = st.sidebar.checkbox("6. Bollinger Bands", value=False)
bb_period = st.sidebar.number_input("BB Period", value=20)
bb_std = st.sidebar.number_input("BB Std Dev", value=2.0)

# 7. Stochastic Oscillator
use_stoch = st.sidebar.checkbox("7. Stochastic Oscillator", value=False)
stoch_k = st.sidebar.number_input("Stoch %K Period", value=14)
stoch_d = st.sidebar.number_input("Stoch %D Period", value=3)

# 8. ATR Volatility Filter
use_atr = st.sidebar.checkbox("8. ATR Min Volatility Filter", value=False)
atr_period = st.sidebar.number_input("ATR Period", value=14)
min_atr_val = st.sidebar.number_input("Min ATR Threshold ($)", value=1.0, step=0.1)

# 9. ADX Trend Filter
use_adx = st.sidebar.checkbox("9. ADX Trend Strength Filter", value=False)
adx_period = st.sidebar.number_input("ADX Period", value=14)
min_adx_val = st.sidebar.number_input("Min ADX Value", value=25.0)

# 10. Simple Price Action Breakout
use_breakout = st.sidebar.checkbox("10. High/Low Donchian Breakout", value=False)
breakout_period = st.sidebar.number_input("Breakout Lookback Candles", value=20)

run_button = st.sidebar.button("⚡ Run Backtest", use_container_width=True)

# ==========================================
# INDICATOR CALCULATION FUNCTIONS
# ==========================================
def calculate_indicators(df):
    df = df.copy()
    
    # 1. EMAs
    df['EMA_Fast'] = df['Close'].ewm(span=fast_ema_p, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=slow_ema_p, adjust=False).mean()
    
    # 2. SMAs
    df['SMA_Fast'] = df['Close'].rolling(window=fast_sma_p).mean()
    df['SMA_Slow'] = df['Close'].rolling(window=slow_sma_p).mean()
    
    # 3. RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 4. MACD
    ema_f = df['Close'].ewm(span=macd_fast, adjust=False).mean()
    ema_s = df['Close'].ewm(span=macd_slow, adjust=False).mean()
    df['MACD'] = ema_f - ema_s
    df['MACD_Signal'] = df['MACD'].ewm(span=macd_sig, adjust=False).mean()
    
    # 5. ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=atr_period).mean()
    
    # 6. Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(window=bb_period).mean()
    df['BB_Std'] = df['Close'].rolling(window=bb_period).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * bb_std)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * bb_std)
    
    # 7. Stochastic
    low_min = df['Low'].rolling(window=stoch_k).min()
    high_max = df['High'].rolling(window=stoch_k).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min + 1e-10))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=stoch_d).mean()
    
    # 8. ADX
    up_move = df['High'].diff()
    down_move = df['Low'].diff().abs()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(adx_period).mean() / (df['ATR'] + 1e-10))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(adx_period).mean() / (df['ATR'] + 1e-10))
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
    df['ADX'] = dx.rolling(adx_period).mean()
    
    # 9. Supertrend
    hl2 = (df['High'] + df['Low']) / 2
    basic_upper = hl2 + (st_mult * df['ATR'])
    basic_lower = hl2 - (st_mult * df['ATR'])
    
    st_dir = np.ones(len(df))
    st_val = np.zeros(len(df))
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > basic_upper.iloc[i-1]:
            st_dir[i] = 1
        elif df['Close'].iloc[i] < basic_lower.iloc[i-1]:
            st_dir[i] = -1
        else:
            st_dir[i] = st_dir[i-1]
            
        if st_dir[i] == 1:
            st_val[i] = basic_lower.iloc[i]
        else:
            st_val[i] = basic_upper.iloc[i]
            
    df['Supertrend_Dir'] = st_dir
    df['Supertrend'] = st_val
    
    # 10. Donchian Breakout
    df['Donchian_High'] = df['High'].shift(1).rolling(breakout_period).max()
    df['Donchian_Low'] = df['Low'].shift(1).rolling(breakout_period).min()
    
    return df

# ==========================================
# MAIN BACKTEST LOGIC
# ==========================================
if run_button:
    with st.spinner("Fetching Historical Data from Yahoo Finance..."):
        df = yf.download(ticker, period=period, interval=interval)
        
    if df.empty:
        st.error("❌ No data returned. Check ticker symbol or timeframe compatibility!")
    else:
        # Clean multi-index columns if returned
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = calculate_indicators(df)
        
        # Determine Signals
        df['Buy_Condition'] = True
        df['Sell_Condition'] = True
        
        # Confluence Checks
        if use_ema:
            df['Buy_Condition'] &= (df['EMA_Fast'] > df['EMA_Slow'])
            df['Sell_Condition'] &= (df['EMA_Fast'] < df['EMA_Slow'])
            
        if use_sma:
            df['Buy_Condition'] &= (df['SMA_Fast'] > df['SMA_Slow'])
            df['Sell_Condition'] &= (df['SMA_Fast'] < df['SMA_Slow'])
            
        if use_rsi:
            df['Buy_Condition'] &= (df['RSI'] < rsi_oversold)
            df['Sell_Condition'] &= (df['RSI'] > rsi_overbought)
            
        if use_macd:
            df['Buy_Condition'] &= (df['MACD'] > df['MACD_Signal'])
            df['Sell_Condition'] &= (df['MACD'] < df['MACD_Signal'])
            
        if use_st:
            df['Buy_Condition'] &= (df['Supertrend_Dir'] == 1)
            df['Sell_Condition'] &= (df['Supertrend_Dir'] == -1)
            
        if use_bb:
            df['Buy_Condition'] &= (df['Close'] <= df['BB_Lower'])
            df['Sell_Condition'] &= (df['Close'] >= df['BB_Upper'])
            
        if use_stoch:
            df['Buy_Condition'] &= (df['Stoch_K'] > df['Stoch_D']) & (df['Stoch_K'] < 30)
            df['Sell_Condition'] &= (df['Stoch_K'] < df['Stoch_D']) & (df['Stoch_K'] > 70)
            
        if use_atr:
            df['Buy_Condition'] &= (df['ATR'] >= min_atr_val)
            df['Sell_Condition'] &= (df['ATR'] >= min_atr_val)
            
        if use_adx:
            df['Buy_Condition'] &= (df['ADX'] >= min_adx_val)
            df['Sell_Condition'] &= (df['ADX'] >= min_adx_val)
            
        if use_breakout:
            df['Buy_Condition'] &= (df['Close'] > df['Donchian_High'])
            df['Sell_Condition'] &= (df['Close'] < df['Donchian_Low'])
            
        if enable_time_filter:
            time_mask = (df.index.time >= start_time_input) & (df.index.time <= end_time_input)
            df['Buy_Condition'] &= time_mask
            df['Sell_Condition'] &= time_mask

        # Trade Simulator Loop
        trades = []
        in_position = False
        pos_type = None
        entry_price = 0.0
        sl_price = 0.0
        tp_price = 0.0
        entry_time = None
        
        balance = initial_capital
        balance_history = []
        usd_per_dollar_move = lot_size * 100.0  # 1 lot = $100 per $1 move in Gold

        for i in range(len(df)):
            current_time = df.index[i]
            current_close = df['Close'].iloc[i]
            current_high = df['High'].iloc[i]
            current_low = df['Low'].iloc[i]
            
            # Check Position Exit
            if in_position:
                if pos_type == 'BUY':
                    if current_low <= sl_price:
                        pnl = (sl_price - entry_price) * usd_per_dollar_move
                        balance += pnl
                        trades.append({'ID': len(trades)+1, 'Type': 'BUY', 'Entry_Time': entry_time, 'Exit_Time': current_time, 'Entry': entry_price, 'Exit': sl_price, 'Result': 'SL HIT', 'PnL_$': pnl, 'Balance': balance})
                        in_position = False
                    elif current_high >= tp_price:
                        pnl = (tp_price - entry_price) * usd_per_dollar_move
                        balance += pnl
                        trades.append({'ID': len(trades)+1, 'Type': 'BUY', 'Entry_Time': entry_time, 'Exit_Time': current_time, 'Entry': entry_price, 'Exit': tp_price, 'Result': 'TP HIT', 'PnL_$': pnl, 'Balance': balance})
                        in_position = False
                        
                elif pos_type == 'SELL':
                    if current_high >= sl_price:
                        pnl = (entry_price - sl_price) * usd_per_dollar_move
                        balance += pnl
                        trades.append({'ID': len(trades)+1, 'Type': 'SELL', 'Entry_Time': entry_time, 'Exit_Time': current_time, 'Entry': entry_price, 'Exit': sl_price, 'Result': 'SL HIT', 'PnL_$': pnl, 'Balance': balance})
                        in_position = False
                    elif current_low <= tp_price:
                        pnl = (entry_price - tp_price) * usd_per_dollar_move
                        balance += pnl
                        trades.append({'ID': len(trades)+1, 'Type': 'SELL', 'Entry_Time': entry_time, 'Exit_Time': current_time, 'Entry': entry_price, 'Exit': tp_price, 'Result': 'TP HIT', 'PnL_$': pnl, 'Balance': balance})
                        in_position = False

            # Check Position Entry
            if not in_position:
                if df['Buy_Condition'].iloc[i]:
                    in_position = True
                    pos_type = 'BUY'
                    entry_price = current_close
                    sl_price = entry_price - sl_dollars
                    tp_price = entry_price + tp_dollars
                    entry_time = current_time
                elif df['Sell_Condition'].iloc[i]:
                    in_position = True
                    pos_type = 'SELL'
                    entry_price = current_close
                    sl_price = entry_price + sl_dollars
                    tp_price = entry_price - tp_dollars
                    entry_time = current_time

            balance_history.append(balance)

        df['Portfolio_Balance'] = balance_history
        trades_df = pd.DataFrame(trades)

        # ==========================================
        # DASHBOARD METRICS DISPLAY
        # ==========================================
        total_pnl = balance - initial_capital
        roi = (total_pnl / initial_capital) * 100
        total_trades = len(trades_df)
        
        if total_trades > 0:
            wins = len(trades_df[trades_df['PnL_$'] > 0])
            losses = len(trades_df[trades_df['PnL_$'] <= 0])
            win_rate = (wins / total_trades) * 100
        else:
            wins, losses, win_rate = 0, 0, 0.0

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Final Balance", f"${balance:,.2f}", f"${total_pnl:+,.2f}")
        col2.metric("Total ROI %", f"{roi:.2f}%")
        col3.metric("Win Rate %", f"{win_rate:.1f}%")
        col4.metric("Total Trades", f"{total_trades}")
        col5.metric("Wins / Losses", f"{wins} W / {losses} L")

        # ==========================================
        # PLOT INTERACTIVE CHARTS
        # ==========================================
        st.subheader("📈 Interactive Price & Trades Chart")
        fig = go.Figure()
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'
        ))
        
        # Plot Indicators if enabled
        if use_ema:
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Fast'], mode='lines', name=f'Fast EMA ({fast_ema_p})', line=dict(color='cyan', width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Slow'], mode='lines', name=f'Slow EMA ({slow_ema_p})', line=dict(color='magenta', width=1)))

        # Plot Buy/Sell Trade Signals on Chart
        if not trades_df.empty:
            buy_trades = trades_df[trades_df['Type'] == 'BUY']
            sell_trades = trades_df[trades_df['Type'] == 'SELL']
            
            fig.add_trace(go.Scatter(
                x=buy_trades['Entry_Time'], y=buy_trades['Entry'], mode='markers', name='BUY Entry',
                marker=dict(symbol='triangle-up', size=12, color='green')
            ))
            fig.add_trace(go.Scatter(
                x=sell_trades['Entry_Time'], y=sell_trades['Entry'], mode='markers', name='SELL Entry',
                marker=dict(symbol='triangle-down', size=12, color='red')
            ))

        fig.update_layout(height=550, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # Equity Growth Line Graph
        st.subheader("📊 Equity Growth Curve (PnL)")
        equity_color = "green" if total_pnl >= 0 else "red"
        
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            x=df.index, y=df['Portfolio_Balance'], mode='lines', name='Account Balance ($)',
            line=dict(color=equity_color, width=2)
        ))
        fig_equity.update_layout(height=350, template="plotly_dark", yaxis_title="Balance ($)")
        st.plotly_chart(fig_equity, use_container_width=True)

        # ==========================================
        # TRADE EXPORT & LOG TABLE
        # ==========================================
        st.subheader("📋 Trade Execution Log Table")
        if not trades_df.empty:
            st.dataframe(trades_df, use_container_width=True)
            
            # Download CSV Button
            csv_data = trades_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Trade History (CSV)",
                data=csv_data,
                file_name=f"Backtest_{ticker}_{interval}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No trades were triggered with the selected indicator rules.")
else:
    st.info("👈 Sidebar se apne parameters adjust karein aur **⚡ Run Backtest** button par click karein!")