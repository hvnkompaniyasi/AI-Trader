import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import datetime
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def fetch_eurusd_m5_jan_to_jul_2026():
    """Fetch M5 historical rates for EUR/USD from Jan 1, 2026 to Jul 24, 2026."""
    if not mt5.initialize():
        return None, None

    candidates = ["EURUSD", "EURUSD#", "EUR.m", "EURUSD.m", "EURUSD_i"]
    target_sym = None
    for cand in candidates:
        info = mt5.symbol_info(cand)
        if info is not None:
            mt5.symbol_select(cand, True)
            target_sym = cand
            break

    if not target_sym:
        return None, None

    utc_from = datetime.datetime(2026, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    utc_to = datetime.datetime(2026, 7, 24, 23, 59, 59, tzinfo=datetime.timezone.utc)

    rates = mt5.copy_rates_range(target_sym, mt5.TIMEFRAME_M5, utc_from, utc_to)
    if rates is None or len(rates) == 0:
        rates = mt5.copy_rates_from_pos(target_sym, mt5.TIMEFRAME_M5, 0, 30000)

    if rates is None or len(rates) == 0:
        return None, target_sym

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'})
    
    df = df[(df['time'] >= '2026-01-01') & (df['time'] <= '2026-07-24 23:59:59')].copy()
    df = df.reset_index(drop=True)
    return df, target_sym

def compute_dynamic_avwap_at_swings(df, window=40):
    """
    Computes Dynamic Anchored VWAP (AVWAP) anchored to major Swing Highs and Swing Lows.
    Identifies significant market structure shifts to re-anchor AVWAP.
    """
    df = df.copy()
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    volumes = df['Volume'].values
    n = len(df)

    avwap = np.full(n, np.nan)
    current_anchor = 0

    for i in range(1, n):
        if i >= window:
            local_max_idx = i - window + np.argmax(highs[i-window:i])
            local_min_idx = i - window + np.argmin(lows[i-window:i])

            if highs[local_max_idx] > highs[current_anchor] or lows[local_min_idx] < lows[current_anchor]:
                current_anchor = max(local_max_idx, local_min_idx)

        sub_closes = closes[current_anchor:i+1]
        sub_highs = highs[current_anchor:i+1]
        sub_lows = lows[current_anchor:i+1]
        sub_vols = volumes[current_anchor:i+1]

        typical_prices = (sub_closes + sub_highs + sub_lows) / 3.0
        cum_vp = np.sum(typical_prices * sub_vols)
        cum_vol = np.sum(sub_vols)

        avwap[i] = cum_vp / (cum_vol + 1e-10)

    df['AVWAP_dynamic'] = avwap
    return df

def compute_volume_profile_and_fvgs(df, num_bins=25):
    """
    Computes:
    1. Daily Session Volume Profile (POC, VAH, VAL, Heavy Volume Zone).
    2. Fair Value Gaps (FVG).
    """
    df = df.copy()
    df['date'] = df['time'].dt.date

    poc_list = [np.nan] * len(df)
    vah_list = [np.nan] * len(df)
    val_list = [np.nan] * len(df)
    heavy_zone_min = [np.nan] * len(df)
    heavy_zone_max = [np.nan] * len(df)

    grouped = df.groupby('date')

    for date, group in grouped:
        indices = group.index
        highs = group['High'].values
        lows = group['Low'].values
        volumes = group['Volume'].values
        n = len(group)

        for k in range(1, n):
            global_idx = indices[k]
            sub_highs = highs[:k]
            sub_lows = lows[:k]
            sub_vols = volumes[:k]

            price_min = np.min(sub_lows)
            price_max = np.max(sub_highs)

            if price_max <= price_min:
                continue

            bins = np.linspace(price_min, price_max, num_bins + 1)
            bin_volumes = np.zeros(num_bins)

            for j in range(k):
                price_mid = (sub_highs[j] + sub_lows[j]) / 2.0
                bin_idx = int((price_mid - price_min) / (price_max - price_min) * (num_bins - 1))
                bin_idx = max(0, min(num_bins - 1, bin_idx))
                bin_volumes[bin_idx] += sub_vols[j]

            poc_idx = np.argmax(bin_volumes)
            poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2.0

            total_vol = np.sum(bin_volumes)
            target_vol = total_vol * 0.70

            sorted_indices = np.argsort(bin_volumes)[::-1]
            accum_vol = 0.0
            va_bins = []
            for idx in sorted_indices:
                va_bins.append(idx)
                accum_vol += bin_volumes[idx]
                if accum_vol >= target_vol:
                    break

            va_min_idx = np.min(va_bins)
            va_max_idx = np.max(va_bins)

            val_price = bins[va_min_idx]
            vah_price = bins[va_max_idx + 1]

            top_bins_count = max(1, int(len(va_bins) * 0.3))
            top_volume_bins = va_bins[:top_bins_count]
            hz_min = bins[np.min(top_volume_bins)]
            hz_max = bins[np.max(top_volume_bins) + 1]

            poc_list[global_idx] = poc_price
            vah_list[global_idx] = vah_price
            val_list[global_idx] = val_price
            heavy_zone_min[global_idx] = hz_min
            heavy_zone_max[global_idx] = hz_max

    df['POC'] = poc_list
    df['VAH'] = vah_list
    df['VAL'] = val_list
    df['HZ_Min'] = heavy_zone_min
    df['HZ_Max'] = heavy_zone_max

    # Fair Value Gaps (FVG)
    bullish_fvg = (df['Low'] > df['High'].shift(2))
    bearish_fvg = (df['High'] < df['Low'].shift(2))

    df['Bullish_FVG'] = bullish_fvg
    df['Bearish_FVG'] = bearish_fvg

    return df

def compute_indicators(df):
    """Compute ATR(14) and Relative Volume strictly shifted by 1 bar to eliminate look-ahead bias."""
    df = df.copy()
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Rel_Vol'] = df['Volume'] / (df['Vol_MA20'] + 1e-10)

    df['ATR_sig'] = df['ATR'].shift(1)
    df['Rel_Vol_sig'] = df['Rel_Vol'].shift(1)
    df['POC_sig'] = df['POC'].shift(1)
    df['VAH_sig'] = df['VAH'].shift(1)
    df['VAL_sig'] = df['VAL'].shift(1)
    df['HZ_Min_sig'] = df['HZ_Min'].shift(1)
    df['HZ_Max_sig'] = df['HZ_Max'].shift(1)
    df['AVWAP_sig'] = df['AVWAP_dynamic'].shift(1)
    df['Close_sig'] = df['Close'].shift(1)
    df['Open_sig'] = df['Open'].shift(1)
    df['Bullish_FVG_sig'] = df['Bullish_FVG'].shift(1)
    df['Bearish_FVG_sig'] = df['Bearish_FVG'].shift(1)

    return df

def run_two_tp_trader_dale_backtest(df, initial_cash=50.0, lot_size=0.01, spread=0.00015, cooldown_bars=35):
    """
    TRADER DALE TWO TAKE-PROFITS (TP1 & TP2) & SMC VOLUME SL ENGINE ($50 CAPITAL):
    - Target: EUR/USD M5 ($50 Starting Capital, 0.01 Fixed Lot)
    - Trend Filter: AVWAP_dynamic solely (Close > AVWAP for BUY, Close < AVWAP for SELL)
    - Mandatory Imbalance Confirmation: Heavy Volume Zone / POC + (FVG AND Rel_Vol_sig >= 1.8)
    - Two Take-Profits (50% / 50% Partial Closure):
      * TP1 = 1.0x ATR. When hit, closes 50% position and moves SL to Break-Even (entry price).
      * TP2 = 2.5x ATR for remaining 50% position.
    - SMC Volume-Cluster Stop-Loss:
      * BUY SL = min(entry_price - 1.5 * ATR, HZ_Min - 0.5 * ATR)
      * SELL SL = max(entry_price + 1.5 * ATR, HZ_Max + 0.5 * ATR)
    """
    cash = float(initial_cash)
    equity_curve = [cash]
    equity_times = [df['time'].iloc[0]]

    position = None
    trades = []
    total_units = lot_size * 100000  # 1,000 units for 0.01 lot EUR/USD
    last_exit_idx = - cooldown_bars - 1

    for i in range(50, len(df)):
        row = df.iloc[i]
        t = row['time']
        close_p = row['Close']
        high_p = row['High']
        low_p = row['Low']
        hour_utc = t.hour

        atr_sig = row['ATR_sig'] if pd.notna(row['ATR_sig']) else 0.0008
        rel_vol_sig = row['Rel_Vol_sig'] if pd.notna(row['Rel_Vol_sig']) else 1.0
        poc_sig = row['POC_sig']
        val_sig = row['VAL_sig']
        vah_sig = row['VAH_sig']
        hz_min_sig = row['HZ_Min_sig']
        hz_max_sig = row['HZ_Max_sig']
        avwap_sig = row['AVWAP_sig']
        close_sig = row['Close_sig']
        open_sig = row['Open_sig']
        bull_fvg_sig = row['Bullish_FVG_sig']
        bear_fvg_sig = row['Bearish_FVG_sig']

        # Position Management: Two TPs Logic
        if position is not None:
            pos_type = position['type']
            entry_p = position['entry_price']
            rem_units = position['remaining_units']

            # Check TP1 Hit (First 50% Leg)
            if not position['tp1_hit']:
                if pos_type == "BUY" and high_p >= position['tp1_price']:
                    leg1_units = total_units * 0.5
                    pnl_leg1 = leg1_units * (position['tp1_price'] - entry_p)
                    cash += pnl_leg1
                    position['remaining_units'] -= leg1_units
                    position['tp1_hit'] = True
                    position['sl_price'] = entry_p  # Shift SL to Break-Even immediately
                    trades.append({
                        'symbol': 'EURUSD', 'entry_time': position['entry_time'], 'exit_time': t,
                        'type': 'BUY_TP1', 'entry_price': entry_p, 'exit_price': position['tp1_price'],
                        'pnl': pnl_leg1, 'win': True
                    })
                elif pos_type == "SELL" and low_p <= position['tp1_price']:
                    leg1_units = total_units * 0.5
                    pnl_leg1 = leg1_units * (entry_p - position['tp1_price'])
                    cash += pnl_leg1
                    position['remaining_units'] -= leg1_units
                    position['tp1_hit'] = True
                    position['sl_price'] = entry_p  # Shift SL to Break-Even immediately
                    trades.append({
                        'symbol': 'EURUSD', 'entry_time': position['entry_time'], 'exit_time': t,
                        'type': 'SELL_TP1', 'entry_price': entry_p, 'exit_price': position['tp1_price'],
                        'pnl': pnl_leg1, 'win': True
                    })

            # Check Final Exit (TP2 or SL) for remaining position
            rem_units = position['remaining_units']
            if pos_type == "BUY":
                curr_equity = cash + rem_units * (close_p - entry_p)
                hit_tp2 = high_p >= position['tp2_price']
                hit_sl = low_p <= position['sl_price']
            else:
                curr_equity = cash + rem_units * (entry_p - close_p)
                hit_tp2 = low_p <= position['tp2_price']
                hit_sl = high_p >= position['sl_price']

            if curr_equity <= 0:
                cash = 0.0
                trades.append({'symbol': 'EURUSD', 'pnl': -position['margin_used'], 'win': False, 'exit_time': t})
                position = None
                last_exit_idx = i
                equity_curve.append(0.0)
                equity_times.append(t)
                break

            equity_curve.append(curr_equity)
            equity_times.append(t)

            if hit_tp2 or hit_sl:
                exit_price = position['tp2_price'] if hit_tp2 else position['sl_price']

                if pos_type == "BUY":
                    pnl_leg2 = rem_units * (exit_price - entry_p)
                else:
                    pnl_leg2 = rem_units * (entry_p - exit_price)

                cash += pnl_leg2
                trades.append({
                    'symbol': 'EURUSD',
                    'entry_time': position['entry_time'],
                    'exit_time': t,
                    'type': f"{pos_type}_LEG2",
                    'entry_price': entry_p,
                    'exit_price': exit_price,
                    'pnl': pnl_leg2,
                    'win': pnl_leg2 > 0
                })
                position = None
                last_exit_idx = i
        else:
            curr_equity = cash
            equity_curve.append(curr_equity)
            equity_times.append(t)

            if (i - last_exit_idx) <= cooldown_bars:
                continue

            # Session Liquidity Window (08:00 to 18:00 UTC)
            session_valid = 8 <= hour_utc <= 18

            if session_valid and pd.notna(poc_sig) and pd.notna(val_sig) and pd.notna(vah_sig) and pd.notna(avwap_sig):
                # 1. AVWAP TREND FILTER
                uptrend = close_sig > avwap_sig
                downtrend = close_sig < avwap_sig

                # 2. HEAVY VOLUME ZONE / POC REACTION
                in_heavy_zone = pd.notna(hz_min_sig) and (hz_min_sig <= close_sig <= hz_max_sig)
                near_poc_val = (abs(close_sig - poc_sig) <= (atr_sig * 0.5)) or (abs(close_sig - val_sig) <= (atr_sig * 0.5))
                near_poc_vah = (abs(close_sig - poc_sig) <= (atr_sig * 0.5)) or (abs(close_sig - vah_sig) <= (atr_sig * 0.5))

                # 3. MANDATORY IMBALANCE CONFIRMATION: FVG AND VOLUME SURGE >= 1.8
                buy_confirmed = bull_fvg_sig and (rel_vol_sig >= 2.0)  # Strict volume threshold
                sell_confirmed = bear_fvg_sig and (rel_vol_sig >= 2.0)  # Strict volume threshold

                bullish_candle = close_sig > open_sig
                bearish_candle = close_sig < open_sig

                buy_signal = uptrend and bullish_candle and (in_heavy_zone or near_poc_val) and buy_confirmed
                sell_signal = downtrend and bearish_candle and (in_heavy_zone or near_poc_vah) and sell_confirmed

                if buy_signal or sell_signal:
                    pos_type = "BUY" if buy_signal else "SELL"
                    entry_price = (close_p + spread/2.0) if buy_signal else (close_p - spread/2.0)

                    # SMC Volume Zone Stop Loss
                    if pos_type == "BUY":
                         sl_price = min(entry_price - 1.2 * atr_sig, (hz_min_sig - 0.5 * atr_sig) if pd.notna(hz_min_sig) else entry_price - 1.2 * atr_sig)
                         tp1_price = entry_price + 1.3 * atr_sig
                         tp2_price = entry_price + 3.0 * atr_sig
                    else:
                         sl_price = max(entry_price + 1.2 * atr_sig, (hz_max_sig + 0.5 * atr_sig) if pd.notna(hz_max_sig) else entry_price + 1.2 * atr_sig)
                         tp1_price = entry_price - 1.3 * atr_sig
                         tp2_price = entry_price - 3.0 * atr_sig

                    position = {
                        'type': pos_type,
                        'entry_time': t,
                        'entry_price': entry_price,
                        'sl_price': sl_price,
                        'tp1_price': tp1_price,
                        'tp2_price': tp2_price,
                        'remaining_units': total_units,
                        'tp1_hit': False,
                        'atr_val': atr_sig,
                        'margin_used': (total_units * entry_price) / 400.0
                    }

    # Calculate Statistics
    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)

    if total_trades > 0:
        wins = trades_df[trades_df['win']]
        losses = trades_df[~trades_df['win']]
        num_wins = len(wins)
        num_losses = len(losses)
        win_rate = (num_wins / total_trades) * 100.0

        gross_profit = wins['pnl'].sum() if num_wins > 0 else 0.0
        gross_loss = abs(losses['pnl'].sum()) if num_losses > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else gross_profit
    else:
        num_wins = 0
        num_losses = 0
        win_rate = 0.0
        profit_factor = 0.0

    eq_series = pd.Series(equity_curve)
    peak = eq_series.cummax()
    drawdown = (eq_series - peak) / (peak + 1e-10)
    max_drawdown_pct = abs(drawdown.min()) * 100.0

    final_equity = equity_curve[-1]
    return_pct = ((final_equity - initial_cash) / initial_cash) * 100.0

    return {
        'initial_cash': initial_cash,
        'final_equity': final_equity,
        'return_pct': return_pct,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'num_wins': num_wins,
        'num_losses': num_losses,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_drawdown_pct,
        'trades_df': trades_df,
        'equity_curve': equity_curve,
        'equity_times': equity_times
    }

def main():
    mt5.initialize(login=352418061, password='Sardor_2007', server='XMGlobal-MT5 11')

    print("==================================================")
    print("  EUR/USD M5 TWO TAKE-PROFITS (TP1/TP2) MODEL   ")
    print("==================================================")

    df, resolved_sym = fetch_eurusd_m5_jan_to_jul_2026()

    if df is None or len(df) == 0:
        print("[-] Could not fetch EUR/USD M5 data for Jan-Jul 2026.")
        mt5.shutdown()
        return

    print(f"[+] Data Source: MetaTrader 5 ({resolved_sym})")
    print(f"[+] Total M5 Candles Loaded: {len(df)}")
    print(f"[+] Date Range: {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")

    print("[*] Computing Volume Profile, FVG, & Dynamic AVWAP at Swing Points...")
    df = compute_dynamic_avwap_at_swings(df)
    df = compute_volume_profile_and_fvgs(df)
    df = compute_indicators(df)

    stats = run_two_tp_trader_dale_backtest(df, initial_cash=50.0, lot_size=0.01, spread=0.00015, cooldown_bars=25)

    print("\n==================================================")
    print("🎯 EUR/USD M5 TWO TAKE-PROFITS PERFORMANCE (JAN - JUL 2026)")
    print("==================================================")
    print(f"  - Starting Capital:           ${stats['initial_cash']:.2f}")
    print(f"  - Final Equity:               ${stats['final_equity']:.2f}")
    print(f"  - Net Return [%]:             {stats['return_pct']:.2f}%")
    print(f"  - Win Rate [%]:               {stats['win_rate']:.2f}%")
    print(f"  - Total Trade Legs:           {stats['total_trades']} ta (Wins: {stats['num_wins']} | Losses: {stats['num_losses']})")
    print(f"  - Profit Factor:              {stats['profit_factor']:.2f}")
    print(f"  - Max. Drawdown [%]:          {stats['max_drawdown_pct']:.2f}%")
    print("==================================================")

    trades_df = stats['trades_df']
    if not trades_df.empty and 'exit_time' in trades_df.columns:
        trades_df['Month'] = pd.to_datetime(trades_df['exit_time']).dt.to_period('M')
        monthly_group = trades_df.groupby('Month').agg(
            Trades=('pnl', 'count'),
            Wins=('win', 'sum'),
            Losses=('win', lambda x: (~x).sum()),
            Monthly_PnL=('pnl', 'sum')
        ).reset_index()

        print("\n📅 OYLIK TAQSIMOT JADVALI (JAN-JUL 2026 MONTHLY BREAKDOWN):")
        print(monthly_group.to_string(index=False))

    plt.figure(figsize=(10, 4))
    plt.plot(stats['equity_times'], stats['equity_curve'], label="Equity ($50)", color='green' if stats['return_pct'] >= 0 else 'red')
    plt.title("EUR/USD M5 Trader Dale Two Take-Profits Equity (Jan-Jul 2026)")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.grid(True)
    plt.savefig("eurusd_m5_trader_dale_advanced_jan_jul_2026.png")
    plt.close()
    print("[+] Equity curve chart saved to: eurusd_m5_trader_dale_advanced_jan_jul_2026.png")

    mt5.shutdown()

if __name__ == "__main__":
    main()
