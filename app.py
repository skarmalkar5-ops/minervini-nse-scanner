import smtplib
import time
from datetime import datetime
from email.message import EmailMessage

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title='Minervini NSE Scanner', page_icon='📈', layout='wide')
st.title('📈 Minervini NSE Scanner')
st.caption('NSE → Trend Template → RS → Leader Score → VCP proxy → CSV → Email')

MIN_TREND = 88.9
MIN_RS = 80.0
MIN_LEADER = 85.0
MIN_VCP = 80.0
BATCH = 100


def nse_universe():
    df = pd.read_csv('https://archives.nseindia.com/content/equities/EQUITY_L.csv')
    df.columns = df.columns.str.strip()
    df = df[df['SERIES'].astype(str).str.strip().eq('EQ')].drop_duplicates('SYMBOL').copy()
    df['YF_SYMBOL'] = df['SYMBOL'].astype(str).str.strip() + '.NS'
    return df


def stock_from_raw(raw, symbol):
    if raw is None or raw.empty:
        return None
    if not isinstance(raw.columns, pd.MultiIndex):
        x = raw.copy()
    else:
        l0 = raw.columns.get_level_values(0)
        l1 = raw.columns.get_level_values(1)
        if symbol in l0:
            x = raw[symbol].copy()
        elif symbol in l1:
            x = raw.xs(symbol, level=1, axis=1).copy()
        else:
            return None
    x = x.loc[:, ~x.columns.duplicated()]
    need = ['High','Low','Close','Volume']
    if not all(c in x.columns for c in need):
        return None
    x = x[need].copy()
    for c in need:
        x[c] = pd.to_numeric(x[c], errors='coerce')
    return x.dropna(subset=['Close'])


def download(symbols, retries=3):
    for attempt in range(retries):
        try:
            raw = yf.download(symbols, period='2y', interval='1d', auto_adjust=True,
                              group_by='ticker', threads=True, progress=False)
            if raw is not None and not raw.empty:
                return raw
        except Exception:
            pass
        time.sleep(2 + attempt * 2)
    return None


def nifty_close():
    for attempt in range(3):
        try:
            raw = yf.download('^NSEI', period='2y', interval='1d', auto_adjust=True, progress=False)
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                if 'Close' in raw.columns.get_level_values(0):
                    x = raw['Close']
                elif 'Close' in raw.columns.get_level_values(1):
                    x = raw.xs('Close', level=1, axis=1)
                else:
                    continue
                if isinstance(x, pd.DataFrame):
                    x = x.iloc[:, 0]
            else:
                x = raw['Close']
            x = pd.to_numeric(x, errors='coerce').dropna()
            if len(x) >= 253:
                return x.rename('NIFTY')
        except Exception:
            pass
        time.sleep(2 + attempt * 2)
    raise RuntimeError('Could not download NIFTY 50 data.')


def trend(stock):
    if stock is None or len(stock) < 252:
        return None
    x = stock.copy()
    x['50_DMA'] = x.Close.rolling(50).mean()
    x['150_DMA'] = x.Close.rolling(150).mean()
    x['200_DMA'] = x.Close.rolling(200).mean()
    x = x.dropna(subset=['50_DMA','150_DMA','200_DMA'])
    if len(x) < 252:
        return None
    a = x.iloc[-1]
    p, d50, d150, d200 = map(float, [a.Close,a['50_DMA'],a['150_DMA'],a['200_DMA']])
    d200_20 = float(x['200_DMA'].iloc[-21])
    hi = float(x.High.tail(252).max())
    lo = float(x.Low.tail(252).min())
    passed = sum([
        p>d50, p>d150, p>d200, d50>d150, d50>d200,
        d150>d200, d200>d200_20, p>=hi*.75, p>=lo*1.30
    ])
    return {'Price':round(p,2),'50_DMA':round(d50,2),'150_DMA':round(d150,2),
            '200_DMA':round(d200,2),'52W_High':round(hi,2),'52W_Low':round(lo,2),
            'Passed':passed,'Trend_Score':round(passed/9*100,1)}


def rs(stock, nifty):
    if stock is None or len(stock) < 253:
        return None
    s = pd.to_numeric(stock.Close, errors='coerce').dropna().rename('Stock')
    z = pd.concat([s, nifty], axis=1, join='inner').dropna()
    if len(z) < 253:
        return None
    out = {}
    for name, days in [('1M',21),('3M',63),('6M',126),('12M',252)]:
        sr = (z.Stock.iloc[-1]/z.Stock.iloc[-days-1]-1)*100
        nr = (z.NIFTY.iloc[-1]/z.NIFTY.iloc[-days-1]-1)*100
        out[f'{name}_Rel'] = round(float(sr-nr),2)
    return out


def vcp(stock):
    if stock is None or len(stock) < 70:
        return None
    r = stock.tail(60); a = r.head(30); b = r.tail(30)
    rf = (a.High.max()-a.Low.min())/a.Close.mean()
    rl = (b.High.max()-b.Low.min())/b.Close.mean()
    rc = rl < rf
    pc = r.Close.shift(1)
    tr = pd.concat([r.High-r.Low,(r.High-pc).abs(),(r.Low-pc).abs()],axis=1).max(axis=1)
    atr = tr.rolling(14).mean(); early=atr.iloc[14:29].mean(); recent=atr.iloc[-15:].mean()
    if pd.isna(early) or early == 0: return None
    ac = recent < early
    vc = b.Volume.mean() < a.Volume.mean()
    hp, lp = r.High.max(), r.Low.min(); cp=float(r.Close.iloc[-1])
    if hp == lp: return None
    pos=(cp-lp)/(hp-lp); upper=pos>=.70; tight=((hp-lp)/cp*100)<=25
    passed=sum([rc,ac,vc,upper,tight])
    return {'VCP Passed':passed,'VCP Score':round(passed/5*100,1),
            'Base Range %':round((hp-lp)/cp*100,2),'Range Position %':round(pos*100,1),
            'ATR Change %':round((recent/early-1)*100,2),
            'Volume Change %':round((b.Volume.mean()/a.Volume.mean()-1)*100,2)}


def email_csv(data, filename):
    host=st.secrets['SMTP_HOST']; port=int(st.secrets.get('SMTP_PORT',587))
    user=st.secrets['SMTP_USER']; pwd=st.secrets['SMTP_PASSWORD']; to=st.secrets['EMAIL_TO']
    msg=EmailMessage(); msg['Subject']=f"Minervini NSE Scan — {datetime.now():%d-%b-%Y}"; msg['From']=user; msg['To']=to
    msg.set_content(f"Minervini scan completed. {len(data)} candidates attached.")
    msg.add_attachment(data, maintype='text', subtype='csv', filename=filename)
    with smtplib.SMTP(host,port,timeout=30) as s:
        s.starttls(); s.login(user,pwd); s.send_message(msg)


def run_scan(progress):
    universe=nse_universe(); symbols=universe.YF_SYMBOL.tolist()
    nifty=nifty_close(); trend_rows=[]; rs_rows=[]
    batches=(len(symbols)+BATCH-1)//BATCH
    for start in range(0,len(symbols),BATCH):
        batch=symbols[start:start+BATCH]; no=start//BATCH+1
        progress(.05+.55*no/batches,f'Scanning batch {no}/{batches}…')
        raw=download(batch)
        if raw is not None:
            for sym in batch:
                try:
                    stx=stock_from_raw(raw,sym)
                    t=trend(stx); r=rs(stx,nifty)
                    if t: trend_rows.append({'Symbol':sym,**t})
                    if r: rs_rows.append({'Symbol':sym,**r})
                except Exception:
                    continue
        time.sleep(.4)
    if not trend_rows: raise RuntimeError('No Trend results returned. Yahoo Finance may be temporarily unavailable.')
    if not rs_rows: raise RuntimeError('No RS results returned. Yahoo Finance may be temporarily unavailable or histories did not overlap.')
    trend_df=pd.DataFrame(trend_rows); rs_df=pd.DataFrame(rs_rows)
    board=trend_df.merge(rs_df,on='Symbol',how='inner')
    if board.empty: raise RuntimeError('Trend and RS results had no common symbols.')
    for p in ['1M_Rel','3M_Rel','6M_Rel','12M_Rel']:
        board[p+'_Rank']=board[p].rank(pct=True)*100
    board['RS_Rating']=(board['1M_Rel_Rank']*.15+board['3M_Rel_Rank']*.25+board['6M_Rel_Rank']*.25+board['12M_Rel_Rank']*.35).round(1)
    board['Leader_Score']=(board.Trend_Score*.5+board.RS_Rating*.5).round(1)
    cand=board[(board.Trend_Score>=MIN_TREND)&(board.RS_Rating>=MIN_RS)&(board.Leader_Score>=MIN_LEADER)].copy()
    progress(.65+.02,f'Leader filter: {len(cand)} candidates')
    vrows=[]; syms=cand.Symbol.tolist(); vbatches=max(1,(len(syms)+BATCH-1)//BATCH)
    for start in range(0,len(syms),BATCH):
        batch=syms[start:start+BATCH]; no=start//BATCH+1
        progress(.67+.28*no/vbatches,f'VCP analysis {no}/{vbatches}…')
        raw=download(batch)
        if raw is None: continue
        for sym in batch:
            try:
                v=vcp(stock_from_raw(raw,sym))
                if v: vrows.append({'Symbol':sym,**v})
            except Exception: continue
    vdf=pd.DataFrame(vrows)
    if vdf.empty:
        final=pd.DataFrame(columns=list(cand.columns)+['VCP Passed','VCP Score','Base Range %','Range Position %','ATR Change %','Volume Change %'])
    else:
        final=cand.merge(vdf,on='Symbol',how='inner'); final = final[final["VCP Score"] >= MIN_VCP_SCORE].copy()
    final=final.sort_values(['Leader_Score','VCP Score'],ascending=False).reset_index(drop=True)
    final.Symbol=final.Symbol.str.replace('.NS','',regex=False)
    progress(1.0,f'Scan complete: {len(final)} final candidates')
    return final,len(symbols),len(trend_df),len(rs_df),len(board)

st.sidebar.header('Scanner rules')
st.sidebar.write(f'Trend Score ≥ {MIN_TREND}')
st.sidebar.write(f'RS Rating ≥ {MIN_RS}')
st.sidebar.write(f'Leader Score ≥ {MIN_LEADER}')
st.sidebar.write(f'VCP Score ≥ {MIN_VCP}')
st.sidebar.caption('No arbitrary top-50 limit.')

if 'results' not in st.session_state: st.session_state.results=None

if st.button('🔄 REFRESH SCAN',type='primary',use_container_width=True):
    bar=st.progress(0); status=st.empty()
    def progress(v,m): bar.progress(min(max(v,0),1)); status.info(m)
    try:
        with st.spinner('Running full NSE scan…'):
            final,n,tr,rsn,leaders=run_scan(progress)
        st.session_state.results=final; st.session_state.meta={'universe':n,'trend':tr,'rs':rsn,'leaders':leaders}
        st.success('Scan completed successfully.')
        csv=final.to_csv(index=False).encode('utf-8'); fn=f"minervini_scan_{datetime.now():%Y-%m-%d}.csv"
        try:
            email_csv(csv,fn); st.success('📧 CSV emailed successfully.')
        except Exception as e:
            st.warning('Scan completed, but email is not configured yet.'); st.exception(e)
    except Exception as e:
        st.error('The scan failed.'); st.exception(e)

if st.session_state.results is not None:
    f=st.session_state.results; m=st.session_state.meta
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric('NSE EQ',m['universe']); c2.metric('Trend',m['trend']); c3.metric('RS',m['rs']); c4.metric('Common',m['leaders']); c5.metric('Final VCP',len(f))
    st.subheader('Final candidates'); st.dataframe(f,use_container_width=True,hide_index=True)
    csv=f.to_csv(index=False).encode('utf-8')
    st.download_button('📥 Download CSV',csv,file_name=f"minervini_scan_{datetime.now():%Y-%m-%d}.csv",mime='text/csv',use_container_width=True)
