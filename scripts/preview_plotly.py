import os
import pandas as pd
import unicodedata
import plotly.express as px


def _norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return ''.join(ch for ch in unicodedata.normalize('NFKD', s) if not unicodedata.combining(ch)).lower()


def main():
    src = os.path.join(os.getcwd(), 'data', 'cleaned', 'station_day_clean.csv')
    out = os.path.join(os.getcwd(), 'frontend', 'preview_animated.html')
    if not os.path.exists(src):
        print('Data file not found:', src)
        return

    df = pd.read_csv(src)
    # normalize city column name possibilities
    city_col = None
    for c in ['city', 'City', 'location']:
        if c in df.columns:
            city_col = c
            break
    if city_col is None:
        # No city column present; fallback to using the overall dataset.
        print('No city column found in CSV. Columns:', df.columns.tolist())
        df_city = df.sort_index().tail(60).copy()
    else:
        df['city_norm'] = df[city_col].astype(str).apply(_norm)
        # look for Ho Chi Minh variants
        target_norms = ['hochiminh', 'ho chi minh', 'hcm', 'saigon']
        mask = df['city_norm'].apply(lambda s: any(t in s for t in target_norms))
        df_city = df[mask].copy()
        if df_city.empty:
            # fallback: take last N rows overall
            print('No rows for Ho Chi Minh found; using last 60 rows overall')
            df_city = df.sort_index().tail(60).copy()
        else:
            df_city = df_city.sort_index().tail(60).copy()

    # determine timestamp column
    ts_col = None
    for c in ['timestamp', 'timestamp_utc', 'date', 'day']:
        if c in df_city.columns:
            ts_col = c
            break
    if ts_col is None:
        # try to infer
        for c in df_city.columns:
            if 'time' in c.lower() or 'date' in c.lower():
                ts_col = c
                break

    if ts_col is None:
        df_city['ts'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df_city), freq='D')
    else:
        df_city['ts'] = pd.to_datetime(df_city[ts_col], errors='coerce')
        if df_city['ts'].isna().all():
            df_city['ts'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df_city), freq='D')

    # choose metric columns (adapt to the cleaned CSV schema)
    temp_col = None
    for c in ['temp', 'temperature', 't']:
        if c in df_city.columns:
            temp_col = c
            break
    # many cleaned datasets use 'PM2.5' with a dot or 'PM2_5'
    pm25_col = None
    for c in ['PM2.5', 'PM2_5', 'PM25', 'pm25', 'pm_25']:
        if c in df_city.columns:
            pm25_col = c
            break
    # AQI column variants
    aqi_col = None
    for c in ['AQI', 'aqi', 'AirQualityIndex']:
        if c in df_city.columns:
            aqi_col = c
            break

    df_city = df_city.sort_values('ts')
    df_city['frame'] = df_city['ts'].dt.strftime('%Y-%m-%d %H:%M:%S')

    figs = []
    if temp_col:
        fig = px.line(df_city, x='ts', y=temp_col, title='Nhiệt độ (animated)', labels={temp_col:'Nhiệt độ (°C)'}, animation_frame='frame')
        figs.append(('temp', fig))
    if pm25_col:
        fig2 = px.bar(df_city, x='frame', y=pm25_col, title='PM2.5 (animated)', labels={pm25_col:'PM2.5 (µg/m3)'}, animation_frame='frame')
        figs.append(('pm25', fig2))

    # If no preferred metrics, attempt to use numeric columns
    if not figs:
        numeric = df_city.select_dtypes('number').columns.tolist()
        if numeric:
            col = numeric[0]
            fig = px.line(df_city, x='ts', y=col, title=f'{col} (animated)', animation_frame='frame')
            figs.append((col, fig))

    if not figs:
        print('No numeric metrics found to plot.')
        return

    # Save a combined HTML with all figs stacked
    html_parts = ["<html><head><meta charset='utf-8'></head><body>"]
    for name, f in figs:
        html_parts.append(f.to_html(full_html=False, include_plotlyjs='cdn'))
        html_parts.append('<hr/>')
    html_parts.append('</body></html>')
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(html_parts))

    print('Saved animated preview to', out)


if __name__ == '__main__':
    main()
