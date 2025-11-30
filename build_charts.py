import pandas as pd
import altair as alt
from pathlib import Path

# Ensure we can embed full datasets
alt.data_transformers.disable_max_rows()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data'
CHART_DIR = ROOT / 'charts'
CHART_DIR.mkdir(exist_ok=True)

# Fonts
FONT_STACK = "SF Pro Display, SF Pro Icons, Helvetica Neue, Helvetica, Arial, sans-serif"


def save_chart(chart: alt.Chart, name: str):
    """Save a chart to both JSON (spec) and standalone HTML."""
    json_path = CHART_DIR / f"{name}.json"
    html_path = CHART_DIR / f"{name}.html"
    chart.save(json_path)
    chart.save(html_path, embed_options={"actions": False})
    print(f"saved {json_path.name}, {html_path.name}")

# Load data
mta = pd.read_csv(DATA_DIR / 'mta_ridership.csv')
weather = pd.read_csv(DATA_DIR / 'weather_daily.csv')
air = pd.read_csv(ROOT / 'cleaned_all_years_2020_2024_ny_air_quality.csv')

# Basic cleaning
mta['date'] = pd.to_datetime(mta['date'])
weather['date'] = pd.to_datetime(weather['date'])
air['date'] = pd.to_datetime(air['date'])

# Merge ridership + weather for weather-focused charts
merged_weather = mta.merge(weather, on='date', how='inner')

# Add day-of-week and precipitation bins for heatmap
merged_weather['day_of_week'] = merged_weather['date'].dt.day_name()
merged_weather['day_of_week'] = pd.Categorical(
    merged_weather['day_of_week'],
    categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
    ordered=True
)


def bin_precipitation(val: float) -> str:
    if pd.isna(val):
        return 'No Data'
    if val == 0:
        return 'No Rain'
    if val <= 0.1:
        return 'Light Rain'
    if val <= 0.5:
        return 'Moderate Rain'
    return 'Heavy Rain'


merged_weather['precipitation_level'] = merged_weather['precipitation_sum'].apply(bin_precipitation)
precip_order = ['No Rain', 'Light Rain', 'Moderate Rain', 'Heavy Rain']
merged_weather['precipitation_level'] = pd.Categorical(
    merged_weather['precipitation_level'], categories=precip_order, ordered=True
)
# Drop rows with missing precipitation buckets to avoid null cells
merged_weather = merged_weather[merged_weather['precipitation_level'] != 'No Data']

heat_grouped = (
    merged_weather
    .dropna(subset=['subways_total_estimated_ridership'])
    .groupby(['day_of_week','precipitation_level'], observed=False)['subways_total_estimated_ridership']
    .mean().reset_index()
    .rename(columns={'subways_total_estimated_ridership':'avg_ridership'})
)

# Chart 1: Ridership vs precipitation/day heatmap
min_val, max_val = heat_grouped['avg_ridership'].min(), heat_grouped['avg_ridership'].max()
threshold = heat_grouped['avg_ridership'].median()

heatmap = (
    alt.Chart(heat_grouped)
    .mark_rect()
    .encode(
        x=alt.X('precipitation_level:O', title='Precipitation Level', sort=precip_order,
                axis=alt.Axis(labelAngle=-45, labelFont=FONT_STACK, titleFont=FONT_STACK)),
        y=alt.Y('day_of_week:O', title='Day of Week',
                axis=alt.Axis(labelFont=FONT_STACK, titleFont=FONT_STACK)),
        color=alt.Color('avg_ridership:Q', title='Avg Daily Ridership',
                        scale=alt.Scale(scheme='blues', domain=[min_val, max_val])),
        tooltip=[
            alt.Tooltip('day_of_week:O', title='Day'),
            alt.Tooltip('precipitation_level:O', title='Precipitation'),
            alt.Tooltip('avg_ridership:Q', title='Avg Ridership', format=',.0f')
        ]
    )
)

text = (
    alt.Chart(heat_grouped)
    .mark_text(baseline='middle')
    .encode(
        x=alt.X('precipitation_level:O', sort=precip_order),
        y='day_of_week:O',
        text=alt.Text('avg_ridership:Q', format=',.0f'),
        color=alt.condition(alt.datum.avg_ridership > threshold, alt.value('white'), alt.value('black'))
    )
)

chart_heatmap = (
    alt.layer(heatmap, text)
    .properties(width='container', height=380, title='Average Subway Ridership by Day & Precipitation')
    .configure_title(font=FONT_STACK, fontSize=18)
    .configure_axis(labelFont=FONT_STACK, titleFont=FONT_STACK)
    .configure(autosize=alt.AutoSizeParams(type='fit', contains='padding'))
)
save_chart(chart_heatmap, "heatmap")

# Chart 2: Weather density ridges across periods

def categorize_period(date: pd.Timestamp) -> str:
    if date < pd.Timestamp('2020-03-15'):
        return 'Pre-Pandemic'
    if date < pd.Timestamp('2021-07-01'):
        return 'During Pandemic'
    if date < pd.Timestamp('2023-01-01'):
        return 'Recovery Phase'
    return 'Post-Pandemic'

merged_weather['period'] = merged_weather['date'].apply(categorize_period)


def categorize_weather(row) -> str:
    precip = row['precipitation_sum']
    temp = row['temperature_2m_mean']
    if pd.isna(precip):
        return None
    if precip == 0:
        return 'Sunny'
    if temp < 0 and precip > 0:
        if precip < 5:
            return 'Light Snow'
        if precip < 15:
            return 'Moderate Snow'
        return 'Heavy Snow'
    if precip <= 2:
        return 'Drizzle'
    if precip <= 8:
        return 'Light Rain'
    if precip <= 20:
        return 'Moderate Rain'
    if precip <= 35:
        return 'Heavy Rain'
    return 'Storm'


merged_weather['weather_category'] = merged_weather.apply(categorize_weather, axis=1)
subway_weather = merged_weather[['subways_total_estimated_ridership', 'weather_category', 'period']].dropna()
subway_weather = subway_weather.rename(columns={'subways_total_estimated_ridership': 'Ridership'})

color_domain = ['Sunny', 'Drizzle', 'Light Rain', 'Moderate Rain', 'Heavy Rain', 'Storm', 'Light Snow', 'Moderate Snow', 'Heavy Snow']
color_range = ['gold', 'cyan', 'deepskyblue', 'dodgerblue', 'blue', 'navy', 'lightcyan', 'skyblue', 'purple']

chart_ridges = (
    alt.Chart(subway_weather)
    .transform_density(
        density='Ridership', bandwidth=150000, groupby=['weather_category', 'period'],
        extent=[0, 6000000], counts=True, steps=200
    )
    .mark_area(orient='horizontal', opacity=0.8, interpolate='monotone')
    .encode(
        x=alt.X('density:Q', stack='center', impute=None, title=None,
                axis=alt.Axis(labels=False, values=[0], grid=False, ticks=False)),
        y=alt.Y('value:Q', title='Daily Subway Ridership', scale=alt.Scale(zero=False),
                axis=alt.Axis(format='~s', labelFont=FONT_STACK, titleFont=FONT_STACK)),
        color=alt.Color('weather_category:N',
                        scale=alt.Scale(domain=color_domain, range=color_range),
                        legend=alt.Legend(title='Weather Condition', orient='right', titleFont=FONT_STACK, labelFont=FONT_STACK, symbolSize=150)),
        column=alt.Column('period:N', title='Time Period',
                          header=alt.Header(labelAngle=0, labelAlign='center', titleFont=FONT_STACK, labelFont=FONT_STACK),
                          sort=['Pre-Pandemic', 'During Pandemic', 'Recovery Phase', 'Post-Pandemic']),
        tooltip=[
            alt.Tooltip('weather_category:N', title='Weather'),
            alt.Tooltip('period:N', title='Period'),
            alt.Tooltip('value:Q', title='Ridership', format=',.0f')
        ]
    )
    .properties(width=170, height=520,
                title={'text': 'NYC Subway Ridership: Weather Impact Across Pandemic Timeline',
                       'subtitle': 'Distribution across different weather conditions (2020-2024)',
                       'font': FONT_STACK,
                       'subtitleFont': FONT_STACK,
                       'fontSize': 18,
                       'subtitleFontSize': 13})
    .configure_axis(labelFont=FONT_STACK, titleFont=FONT_STACK)
    .configure(autosize=alt.AutoSizeParams(type='fit', contains='padding'))
)

save_chart(chart_ridges, "weather_density")

# Merge MTA + AQI
mta_for_aqi = mta.copy()
mta_for_aqi['date'] = mta_for_aqi['date'].dt.strftime('%Y-%m-%d')
air_for_merge = air.copy()
air_for_merge['date'] = air_for_merge['date'].dt.strftime('%Y-%m-%d')
merged_aqi = pd.merge(mta_for_aqi, air_for_merge, on='date', how='inner')
merged_aqi['date'] = pd.to_datetime(merged_aqi['date'])

# Helper to build AQI/ridership chart variations

def build_aqi_chart(df: pd.DataFrame, periods, colors, title_suffix: str, outfile: Path):
    brush = alt.selection_interval(encodings=['x'], name='DateRange')
    aqi_min, aqi_max = float(df['daily_aqi'].min()), float(df['daily_aqi'].max())
    aqi_param = alt.param(name='AQIMax', value=min(120, aqi_max),
                          bind=alt.binding_range(min=aqi_min, max=aqi_max, step=1, name='Max AQI: '))

    x_domain = [pd.Timestamp('2020-03-01'), df['date'].max()]

    top = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X('date:T', axis=alt.Axis(format='%b %Y', tickCount='month', title='Date', labelFont=FONT_STACK, titleFont=FONT_STACK),
                    scale=alt.Scale(domain=x_domain)),
            y=alt.Y('daily_aqi:Q', title='Daily AQI', axis=alt.Axis(labelFont=FONT_STACK, titleFont=FONT_STACK)),
            color=alt.Color('period:N', title='Period', scale=alt.Scale(domain=list(periods.keys()), range=list(periods.values()))),
            tooltip=['date:T','daily_aqi:Q','period:N']
        )
        .add_params(brush, aqi_param)
        .properties(width=800, height=140, title='Brush to filter the scatter below')
    )

    main = (
        alt.Chart(df)
        .transform_filter(brush)
        .transform_filter('datum.daily_aqi <= AQIMax')
        .mark_point(opacity=0.6, size=35)
        .encode(
            x=alt.X('date:T', axis=alt.Axis(format='%Y-%m', title='Date', labelFont=FONT_STACK, titleFont=FONT_STACK),
                    scale=alt.Scale(domain=x_domain)),
            y=alt.Y('ridership:Q', title='Ridership', axis=alt.Axis(labelFont=FONT_STACK, titleFont=FONT_STACK)),
            color=alt.Color('period:N', title='Period', scale=alt.Scale(domain=list(periods.keys()), range=list(periods.values()))),
            tooltip=['date:T', alt.Tooltip('ridership:Q', title='Ridership'), 'daily_aqi:Q','period:N']
        )
        .properties(width=800, height=360, title=f'Subway Ridership over Time under AQI Threshold ({title_suffix})')
    )

    reg = (
        alt.Chart(df)
        .transform_filter(brush)
        .transform_filter('datum.daily_aqi <= AQIMax')
        .transform_regression('date', 'ridership', groupby=['period'])
        .mark_line(strokeDash=[4,2])
        .encode(
            x=alt.X('date:T', scale=alt.Scale(domain=x_domain)),
            y='ridership:Q',
            color=alt.Color('period:N', scale=alt.Scale(domain=list(periods.keys()), range=list(periods.values())))
        )
    )

    chart = (
        alt.vconcat(top, main + reg)
        .resolve_scale(color='shared')
        .configure_title(font=FONT_STACK)
    )
    name = outfile.stem
    save_chart(chart, name)

# Chart 3: AQI version 1 (Pre/During/Recovery/Post)
periods_v1 = {
    'Pre-COVID': '#2ca02c',
    'During-COVID': '#1f77b4',
    'Recovery': '#d62728',
    'Post-COVID': '#ff7f0e'
}

df_v1 = merged_aqi.copy().sort_values('date')
df_v1['period'] = 'Post-COVID'
df_v1.loc[df_v1['date'] < pd.Timestamp('2020-03-01'), 'period'] = 'Pre-COVID'
df_v1.loc[(df_v1['date'] >= pd.Timestamp('2020-03-01')) & (df_v1['date'] <= pd.Timestamp('2021-06-30')), 'period'] = 'During-COVID'
df_v1.loc[(df_v1['date'] > pd.Timestamp('2021-06-30')) & (df_v1['date'] <= pd.Timestamp('2022-12-31')), 'period'] = 'Recovery'

build_aqi_chart(df_v1, periods_v1, None, 'COVID period bands', CHART_DIR / 'aqi_periods.json')

# Chart 4: AQI version 2 (Pre/During/Recovery Phase/Post as in notebook cell 11)
periods_v2 = {
    'Pre-Pandemic': '#2ca02c',
    'During Pandemic': '#1f77b4',
    'Recovery Phase': '#d62728',
    'Post-Pandemic': '#ff7f0e'
}

df_v2 = merged_aqi.copy().sort_values('date')
df_v2['period'] = df_v2['date'].apply(categorize_period)

build_aqi_chart(df_v2, periods_v2, None, 'Pandemic timeline', CHART_DIR / 'aqi_periods_v2.json')

# Chart 5: MTA usage over time (from notebook)
def prepare_mta_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Create tidy totals and percentage columns for subway/bus/bridge usage."""
    tidy_df = df.copy()
    tidy_df = tidy_df.rename(columns={'buses_total_estimated_ridersip': 'buses_total_estimated_ridership'})

    totals = {
        'subways_total_estimated_ridership': 'Subways',
        'buses_total_estimated_ridership': 'Buses',
        'bridges_and_tunnels_total_traffic': 'Bridges & Tunnels'
    }
    percents = {
        'subways_of_comparable_pre_pandemic_day': 'Subways',
        'buses_of_comparable_pre_pandemic_day': 'Buses',
        'bridges_and_tunnels_of_comparable_pre_pandemic_day': 'Bridges & Tunnels'
    }

    for col in list(totals.keys()) + list(percents.keys()):
        tidy_df[col] = pd.to_numeric(tidy_df[col], errors='coerce')

    total_long = (
        tidy_df[['date'] + list(totals.keys())]
        .rename(columns=totals)
        .melt(id_vars='date', var_name='mode', value_name='value_total')
    )
    pct_long = (
        tidy_df[['date'] + list(percents.keys())]
        .rename(columns=percents)
        .melt(id_vars='date', var_name='mode', value_name='value_pct')
    )
    tidy = total_long.merge(pct_long, on=['date', 'mode'])
    tidy['date'] = pd.to_datetime(tidy['date'])

    # Express percentages as 0-100 instead of ratios
    tidy['value_pct'] = tidy['value_pct'] * 100
    return tidy


def build_mta_usage_chart(tidy: pd.DataFrame) -> alt.VConcatChart:
    """Interactive multi-line chart with brush + toggle for counts vs percentages."""
    toggle = alt.param(
        name="Show_Percentage",
        bind=alt.binding_checkbox(name=" Show percentage of comparable day"),
        value=False
    )
    legend_sel = alt.selection_point(fields=["mode"], bind="legend")
    brush = alt.selection_interval(encodings=["x"])

    base = (
        alt.Chart(tidy)
        .transform_calculate(
            y_value="Show_Percentage ? datum.value_pct : datum.value_total"
        )
        .encode(
            y=alt.Y(
                "y_value:Q",
                title="Ridership (counts or % of comparable day)",
                axis=alt.Axis(labelFont=FONT_STACK, titleFont=FONT_STACK, format=",.0f")
            ),
            color=alt.Color(
                "mode:N",
                title="Mode",
                legend=alt.Legend(titleFont=FONT_STACK, labelFont=FONT_STACK)
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("mode:N", title="Mode"),
                alt.Tooltip("value_total:Q", title="Total", format=","),
                alt.Tooltip("value_pct:Q", title="% of pre-pandemic", format=",.1f")
            ]
        )
        .add_params(toggle, legend_sel)
        .transform_filter(legend_sel)
    )

    upper = (
        base.mark_line()
        .encode(
            x=alt.X(
                "date:T",
                title="Date",
                scale=alt.Scale(domain=brush),
                axis=alt.Axis(labelFont=FONT_STACK, titleFont=FONT_STACK)
            )
        )
        .properties(
            width='container',
            height=360,
            title="NYC Transit Usage by Mode (toggle % to view recovery)"
        )
    )

    lower = (
        base.mark_area(opacity=0.2)
        .encode(
            x=alt.X(
                "date:T",
                title="Drag to zoom / pan the main view",
                axis=alt.Axis(labelFont=FONT_STACK, titleFont=FONT_STACK)
            ),
            y=alt.Y("y_value:Q", title="", axis=alt.Axis(labels=False, ticks=False))
        )
        .add_params(brush)
        .properties(width='container', height=90)
    )

    return (
        alt.vconcat(upper, lower, spacing=8)
        .resolve_scale(color='shared')
        .configure_title(font=FONT_STACK, fontSize=18)
        .configure_axis(labelFont=FONT_STACK, titleFont=FONT_STACK)
        .configure(autosize=alt.AutoSizeParams(type='fit', contains='padding'))
    )


tidy_usage = prepare_mta_usage(mta)
mta_usage_chart = build_mta_usage_chart(tidy_usage)
save_chart(mta_usage_chart, "mta_usage")

print('Charts saved to', CHART_DIR)
