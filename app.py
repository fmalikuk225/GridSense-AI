import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.forecast import sample_data, validate_frame, fit_evaluate, forecast_next_24, evaluate_24h_origins

st.set_page_config(page_title='GridSense AI | Energy Intelligence',page_icon='⚡',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
.stApp {background:linear-gradient(145deg,#081321,#101f34 70%,#071420);color:#eaf2fb}
[data-testid="stSidebar"] {background:#0d1b2d}
h1,h2,h3 {color:#f3f8ff!important;letter-spacing:-.035em}
[data-testid="stMetric"] {background:#152a42;border:1px solid #294563;padding:16px;border-radius:14px}
[data-testid="stMetricLabel"] {color:#b9cce2}
[data-testid="stMetricValue"] {color:#66e0ca}
.stAlert {border-radius:12px}
</style>''',unsafe_allow_html=True)
st.sidebar.markdown('## ⚡ GridSense AI')
st.sidebar.caption('ENERGY INTELLIGENCE LAB · v1.0')
st.sidebar.divider()
st.sidebar.markdown('### Data source')
source=st.sidebar.radio('Select input',['Synthetic demonstration','Upload hourly CSV'],label_visibility='collapsed')
st.sidebar.info('Prototype only. Not an operational grid dispatch or trading system.')
st.sidebar.markdown('**CSV schema**\n\n`timestamp,demand_mw` · hourly UTC, continuous · ≥21 days')
st.sidebar.download_button('Download example CSV',sample_data(30).rename_axis('timestamp').to_csv().encode(),file_name='example_synthetic.csv',mime='text/csv')
st.title('⚡ GridSense AI')
st.caption('Electricity demand forecasting · peak-load intelligence · transparent model benchmarking')
st.markdown('### Grid demand intelligence')
try:
    if source=='Synthetic demonstration':
        data=sample_data()
        st.warning('DEMONSTRATION DATA — entirely synthetic. No real National Grid, RWE, or system operator data is represented.')
    else:
        file=st.sidebar.file_uploader('Upload CSV',type='csv')
        if file is None:
            st.info('Upload a CSV file using the sidebar to begin.');st.stop()
        data=validate_frame(pd.read_csv(file))
    with st.spinner('Training model and evaluating chronological holdout...'):
        model,results,scores=fit_evaluate(data)
        future=forecast_next_24(model,data)
        horizon_results,horizon_scores=evaluate_24h_origins(model,data)
    threshold=float(data['demand_mw'].quantile(.9))
    peak_count=int((future.iloc[:,0]>threshold).sum())
    a,b,c,d=st.columns(4)
    a.metric('Latest demand',f"{data['demand_mw'].iloc[-1]:,.0f} MW")
    b.metric('Next 24h peak',f'{future.iloc[:,0].max():,.0f} MW')
    c.metric('Forecast peak hours',f'{peak_count} / 24')
    d.metric('Holdout MAE',f"{scores.loc['Random Forest','MAE (MW)']:,.0f} MW")
    tab1,tab2,tab3=st.tabs(['📈 Forecast','🧪 Model evaluation','📋 Operational insights'])
    with tab1:
        st.subheader('Historical load and next 24-hour forecast')
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=data.index[-168:],y=data['demand_mw'].iloc[-168:],name='Observed demand',line=dict(color='#5ca8ff',width=2)))
        fig.add_trace(go.Scatter(x=future.index,y=future.iloc[:,0],name='AI forecast',line=dict(color='#64e2c7',width=3)))
        fig.add_hline(y=threshold,line_dash='dash',line_color='#ffbb70',annotation_text='Historical 90th percentile')
        fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',height=440,xaxis_title='UTC',yaxis_title='MW',legend_orientation='h',margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig,use_container_width=True)
        st.download_button('Export 24-hour forecast CSV',future.rename_axis('timestamp').to_csv().encode(),file_name='gridsense_forecast.csv',mime='text/csv')
    with tab2:
        st.subheader('Chronological holdout: last 20%')
        st.dataframe(scores,use_container_width=True)
        fig2=go.Figure()
        for col,color in [('Actual','#e9f0fb'),('Random Forest','#64e2c7'),('Seasonal naive (24h)','#ffbb70')]:
            fig2.add_trace(go.Scatter(x=results.index[-168:],y=results[col].iloc[-168:],name=col,line=dict(color=color)))
        fig2.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',height=370,xaxis_title='UTC',yaxis_title='MW')
        st.plotly_chart(fig2,use_container_width=True)
        st.subheader('True recursive 24-hour backtest')
        st.dataframe(horizon_scores,use_container_width=True)
        st.caption(f'Backtest uses {horizon_results.origin_utc.nunique()} origins in the chronological holdout, forecasting 24 hours from each origin without accessing future observations. Model is fitted once on the earlier training split; overlapping forecast windows mean errors are not independent.')
        st.download_button('Export recursive backtest CSV',horizon_results.to_csv(index=False).encode(),file_name='recursive_backtest.csv',mime='text/csv')
        st.caption('Holdout evaluation uses observed lag values. The next-24h forecast is recursive and may accumulate error; holdout metrics do not measure that full recursive horizon.')
    with tab3:
        st.subheader('Forecast-based demand signals')
        st.write(f'Historical 90th-percentile threshold: **{threshold:,.0f} MW**. Predicted hours above threshold: **{peak_count}**.')
        peaks=future[future.iloc[:,0]>threshold].copy()
        st.dataframe(peaks.rename(columns={'Forecast demand (MW)':'Predicted high demand (MW)'}),use_container_width=True)
        if peaks.empty:st.success('No forecast hours exceed the selected historical threshold.')
        else:st.warning('Potential high-demand periods identified. Human review and independent validation required before operational use.')
        st.info('These are illustrative analytics, not grid-balancing instructions, trading advice, or calibrated risk probabilities.')
    st.divider()
    st.caption('GridSense AI · Independent portfolio demonstration · No affiliation with National Grid or RWE · All timestamps UTC')
except Exception as exc:
    st.error(f'Data validation or model execution failed: {exc}')
