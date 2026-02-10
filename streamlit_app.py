import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Traffic Prediction System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        padding: 0.5rem;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# GRU Model Class
class TrafficGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(TrafficGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Fuzzy Membership Functions
def membership_night(hour):
    if 0 <= hour <= 3:
        return 1.0
    elif 3 < hour <= 6:
        return (6 - hour) / 3
    elif 22 <= hour <= 24:
        return (hour - 22) / 2
    else:
        return 0.0

def membership_peak(hour):
    if 6 <= hour < 7:
        return (hour - 6)
    elif 7 <= hour <= 8:
        return 1.0
    elif 8 < hour <= 9:
        return (9 - hour)
    elif 16 <= hour < 17:
        return (hour - 16)
    elif 17 <= hour <= 18:
        return 1.0
    elif 18 < hour <= 19:
        return (19 - hour)
    else:
        return 0.0

# Load model and scalers
@st.cache_resource
def load_model(junction_id):
    try:
        # Load model checkpoint
        checkpoint = torch.load(f'models/gru_model_junction_{junction_id}.pth', 
                               map_location=torch.device('cpu'))
        
        # Initialize model
        model = TrafficGRU(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load scalers
        with open(f'models/scaler_X_junction_{junction_id}.pkl', 'rb') as f:
            scaler_X = pickle.load(f)
        
        with open(f'models/scaler_y_junction_{junction_id}.pkl', 'rb') as f:
            scaler_y = pickle.load(f)
        
        return model, scaler_X, scaler_y, checkpoint
    except Exception as e:
        st.error(f"Error loading model for Junction {junction_id}: {str(e)}")
        return None, None, None, None

# Load traffic data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('traffic.csv')
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Feature engineering function
def create_features(hour, weekday, lag_1, lag_24, roll_mean_24):
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    return np.array([[hour_sin, hour_cos, lag_1, lag_24, roll_mean_24]])

# Apply fuzzy adjustment
def apply_fuzzy_adjustment(prediction, hour, is_weekend):
    night_degree = membership_night(hour)
    peak_degree = membership_peak(hour)
    
    adjustment = 0.0
    
    if is_weekend == 0:  # Weekday
        adjustment -= 0.15 * night_degree
        adjustment += 0.10 * peak_degree
    else:  # Weekend
        adjustment -= 0.08
    
    adjusted_pred = prediction * (1 + adjustment)
    return max(0, adjusted_pred)

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🚦 Traffic Prediction System</div>', 
                unsafe_allow_html=True)
    st.markdown("### GRU + Fuzzy Logic + PSO Optimization")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Junction selection
    junction_id = st.sidebar.selectbox(
        "Select Junction",
        options=[1, 2, 3, 4],
        format_func=lambda x: f"Junction {x}"
    )
    
    # Prediction mode
    prediction_mode = st.sidebar.radio(
        "Prediction Mode",
        ["Single Prediction", "Batch Prediction", "Historical Analysis"]
    )
    
    # Load model
    with st.spinner(f'Loading model for Junction {junction_id}...'):
        model, scaler_X, scaler_y, checkpoint = load_model(junction_id)
    
    if model is None:
        st.error("Failed to load model. Please check if model files exist.")
        return
    
    # Model info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Info")
    st.sidebar.info(f"""
    **Junction:** {junction_id}  
    **Hidden Size:** {checkpoint['hidden_size']}  
    **Layers:** {checkpoint['num_layers']}  
    **Learning Rate:** {checkpoint['best_params']['learning_rate']:.6f}
    """)
    
    if 'metrics' in checkpoint:
        st.sidebar.markdown("### 📈 Performance Metrics")
        metrics = checkpoint['metrics']
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("RMSE (GRU)", f"{metrics['rmse_gru']:.2f}")
            st.metric("R² (GRU)", f"{metrics['r2_gru']:.3f}")
        with col2:
            st.metric("RMSE (Fuzzy)", f"{metrics['rmse_fuzzy']:.2f}")
            st.metric("R² (Fuzzy)", f"{metrics['r2_fuzzy']:.3f}")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Filter to selected junction
    df_junction = df[df['Junction'] == junction_id].copy()
    
    # Main content based on mode
    if prediction_mode == "Single Prediction":
        single_prediction_mode(model, scaler_X, scaler_y, df_junction)
    
    elif prediction_mode == "Batch Prediction":
        batch_prediction_mode(model, scaler_X, scaler_y, df_junction)
    
    else:  # Historical Analysis
        historical_analysis_mode(df_junction)

# Single Prediction Mode
def single_prediction_mode(model, scaler_X, scaler_y, df_junction):
    st.markdown('<div class="sub-header">🎯 Single Time Prediction</div>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_date = st.date_input(
            "Select Date",
            value=datetime.now().date(),
            min_value=datetime(2015, 11, 1).date(),
            max_value=datetime.now().date()
        )
        
        selected_hour = st.slider("Select Hour", 0, 23, 12)
        
        is_weekend = st.checkbox("Is Weekend?", value=False)
    
    with col2:
        st.markdown("### 📝 Input Features")
        
        # Get historical data for lag features
        weekday = selected_date.weekday()
        
        # Find similar historical data
        df_temp = df_junction[df_junction['DateTime'].dt.date <= selected_date].copy()
        
        if len(df_temp) > 24:
            lag_1 = st.number_input("Traffic 1 hour ago", value=50.0, min_value=0.0)
            lag_24 = st.number_input("Traffic 24 hours ago", value=50.0, min_value=0.0)
            roll_mean_24 = st.number_input("24-hour average", value=50.0, min_value=0.0)
        else:
            st.warning("Insufficient historical data. Using defaults.")
            lag_1 = 50.0
            lag_24 = 50.0
            roll_mean_24 = 50.0
    
    if st.button("🔮 Predict Traffic"):
        with st.spinner("Predicting..."):
            # Create features
            features = create_features(selected_hour, weekday, lag_1, lag_24, roll_mean_24)
            
            # Scale features
            features_scaled = scaler_X.transform(features)
            
            # Create sequence (repeat for 24 timesteps)
            sequence = np.repeat(features_scaled, 24, axis=0).reshape(1, 24, -1)
            
            # Predict
            with torch.no_grad():
                pred_scaled = model(torch.FloatTensor(sequence)).numpy()
            
            # Inverse transform
            pred_gru = scaler_y.inverse_transform(pred_scaled)[0][0]
            
            # Apply fuzzy logic
            pred_fuzzy = apply_fuzzy_adjustment(pred_gru, selected_hour, int(is_weekend))
            
            # Display results
            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    label="GRU Prediction",
                    value=f"{pred_gru:.0f} vehicles"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    label="Fuzzy Adjusted",
                    value=f"{pred_fuzzy:.0f} vehicles",
                    delta=f"{pred_fuzzy - pred_gru:+.0f}"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                night_deg = membership_night(selected_hour)
                peak_deg = membership_peak(selected_hour)
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.write("**Membership Degrees:**")
                st.write(f"Night: {night_deg:.2f}")
                st.write(f"Peak: {peak_deg:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Visualization
            st.markdown("### 📊 Fuzzy Membership Functions")
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Night-Time Membership", "Peak-Time Membership")
            )
            
            hours = np.arange(0, 24, 0.1)
            night_values = [membership_night(h) for h in hours]
            peak_values = [membership_peak(h) for h in hours]
            
            fig.add_trace(
                go.Scatter(x=hours, y=night_values, fill='tozeroy', name='Night'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=hours, y=peak_values, fill='tozeroy', name='Peak'),
                row=1, col=2
            )
            
            # Add current hour marker
            fig.add_vline(x=selected_hour, line_dash="dash", line_color="red", 
                         row=1, col=1, annotation_text=f"Hour: {selected_hour}")
            fig.add_vline(x=selected_hour, line_dash="dash", line_color="red", 
                         row=1, col=2, annotation_text=f"Hour: {selected_hour}")
            
            fig.update_xaxes(title_text="Hour of Day", row=1, col=1)
            fig.update_xaxes(title_text="Hour of Day", row=1, col=2)
            fig.update_yaxes(title_text="Membership Degree", row=1, col=1)
            fig.update_yaxes(title_text="Membership Degree", row=1, col=2)
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# Batch Prediction Mode
def batch_prediction_mode(model, scaler_X, scaler_y, df_junction):
    st.markdown('<div class="sub-header">📅 24-Hour Forecast</div>', 
                unsafe_allow_html=True)
    
    selected_date = st.date_input(
        "Select Start Date",
        value=datetime.now().date(),
        min_value=datetime(2015, 11, 1).date(),
        max_value=datetime.now().date()
    )
    
    use_fuzzy = st.checkbox("Apply Fuzzy Logic Adjustment", value=True)
    
    if st.button("🔮 Generate 24-Hour Forecast"):
        with st.spinner("Generating predictions..."):
            # Prepare data
            predictions_gru = []
            predictions_fuzzy = []
            hours_list = []
            
            # Get last 24 hours of data for initial features
            df_temp = df_junction[df_junction['DateTime'].dt.date <= selected_date].copy()
            
            if len(df_temp) < 24:
                st.error("Insufficient historical data for prediction.")
                return
            
            # Extract features for each hour
            for hour in range(24):
                current_datetime = datetime.combine(selected_date, datetime.min.time()) + timedelta(hours=hour)
                weekday = current_datetime.weekday()
                is_weekend = 1 if weekday >= 5 else 0
                
                # Use last known values (simplified)
                lag_1 = df_temp['Vehicles'].iloc[-1] if len(df_temp) > 0 else 50
                lag_24 = df_temp['Vehicles'].iloc[-24] if len(df_temp) >= 24 else 50
                roll_mean_24 = df_temp['Vehicles'].tail(24).mean() if len(df_temp) >= 24 else 50
                
                # Create features
                features = create_features(hour, weekday, lag_1, lag_24, roll_mean_24)
                features_scaled = scaler_X.transform(features)
                sequence = np.repeat(features_scaled, 24, axis=0).reshape(1, 24, -1)
                
                # Predict
                with torch.no_grad():
                    pred_scaled = model(torch.FloatTensor(sequence)).numpy()
                pred_gru = scaler_y.inverse_transform(pred_scaled)[0][0]
                
                # Apply fuzzy
                pred_fuzzy = apply_fuzzy_adjustment(pred_gru, hour, is_weekend)
                
                predictions_gru.append(pred_gru)
                predictions_fuzzy.append(pred_fuzzy)
                hours_list.append(hour)
            
            # Create visualization
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=hours_list,
                y=predictions_gru,
                mode='lines+markers',
                name='GRU Prediction',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ))
            
            if use_fuzzy:
                fig.add_trace(go.Scatter(
                    x=hours_list,
                    y=predictions_fuzzy,
                    mode='lines+markers',
                    name='Fuzzy Adjusted',
                    line=dict(color='green', width=2),
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title=f"24-Hour Traffic Forecast for {selected_date}",
                xaxis_title="Hour of Day",
                yaxis_title="Number of Vehicles",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.markdown("### 📊 Forecast Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Peak Traffic", f"{max(predictions_fuzzy if use_fuzzy else predictions_gru):.0f}")
            with col2:
                st.metric("Low Traffic", f"{min(predictions_fuzzy if use_fuzzy else predictions_gru):.0f}")
            with col3:
                st.metric("Average", f"{np.mean(predictions_fuzzy if use_fuzzy else predictions_gru):.0f}")
            with col4:
                peak_hour = hours_list[np.argmax(predictions_fuzzy if use_fuzzy else predictions_gru)]
                st.metric("Peak Hour", f"{peak_hour}:00")
            
            # Download data
            df_forecast = pd.DataFrame({
                'Hour': hours_list,
                'GRU_Prediction': predictions_gru,
                'Fuzzy_Prediction': predictions_fuzzy
            })
            
            csv = df_forecast.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv,
                file_name=f"traffic_forecast_{selected_date}.csv",
                mime="text/csv"
            )

# Historical Analysis Mode
def historical_analysis_mode(df_junction):
    st.markdown('<div class="sub-header">📈 Historical Traffic Analysis</div>', 
                unsafe_allow_html=True)
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=df_junction['DateTime'].min().date(),
            min_value=df_junction['DateTime'].min().date(),
            max_value=df_junction['DateTime'].max().date()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=df_junction['DateTime'].max().date(),
            min_value=df_junction['DateTime'].min().date(),
            max_value=df_junction['DateTime'].max().date()
        )
    
    # Filter data
    mask = (df_junction['DateTime'].dt.date >= start_date) & (df_junction['DateTime'].dt.date <= end_date)
    df_filtered = df_junction.loc[mask].copy()
    
    if len(df_filtered) == 0:
        st.warning("No data available for selected date range.")
        return
    
    # Add time features
    df_filtered['Hour'] = df_filtered['DateTime'].dt.hour
    df_filtered['Weekday'] = df_filtered['DateTime'].dt.day_name()
    df_filtered['Date'] = df_filtered['DateTime'].dt.date
    
    # Time series plot
    st.markdown("### 📊 Traffic Volume Over Time")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df_filtered['DateTime'],
        y=df_filtered['Vehicles'],
        mode='lines',
        name='Traffic Volume',
        line=dict(color='blue', width=1)
    ))
    fig1.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Vehicles",
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Hourly patterns
    st.markdown("### ⏰ Average Traffic by Hour")
    hourly_avg = df_filtered.groupby('Hour')['Vehicles'].mean().reset_index()
    
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=hourly_avg['Hour'],
        y=hourly_avg['Vehicles'],
        marker_color='lightblue'
    ))
    fig2.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Average Vehicles",
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Weekday patterns
    st.markdown("### 📅 Average Traffic by Day of Week")
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_avg = df_filtered.groupby('Weekday')['Vehicles'].mean().reindex(weekday_order).reset_index()
    
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=weekday_avg['Weekday'],
        y=weekday_avg['Vehicles'],
        marker_color='lightgreen'
    ))
    fig3.update_layout(
        xaxis_title="Day of Week",
        yaxis_title="Average Vehicles",
        height=400
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # Statistics
    st.markdown("### 📊 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df_filtered):,}")
    with col2:
        st.metric("Average Traffic", f"{df_filtered['Vehicles'].mean():.0f}")
    with col3:
        st.metric("Max Traffic", f"{df_filtered['Vehicles'].max():.0f}")
    with col4:
        st.metric("Min Traffic", f"{df_filtered['Vehicles'].min():.0f}")

# Run app
if __name__ == "__main__":
    main()