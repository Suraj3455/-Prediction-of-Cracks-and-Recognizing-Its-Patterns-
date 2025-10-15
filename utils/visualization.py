import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List

def create_confidence_chart(predictions: np.ndarray, class_names: List[str]):
    """
    Create a confidence score visualization chart.
    
    Args:
        predictions: Array of prediction probabilities
        class_names: List of class names
    
    Returns:
        Plotly figure object
    """
    # Convert to percentages
    probabilities = predictions * 100
    
    # Create color mapping
    colors = ['#ff4444', '#44ff44', '#ffaa44', '#4444ff']
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=class_names,
        y=probabilities,
        marker_color=colors,
        text=[f'{p:.1f}%' for p in probabilities],
        textposition='auto',
        textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': 'Crack Detection Confidence Scores',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title='Crack Type',
        yaxis_title='Confidence (%)',
        yaxis=dict(range=[0, 100]),
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_prediction_overlay(image_array: np.ndarray, prediction_class: str, confidence: float):
    """
    Create a prediction overlay visualization.
    
    Args:
        image_array: Original image array
        prediction_class: Predicted class name
        confidence: Confidence score
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add image
    fig.add_trace(go.Image(z=image_array))
    
    # Add text annotation
    fig.add_annotation(
        x=0.05,
        y=0.95,
        xref="paper",
        yref="paper",
        text=f"<b>{prediction_class}</b><br>Confidence: {confidence:.1%}",
        showarrow=False,
        font=dict(size=14, color="white"),
        bgcolor="rgba(0,0,0,0.7)",
        bordercolor="white",
        borderwidth=1
    )
    
    fig.update_layout(
        title="Prediction Overlay",
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        height=400,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

def create_uncertainty_gauge(entropy: float):
    """
    Create an uncertainty gauge visualization.
    
    Args:
        entropy: Entropy value representing uncertainty
    
    Returns:
        Plotly figure object
    """
    # Normalize entropy to 0-100 scale (approximate)
    normalized_uncertainty = min(entropy * 50, 100)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = normalized_uncertainty,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Prediction Uncertainty"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgray"},
                {'range': [25, 50], 'color': "gray"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(height=300)
    
    return fig

def create_comparison_chart(current_prediction: dict, history: List[dict]):
    """
    Create a comparison chart with prediction history.
    
    Args:
        current_prediction: Current prediction results
        history: List of previous predictions
    
    Returns:
        Plotly figure object
    """
    if not history:
        return None
    
    # Extract data from history
    timestamps = [pred['timestamp'] for pred in history[-10:]]  # Last 10 predictions
    confidences = [pred['confidence'] for pred in history[-10:]]
    classes = [pred['predicted_class'] for pred in history[-10:]]
    
    # Create line chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=confidences,
        mode='lines+markers',
        name='Confidence',
        text=classes,
        hovertemplate='<b>%{text}</b><br>Confidence: %{y:.1%}<br>Time: %{x}<extra></extra>',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Prediction History',
        xaxis_title='Time',
        yaxis_title='Confidence',
        yaxis=dict(tickformat='.1%'),
        height=300,
        showlegend=False
    )
    
    return fig
