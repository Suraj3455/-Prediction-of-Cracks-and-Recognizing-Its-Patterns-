import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
from utils.model_utils import load_model, predict_crack
from utils.image_utils import preprocess_image, validate_image
from utils.visualization import create_confidence_chart, create_prediction_overlay
from utils.pdf_generator import generate_pdf_report

st.set_page_config(
    page_title="Geopolymer Concrete Crack Detection",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'model' not in st.session_state:
    st.session_state.model = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []

def adjust_image(image, brightness=1.0, contrast=1.0):
    """Adjust image brightness and contrast."""
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
    return image

def process_single_image(image, image_name, brightness=1.0, contrast=1.0):
    """Process a single image and return prediction results."""
    adjusted_image = adjust_image(image.copy(), brightness, contrast)
    processed_image = preprocess_image(adjusted_image)
    predictions, predicted_class, confidence = predict_crack(st.session_state.model, processed_image)
    
    return {
        'image_name': image_name,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'predictions': predictions[0],
        'timestamp': datetime.now(),
        'original_image': image,
        'adjusted_image': adjusted_image
    }

def export_to_csv(results):
    """Export results to CSV format."""
    df_data = []
    class_names = ['Longitudinal Crack', 'No Crack', 'Oblique Crack', 'Transverse Crack']
    
    for result in results:
        row = {
            'Image': result['image_name'],
            'Predicted Class': result['predicted_class'],
            'Confidence': f"{result['confidence']:.2%}",
            'Timestamp': result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        }
        for i, name in enumerate(class_names):
            row[f'{name} Probability'] = f"{result['predictions'][i]:.2%}"
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    return df.to_csv(index=False)

def main():
    st.title("🏗️ Geopolymer Concrete Crack Detection")
    st.markdown("**Advanced Machine Learning System for Crack Pattern Recognition in Concrete Beams**")
    
    if st.session_state.model is None:
        with st.spinner("Loading trained model..."):
            try:
                st.session_state.model = load_model()
                st.success("✅ Model loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
                st.stop()
    
    with st.sidebar:
        st.header("📊 Model Information")
        st.info("""
        **Crack Types Detected:**
        - Longitudinal Cracks
        - Oblique Cracks  
        - Transverse Cracks
        - No Crack
        """)
        
        st.header("📈 Statistics")
        total_predictions = len(st.session_state.prediction_history)
        st.metric("Total Predictions", total_predictions)
        
        if st.session_state.prediction_history:
            crack_counts = {}
            for pred in st.session_state.prediction_history:
                crack_type = pred['predicted_class']
                crack_counts[crack_type] = crack_counts.get(crack_type, 0) + 1
            
            st.subheader("Crack Distribution")
            for crack_type, count in crack_counts.items():
                st.text(f"{crack_type}: {count}")
    
    tabs = st.tabs(["🔍 Single Image", "📦 Batch Processing", "📊 History & Reports", "⚙️ Image Preprocessing"])
    
    with tabs[0]:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("🖼️ Image Upload")
            uploaded_file = st.file_uploader(
                "Choose a concrete beam image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload high-quality images of concrete beams for crack detection",
                key="single_upload"
            )
            
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file)
                    is_valid, message = validate_image(image)
                    if not is_valid:
                        st.error(f"❌ {message}")
                    else:
                        st.image(image, caption="Uploaded Image", use_column_width=True)
                        st.subheader("📏 Image Details")
                        st.write(f"**Dimensions:** {image.size[0]} x {image.size[1]} pixels")
                        st.write(f"**Format:** {image.format}")
                except Exception as e:
                    st.error(f"❌ Error loading image: {str(e)}")
                    uploaded_file = None
        
        with col2:
            st.header("🔍 Prediction Results")
            
            if uploaded_file is not None:
                with st.spinner("Analyzing image for crack patterns..."):
                    try:
                        result = process_single_image(image, uploaded_file.name)
                        
                        if result['predicted_class'] == "No Crack":
                            st.success(f"✅ **{result['predicted_class']}** (Confidence: {result['confidence']:.1%})")
                        else:
                            st.warning(f"⚠️ **{result['predicted_class']}** detected (Confidence: {result['confidence']:.1%})")
                        
                        st.session_state.prediction_history.append(result)
                        
                        st.subheader("📊 Detailed Classification Results")
                        class_names = ['Longitudinal Crack', 'No Crack', 'Oblique Crack', 'Transverse Crack']
                        results_df = pd.DataFrame({
                            'Crack Type': class_names,
                            'Probability': result['predictions'] * 100
                        }).sort_values('Probability', ascending=False)
                        
                        st.dataframe(
                            results_df.style.format({'Probability': '{:.2f}%'}),
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
            else:
                st.info("👆 Upload an image to see prediction results")
        
        if uploaded_file is not None and 'result' in locals():
            st.header("📈 Confidence Score Visualization")
            class_names = ['Longitudinal Crack', 'No Crack', 'Oblique Crack', 'Transverse Crack']
            fig = create_confidence_chart(result['predictions'], class_names)
            st.plotly_chart(fig, use_container_width=True)
            
            st.header("🔬 Analysis Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Predicted Class", result['predicted_class'])
            with col2:
                st.metric("Confidence Level", f"{result['confidence']:.1%}")
            with col3:
                st.metric("Max Probability", f"{np.max(result['predictions']):.1%}")
            with col4:
                entropy = -np.sum(result['predictions'] * np.log(result['predictions'] + 1e-15))
                st.metric("Uncertainty", f"{entropy:.3f}")
            
            st.subheader("🚨 Risk Assessment")
            if result['predicted_class'] != "No Crack":
                if result['confidence'] > 0.8:
                    st.error("🔴 **HIGH RISK**: High confidence crack detection - Immediate inspection recommended")
                elif result['confidence'] > 0.6:
                    st.warning("🟡 **MEDIUM RISK**: Moderate confidence crack detection - Further analysis advised")
                else:
                    st.info("🟢 **LOW RISK**: Low confidence detection - Monitor for changes")
            else:
                st.success("✅ **NO RISK**: No cracks detected in the concrete beam")
    
    with tabs[1]:
        st.header("📦 Batch Image Processing")
        st.markdown("Upload multiple images for simultaneous crack detection analysis")
        
        uploaded_files = st.file_uploader(
            "Choose multiple concrete beam images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key="batch_upload"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} images uploaded")
            
            if st.button("🚀 Process All Images", type="primary"):
                batch_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    try:
                        image = Image.open(file)
                        is_valid, message = validate_image(image)
                        if is_valid:
                            result = process_single_image(image, file.name)
                            batch_results.append(result)
                            st.session_state.prediction_history.append(result)
                    except Exception as e:
                        st.warning(f"Error processing {file.name}: {str(e)}")
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                st.session_state.batch_results = batch_results
                status_text.text("✅ Batch processing complete!")
                st.success(f"Successfully processed {len(batch_results)} images")
        
        if st.session_state.batch_results:
            st.subheader("📊 Batch Results Summary")
            
            summary_data = []
            for result in st.session_state.batch_results:
                summary_data.append({
                    'Image': result['image_name'],
                    'Predicted Class': result['predicted_class'],
                    'Confidence': result['confidence']
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df.style.format({'Confidence': '{:.1%}'}), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(summary_df, names='Predicted Class', title='Crack Type Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(summary_df, x='Image', y='Confidence', color='Predicted Class',
                           title='Confidence Scores by Image')
                st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                csv_data = export_to_csv(st.session_state.batch_results)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv_data,
                    file_name=f"crack_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col2:
                pdf_buffer = generate_pdf_report(st.session_state.batch_results, "Batch Processing Report")
                st.download_button(
                    label="📄 Download Report (PDF)",
                    data=pdf_buffer.getvalue(),
                    file_name=f"crack_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
    
    with tabs[2]:
        st.header("📊 Prediction History & Analysis Reports")
        
        if st.session_state.prediction_history:
            st.subheader(f"📈 All Predictions ({len(st.session_state.prediction_history)} total)")
            
            history_data = []
            for pred in st.session_state.prediction_history:
                history_data.append({
                    'Image': pred['image_name'],
                    'Predicted Class': pred['predicted_class'],
                    'Confidence': pred['confidence'],
                    'Timestamp': pred['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                })
            
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df.style.format({'Confidence': '{:.1%}'}), use_container_width=True)
            
            st.subheader("📊 Statistical Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                crack_distribution = history_df['Predicted Class'].value_counts()
                fig = px.pie(values=crack_distribution.values, names=crack_distribution.index,
                           title='Overall Crack Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                avg_conf = history_df.groupby('Predicted Class')['Confidence'].mean()
                fig = px.bar(x=avg_conf.index, y=avg_conf.values,
                           title='Average Confidence by Crack Type',
                           labels={'x': 'Crack Type', 'y': 'Average Confidence'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(history_df))),
                    y=history_df['Confidence'],
                    mode='lines+markers',
                    name='Confidence'
                ))
                fig.update_layout(title='Confidence Trend Over Time',
                                xaxis_title='Prediction Number',
                                yaxis_title='Confidence')
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📝 Detailed Report")
            
            total_cracks = len([p for p in st.session_state.prediction_history if p['predicted_class'] != 'No Crack'])
            crack_rate = total_cracks / len(st.session_state.prediction_history) * 100
            
            st.markdown(f"""
            ### Summary Statistics
            - **Total Images Analyzed:** {len(st.session_state.prediction_history)}
            - **Images with Cracks:** {total_cracks} ({crack_rate:.1f}%)
            - **Images without Cracks:** {len(st.session_state.prediction_history) - total_cracks}
            - **Average Confidence:** {history_df['Confidence'].mean():.1%}
            - **Most Common Crack Type:** {history_df['Predicted Class'].mode()[0]}
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                csv_data = export_to_csv(st.session_state.prediction_history)
                st.download_button(
                    label="📥 Download Full History (CSV)",
                    data=csv_data,
                    file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col2:
                pdf_buffer = generate_pdf_report(st.session_state.prediction_history, "Complete Prediction History Report")
                st.download_button(
                    label="📄 Download Report (PDF)",
                    data=pdf_buffer.getvalue(),
                    file_name=f"prediction_history_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.session_state.batch_results = []
                st.rerun()
        else:
            st.info("No predictions yet. Upload and analyze images to build your prediction history.")
    
    with tabs[3]:
        st.header("⚙️ Image Preprocessing Options")
        st.markdown("Adjust image properties for better crack detection")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload & Adjust Image")
            
            preprocess_file = st.file_uploader(
                "Choose an image to preprocess",
                type=['png', 'jpg', 'jpeg'],
                key="preprocess_upload"
            )
            
            if preprocess_file:
                original_image = Image.open(preprocess_file)
                
                st.image(original_image, caption="Original Image", use_column_width=True)
                
                brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
                contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
                
                adjusted_image = adjust_image(original_image, brightness, contrast)
                
                st.subheader("Adjusted Image")
                st.image(adjusted_image, caption=f"Brightness: {brightness}, Contrast: {contrast}", 
                        use_column_width=True)
        
        with col2:
            st.subheader("Prediction with Preprocessing")
            
            if preprocess_file:
                if st.button("🔍 Analyze Adjusted Image", type="primary"):
                    with st.spinner("Analyzing preprocessed image..."):
                        try:
                            result = process_single_image(original_image, preprocess_file.name, brightness, contrast)
                            
                            if result['predicted_class'] == "No Crack":
                                st.success(f"✅ **{result['predicted_class']}** (Confidence: {result['confidence']:.1%})")
                            else:
                                st.warning(f"⚠️ **{result['predicted_class']}** detected (Confidence: {result['confidence']:.1%})")
                            
                            st.session_state.prediction_history.append(result)
                            
                            class_names = ['Longitudinal Crack', 'No Crack', 'Oblique Crack', 'Transverse Crack']
                            results_df = pd.DataFrame({
                                'Crack Type': class_names,
                                'Probability': result['predictions'] * 100
                            }).sort_values('Probability', ascending=False)
                            
                            st.dataframe(
                                results_df.style.format({'Probability': '{:.2f}%'}),
                                use_container_width=True
                            )
                            
                            fig = create_confidence_chart(result['predictions'], class_names)
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"❌ Error during prediction: {str(e)}")
            else:
                st.info("👆 Upload an image above to see preprocessing options")

if __name__ == "__main__":
    main()
