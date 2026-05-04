# streamlit_app.py
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

st.set_page_config(page_title="NIDS Dashboard", layout="wide")
st.title("🛡️ Network Intrusion Detection System")
st.markdown("Upload a CSV file with NSL-KDD features to detect intrusions.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of uploaded data")
    st.dataframe(df.head())
    
    if st.button("Analyze Traffic"):
        # Send to Flask backend
        files = {"file": uploaded_file.getvalue()}
        try:
            response = requests.post("http://127.0.0.1:5000/api/batch_predict", files=files)
            if response.status_code == 200:
                result = response.json()
                
                # Build results dataframe
                results_df = pd.DataFrame(result['results'])
                df['prediction'] = results_df['prediction']
                df['confidence'] = results_df['confidence']
                df['class'] = results_df['class']
                
                attack_count = result['attacks']
                normal_count = result['normal']
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Flows", result['total'])
                col2.metric("Attacks Detected", attack_count, delta=None, delta_color="inverse")
                col3.metric("Normal Traffic", normal_count)
                
                if attack_count > 0:
                    st.error(f"⚠️ {attack_count} intrusions detected! Review the attack records below.")
                else:
                    st.success("✅ No threats detected. All traffic appears normal.")
                
                # Pie chart
                fig, ax = plt.subplots()
                ax.pie([attack_count, normal_count], labels=['Attack', 'Normal'], autopct='%1.1f%%', colors=['#ff4444', '#00ff88'])
                ax.set_title('Traffic Classification')
                st.pyplot(fig)
                
                # Show attack records
                st.subheader("📋 Detailed Attack Records")
                attack_records = df[df['prediction'] != 0]
                if len(attack_records) > 0:
                    st.dataframe(attack_records[['prediction', 'class', 'confidence'] + list(df.columns[:5])])
                else:
                    st.info("No attack records to display.")
                
                # Download button
                st.download_button(
                    "📥 Download Results as CSV",
                    df.to_csv(index=False),
                    "nids_results.csv",
                    "text/csv"
                )
            else:
                st.error(f"Backend error: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to Flask backend. Make sure `app.py` is running on port 5000.")