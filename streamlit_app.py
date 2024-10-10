import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from io import StringIO

# Page configuration
st.set_page_config(page_title="ระบบวิเคราะห์การผลิต", page_icon="📊", layout="wide")

# Load data with debug information
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/inuneko89/PRODUCE-DEMO-01/refs/heads/main/production_management_data.csv"
    df = pd.read_csv(url)
    
    # Debug information
    st.write("ข้อมูลคอลัมน์ที่มีในไฟล์ CSV:", df.columns.tolist())
    st.write("ตัวอย่างข้อมูล 5 แถวแรก:")
    st.write(df.head())
    
    return df

# Initialize Gemini
gemini_api_key = "AIzaSyCCQumrGPGSzDgY7_YFSSI5kFzYb-WXFB4"
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel("gemini-pro")

# Main title
st.title("📊 ระบบวิเคราะห์ข้อมูลการผลิต")

try:
    # Load data
    st.write("กำลังโหลดข้อมูล...")
    df = load_data()
    
    # Sidebar for controls
    st.sidebar.header("🎯 ตัวควบคุมแดชบอร์ด")
    
    # Date column detection and conversion
    date_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
    if not date_columns:
        # Try to identify potential date columns
        for col in df.columns:
            try:
                pd.to_datetime(df[col])
                date_columns.append(col)
            except:
                continue
    
    if date_columns:
        selected_date_column = st.sidebar.selectbox(
            "เลือกคอลัมน์วันที่",
            date_columns
        )
        df[selected_date_column] = pd.to_datetime(df[selected_date_column])
        
        # Date range selector
        st.sidebar.subheader("📅 เลือกช่วงวันที่")
        start_date = st.sidebar.date_input(
            "วันที่เริ่มต้น",
            df[selected_date_column].min()
        )
        end_date = st.sidebar.date_input(
            "วันที่สิ้นสุด",
            df[selected_date_column].max()
        )
        
        # Filter data by date
        mask = (df[selected_date_column].dt.date >= start_date) & (df[selected_date_column].dt.date <= end_date)
        filtered_df = df.loc[mask]
    else:
        filtered_df = df
        st.warning("ไม่พบคอลัมน์วันที่ในข้อมูล")
    
    # View options
    view_option = st.sidebar.selectbox(
        "📊 เลือกมุมมอง",
        ["หน้าแรก", "ดูข้อมูล", "สร้างกราฟ", "วิเคราะห์ข้อมูล"]
    )
    
    if view_option == "หน้าแรก":
        # Dashboard Overview
        col1, col2, col3 = st.columns(3)
        
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        
        with col1:
            if len(numeric_columns) > 0:
                selected_metric_1 = st.selectbox("เลือกตัวชี้วัดที่ 1", numeric_columns, key='metric1')
                total_value = filtered_df[selected_metric_1].sum()
                st.metric("📦 ยอดรวม " + selected_metric_1, f"{total_value:,.0f}")
        
        with col2:
            if len(numeric_columns) > 1:
                selected_metric_2 = st.selectbox("เลือกตัวชี้วัดที่ 2", numeric_columns, key='metric2')
                avg_value = filtered_df[selected_metric_2].mean()
                st.metric("📊 ค่าเฉลี่ย " + selected_metric_2, f"{avg_value:.2f}")
        
        with col3:
            if len(numeric_columns) > 2:
                selected_metric_3 = st.selectbox("เลือกตัวชี้วัดที่ 3", numeric_columns, key='metric3')
                max_value = filtered_df[selected_metric_3].max()
                st.metric("🔝 ค่าสูงสุด " + selected_metric_3, f"{max_value:.2f}")
        
        # Recommended Analysis
        st.subheader("📈 การวิเคราะห์ที่แนะนำ")
        
        # Time Series Analysis (if date column exists)
        if date_columns:
            selected_metric = st.selectbox("เลือกตัวชี้วัดที่ต้องการดูแนวโน้ม", numeric_columns)
            fig_trend = px.line(filtered_df, 
                              x=selected_date_column, 
                              y=selected_metric,
                              title=f'แนวโน้ม {selected_metric} ตามเวลา')
            st.plotly_chart(fig_trend, use_container_width=True)
        
        # Distribution Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            selected_metric_dist = st.selectbox("เลือกตัวชี้วัดที่ต้องการดูการกระจายตัว", numeric_columns)
            fig_dist = px.histogram(filtered_df, 
                                  x=selected_metric_dist,
                                  title=f'การกระจายตัวของ {selected_metric_dist}')
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            selected_metrics = st.multiselect("เลือกตัวชี้วัดที่ต้องการเปรียบเทียบ", 
                                            numeric_columns,
                                            default=list(numeric_columns)[:2] if len(numeric_columns) >= 2 else [])
            if len(selected_metrics) >= 2:
                fig_scatter = px.scatter(filtered_df,
                                       x=selected_metrics[0],
                                       y=selected_metrics[1],
                                       title=f'ความสัมพันธ์ระหว่าง {selected_metrics[0]} และ {selected_metrics[1]}')
                st.plotly_chart(fig_scatter, use_container_width=True)
        
    elif view_option == "ดูข้อมูล":
        st.subheader("📋 ดูข้อมูล")
        
        # Column selection
        selected_columns = st.multiselect(
            "เลือกคอลัมน์ที่ต้องการแสดง",
            filtered_df.columns.tolist(),
            default=filtered_df.columns.tolist()[:5]
        )
        
        # Number of rows to display
        n_rows = st.slider("จำนวนแถวที่ต้องการแสดง", 5, 50, 10)
        
        # Display selected data
        st.dataframe(filtered_df[selected_columns].head(n_rows))
        
        # Basic statistics
        if st.checkbox("แสดงสถิติพื้นฐาน"):
            st.write(filtered_df[selected_columns].describe())
            
    elif view_option == "สร้างกราฟ":
        st.subheader("📊 สร้างกราฟ")
        
        # Graph type selection
        graph_type = st.selectbox(
            "เลือกประเภทกราฟ",
            ["กราฟเส้น", "กราฟแท่ง", "กราฟกระจาย", "แผนภาพกล่อง", "ฮิสโตแกรม"]
        )
        
        # Column selection for X and Y axis
        x_column = st.selectbox("เลือกแกน X", filtered_df.columns.tolist())
        y_column = st.selectbox("เลือกแกน Y", filtered_df.columns.tolist())
        
        # Optional color grouping
        color_column = st.selectbox(
            "เลือกกลุ่มสี (ไม่จำเป็น)",
            ["ไม่มี"] + filtered_df.columns.tolist()
        )
        
        # Create graph based on selection
        if graph_type == "กราฟเส้น":
            fig = px.line(filtered_df, x=x_column, y=y_column, 
                         color=None if color_column == "ไม่มี" else color_column,
                         title=f"{y_column} vs {x_column}")
            
        elif graph_type == "กราฟแท่ง":
            fig = px.bar(filtered_df, x=x_column, y=y_column,
                        color=None if color_column == "ไม่มี" else color_column,
                        title=f"{y_column} by {x_column}")
            
        elif graph_type == "กราฟกระจาย":
            fig = px.scatter(filtered_df, x=x_column, y=y_column,
                           color=None if color_column == "ไม่มี" else color_column,
                           title=f"{y_column} vs {x_column}")
            
        elif graph_type == "แผนภาพกล่อง":
            fig = px.box(filtered_df, x=x_column, y=y_column,
                        color=None if color_column == "ไม่มี" else color_column,
                        title=f"Box Plot of {y_column} by {x_column}")
            
        elif graph_type == "ฮิสโตแกรม":
            fig = px.histogram(filtered_df, x=x_column,
                             color=None if color_column == "ไม่มี" else color_column,
                             title=f"Histogram of {x_column}")
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:  # วิเคราะห์ข้อมูล
        st.subheader("🔍 วิเคราะห์ข้อมูล")
        
        # Text area for custom query
        query = st.text_area(
            "ใส่คำถามที่ต้องการวิเคราะห์:",
            "ตัวอย่าง: แนวโน้มการผลิตเป็นอย่างไร? มีข้อเสนอแนะอะไรบ้าง?"
        )
        
        if st.button("วิเคราะห์"):
            try:
                # Create context for AI
                data_summary = filtered_df.describe().to_string()
                prompt = f"""
                วิเคราะห์ข้อมูลต่อไปนี้:
                {data_summary}
                
                คำถาม: {query}
                
                กรุณาตอบเป็นภาษาไทย โดยใช้ข้อมูลประกอบการวิเคราะห์ พร้อมให้ข้อเสนอแนะที่เป็นประโยชน์
                """
                
                response = model.generate_content(prompt)
                st.write("ผลการวิเคราะห์:")
                st.write(response.text)
                
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")
    
    # Download option
    if st.sidebar.button("ดาวน์โหลดข้อมูลเป็น CSV"):
        csv = filtered_df.to_csv(index=False)
        st.sidebar.download_button(
            label="คลิกเพื่อดาวน์โหลด",
            data=csv,
            file_name="production_data.csv",
            mime="text/csv"
        )

except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการโหลดข้อมูล: {str(e)}")
    st.write("กรุณาตรวจสอบการเชื่อมต่อข้อมูลและลองใหม่อีกครั้ง")