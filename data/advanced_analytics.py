import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import json
import os

def load_fraud_patterns():
    """Load fraud patterns from JSON file"""
    try:
        with open('data/fraud_patterns.json', 'r') as f:
            return json.load(f)
    except:
        # Fallback if file not found
        return {"patterns": []}

def calculate_risk_score(transaction, user_history=None):
    """
    Calculate risk score for a transaction based on various factors
    
    Parameters:
    - transaction: dict containing transaction details
    - user_history: optional dataframe of user's transaction history
    
    Returns:
    - risk_score: float between 0 and 1
    - triggered_patterns: list of pattern IDs that were triggered
    """
    # Load fraud patterns
    fraud_patterns = load_fraud_patterns()
    
    # Base risk score
    risk_score = 0.0
    triggered_patterns = []
    
    # Check for amount anomalies
    if 'amount' in transaction:
        amount = float(transaction['amount'])
        
        # Check for unusually large amounts
        if amount > 1000:
            risk_score += 0.2
            triggered_patterns.append("FP001")
        
        # Check for round amounts
        if amount % 100 == 0 and amount > 0:
            risk_score += 0.1
            triggered_patterns.append("FP007")
    
    # Check for location anomalies
    if 'location' in transaction:
        location = transaction['location'].lower()
        
        # Check for high-risk countries/locations
        high_risk_locations = ['russia', 'nigeria', 'ukraine', 'belarus', 'north korea']
        for high_risk in high_risk_locations:
            if high_risk in location:
                risk_score += 0.3
                triggered_patterns.append("FP002")
                break
    
    # Check for merchant anomalies
    if 'merchant' in transaction:
        merchant = transaction['merchant'].lower()
        
        # Check for high-risk merchants
        high_risk_merchants = ['casino', 'betting', 'cryptocurrency', 'bitcoin']
        for high_risk in high_risk_merchants:
            if high_risk in merchant:
                risk_score += 0.2
                triggered_patterns.append("FP006")
                break
    
    # Check for time anomalies if timestamp is provided
    if 'timestamp' in transaction:
        try:
            timestamp = pd.to_datetime(transaction['timestamp'])
            hour = timestamp.hour
            
            # Transactions between 1am and 5am are higher risk
            if 1 <= hour <= 5:
                risk_score += 0.15
                triggered_patterns.append("FP008")
        except:
            pass
    
    # Check for velocity if user history is provided
    if user_history is not None and len(user_history) > 0:
        try:
            # Check for multiple transactions in short time
            recent_transactions = user_history[user_history['timestamp'] > (pd.Timestamp.now() - pd.Timedelta(hours=1))]
            if len(recent_transactions) > 3:
                risk_score += 0.25
                triggered_patterns.append("FP003")
            
            # Check for card testing pattern
            if len(recent_transactions) > 2:
                amounts = recent_transactions['amount'].tolist()
                if sorted(amounts) == amounts and len(amounts) >= 3:
                    risk_score += 0.3
                    triggered_patterns.append("FP004")
        except:
            pass
    
    # Add some randomness to simulate other factors
    risk_score += np.random.uniform(0, 0.2)
    
    # Cap risk score at 1.0
    risk_score = min(risk_score, 1.0)
    
    return risk_score, triggered_patterns

def get_pattern_details(pattern_id):
    """Get details for a specific fraud pattern"""
    fraud_patterns = load_fraud_patterns()
    
    for pattern in fraud_patterns.get('patterns', []):
        if pattern['id'] == pattern_id:
            return pattern
    
    return None

def analyze_transactions_advanced(df):
    """
    Analyze transactions with advanced risk scoring
    
    Parameters:
    - df: pandas DataFrame containing transaction data
    
    Returns:
    - df: DataFrame with added risk analysis columns
    - summary: dict with summary statistics
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Add risk score calculation
    risk_scores = []
    risk_categories = []
    triggered_patterns_list = []
    
    # Process each transaction
    for _, row in df.iterrows():
        # Convert row to dict for processing
        transaction = row.to_dict()
        
        # Calculate risk score
        risk_score, triggered_patterns = calculate_risk_score(transaction)
        risk_scores.append(risk_score)
        
        # Assign risk category
        if risk_score < 0.3:
            risk_categories.append("Low")
        elif risk_score < 0.7:
            risk_categories.append("Medium")
        else:
            risk_categories.append("High")
        
        # Store triggered patterns
        triggered_patterns_list.append(triggered_patterns)
    
    # Add columns to dataframe
    df['risk_score'] = risk_scores
    df['risk_category'] = risk_categories
    df['triggered_patterns'] = triggered_patterns_list
    
    # Generate fraud indicators
    def get_fraud_indicators(patterns):
        indicators = []
        for pattern_id in patterns:
            pattern = get_pattern_details(pattern_id)
            if pattern:
                indicators.append(pattern['name'])
        return ', '.join(indicators) if indicators else ''
    
    df['fraud_indicators'] = df['triggered_patterns'].apply(get_fraud_indicators)
    
    # Count risk categories
    risk_counts = df['risk_category'].value_counts().to_dict()
    high_count = risk_counts.get('High', 0)
    medium_count = risk_counts.get('Medium', 0)
    low_count = risk_counts.get('Low', 0)
    
    # Calculate percentages
    total = len(df)
    high_percent = round((high_count / total) * 100) if total > 0 else 0
    medium_percent = round((medium_count / total) * 100) if total > 0 else 0
    low_percent = round((low_count / total) * 100) if total > 0 else 0
    
    # Generate summary
    summary = f"{high_percent}% of transactions were high risk, {medium_percent}% medium risk, and {low_percent}% low risk."
    
    return df, {
        'total': total,
        'high_count': high_count,
        'medium_count': medium_count,
        'low_count': low_count,
        'high_percent': high_percent,
        'medium_percent': medium_percent,
        'low_percent': low_percent,
        'summary': summary
    }

def generate_risk_distribution_chart(summary):
    """Generate risk distribution pie chart"""
    fig = px.pie(
        names=["High", "Medium", "Low"],
        values=[summary["high_count"], summary["medium_count"], summary["low_count"]],
        color=["High", "Medium", "Low"],
        color_discrete_map={"High": "#ff4b4b", "Medium": "#ffa500", "Low": "#00cc96"},
        title="Risk Distribution"
    )
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    return fig

def generate_fraud_indicators_chart(df):
    """Generate fraud indicators bar chart"""
    # Count fraud indicators
    indicators = []
    for ind in df["fraud_indicators"]:
        if ind:
            indicators.extend([i.strip() for i in ind.split(",")])
    
    if not indicators:
        # Return empty figure if no indicators
        fig = go.Figure()
        fig.update_layout(
            title="No Fraud Indicators Detected",
            xaxis_title="Count",
            yaxis_title="Indicator"
        )
        return fig
    
    indicator_counts = pd.Series(indicators).value_counts().reset_index()
    indicator_counts.columns = ["Indicator", "Count"]
    
    fig = px.bar(
        indicator_counts,
        x="Count",
        y="Indicator",
        orientation="h",
        color_discrete_sequence=["#14274E"],
        title="Fraud Indicators"
    )
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    return fig

def generate_risk_trend_chart(df):
    """Generate risk trend chart by date if date column exists"""
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    
    if not date_columns:
        # Return empty figure if no date column
        fig = go.Figure()
        fig.update_layout(
            title="No Date Column Available for Trend Analysis",
            xaxis_title="Date",
            yaxis_title="Risk Score"
        )
        return fig
    
    # Use the first date column found
    date_col = date_columns[0]
    
    try:
        # Convert to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Group by date and calculate average risk score
        daily_risk = df.groupby(df[date_col].dt.date)['risk_score'].mean().reset_index()
        daily_risk.columns = ['Date', 'Average Risk Score']
        
        fig = px.line(
            daily_risk,
            x='Date',
            y='Average Risk Score',
            title="Risk Score Trend Over Time",
            markers=True
        )
        fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
        return fig
    except:
        # Return empty figure if conversion fails
        fig = go.Figure()
        fig.update_layout(
            title="Could Not Generate Trend Analysis",
            xaxis_title="Date",
            yaxis_title="Risk Score"
        )
        return fig

def generate_amount_vs_risk_chart(df):
    """Generate scatter plot of transaction amount vs risk score"""
    if 'amount' not in df.columns:
        # Return empty figure if no amount column
        fig = go.Figure()
        fig.update_layout(
            title="No Amount Column Available for Analysis",
            xaxis_title="Amount",
            yaxis_title="Risk Score"
        )
        return fig
    
    fig = px.scatter(
        df,
        x='amount',
        y='risk_score',
        color='risk_category',
        color_discrete_map={"High": "#ff4b4b", "Medium": "#ffa500", "Low": "#00cc96"},
        title="Transaction Amount vs Risk Score",
        hover_data=['transaction_id'] if 'transaction_id' in df.columns else None
    )
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    return fig

def generate_pdf_report(df, summary, output_path="transaction_analysis_report.pdf"):
    """Generate PDF report with analysis results"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        import io
        import matplotlib.pyplot as plt
        
        # Create a buffer for the plots
        risk_dist_buffer = io.BytesIO()
        fraud_ind_buffer = io.BytesIO()
        
        # Create simple matplotlib plots for PDF
        # Risk distribution pie chart
        plt.figure(figsize=(6, 4))
        plt.pie(
            [summary["high_count"], summary["medium_count"], summary["low_count"]],
            labels=["High", "Medium", "Low"],
            colors=["#ff4b4b", "#ffa500", "#00cc96"],
            autopct='%1.1f%%'
        )
        plt.title("Risk Distribution")
        plt.savefig(risk_dist_buffer, format='png', bbox_inches='tight')
        plt.close()
        
        # Fraud indicators bar chart
        indicators = []
        for ind in df["fraud_indicators"]:
            if ind:
                indicators.extend([i.strip() for i in ind.split(",")])
        
        if indicators:
            indicator_counts = pd.Series(indicators).value_counts()
            plt.figure(figsize=(8, 4))
            indicator_counts.plot(kind='barh', color="#14274E")
            plt.title("Fraud Indicators")
            plt.xlabel("Count")
            plt.tight_layout()
            plt.savefig(fraud_ind_buffer, format='png', bbox_inches='tight')
            plt.close()
        
        # Create the PDF document
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Create custom styles
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.darkblue,
            spaceAfter=12
        )
        
        heading_style = ParagraphStyle(
            'Heading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.darkblue,
            spaceAfter=6
        )
        
        normal_style = styles['Normal']
        
        # Build the document content
        content = []
        
        # Title
        content.append(Paragraph("Transaction Analysis Report", title_style))
        content.append(Spacer(1, 0.25*inch))
        
        # Summary section
        content.append(Paragraph("Summary", heading_style))
        content.append(Paragraph(f"Total Transactions: {summary['total']}", normal_style))
        content.append(Paragraph(f"High Risk Transactions: {summary['high_count']} ({summary['high_percent']}%)", normal_style))
        content.append(Paragraph(f"Medium Risk Transactions: {summary['medium_count']} ({summary['medium_percent']}%)", normal_style))
        content.append(Paragraph(f"Low Risk Transactions: {summary['low_count']} ({summary['low_percent']}%)", normal_style))
        content.append(Spacer(1, 0.25*inch))
        
        # Risk distribution chart
        content.append(Paragraph("Risk Distribution", heading_style))
        risk_dist_buffer.seek(0)
        content.append(Image(risk_dist_buffer, width=4*inch, height=3*inch))
        content.append(Spacer(1, 0.25*inch))
        
        # Fraud indicators chart
        if indicators:
            content.append(Paragraph("Fraud Indicators", heading_style))
            fraud_ind_buffer.seek(0)
            content.append(Image(fraud_ind_buffer, width=6*inch, height=3*inch))
            content.append(Spacer(1, 0.25*inch))
        
        # High risk transactions table
        content.append(Paragraph("High Risk Transactions", heading_style))
        high_risk_df = df[df['risk_category'] == 'High'].copy()
        
        if len(high_risk_df) > 0:
            # Select relevant columns
            display_cols = ['transaction_id', 'amount', 'merchant', 'risk_score', 'fraud_indicators'] 
            display_cols = [col for col in display_cols if col in high_risk_df.columns]
            
            # Prepare table data
            table_data = [display_cols]  # Header row
            for _, row in high_risk_df[display_cols].iterrows():
                table_data.append([str(row[col]) for col in display_cols])
            
            # Create table
            table = Table(table_data, repeatRows=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.darkblue),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            content.append(table)
        else:
            content.append(Paragraph("No high risk transactions found.", normal_style))
        
        # Build the PDF
        doc.build(content)
        
        return True, output_path
    except Exception as e:
        return False, str(e)

# Example usage
if __name__ == "__main__":
    # Create sample data
    data = {
        'transaction_id': [f'TX{i:06d}' for i in range(1, 11)],
        'date': pd.date_range(start='2025-04-01', periods=10),
        'amount': [120.50, 1500.00, 75.25, 2000.00, 25.00, 300.00, 450.00, 5000.00, 125.75, 1000.00],
        'merchant': ['Amazon', 'PayPal Transfer', 'Grocery Store', 'Cryptocurrency Exchange', 
                    'Coffee Shop', 'Unknown Merchant', 'Hotel Booking', 'Investment Platform', 
                    'Clothing Store', 'Online Subscription'],
        'location': ['New York, USA', 'Online', 'Chicago, USA', 'Online', 'Seattle, USA', 
                    'Moscow, Russia', 'Paris, France', 'Online', 'Miami, USA', 'Online']
    }
    
    df = pd.DataFrame(data)
    
    # Analyze transactions
    results_df, summary = analyze_transactions_advanced(df)
    
    # Print results
    print(f"Analysis complete. {summary['summary']}")
    print(f"High risk: {summary['high_count']}, Medium risk: {summary['medium_count']}, Low risk: {summary['low_count']}")
    
    # Generate PDF report
    success, message = generate_pdf_report(results_df, summary)
    if success:
        print(f"PDF report generated: {message}")
    else:
        print(f"Error generating PDF report: {message}")
