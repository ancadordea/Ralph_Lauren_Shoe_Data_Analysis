import pandas as pd
import re
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import openpyxl  # noqa: F401

def load_and_filter_data(filepath):
    """Load data and filter for Ralph Lauren products with brand variation detection"""
    df = pd.read_csv(filepath)
    
    df['brand'].str.extract(r'(ralph.*lauren|lauren.*ralph|polo.*ralph|rlx|lauren by)', 
                                              flags=re.IGNORECASE)[0].dropna().unique()
    
    pattern = r'\b(ralph\s*lauren|lauren\s*ralph|polo\s*ralph|rlx|lauren\s*by)\b'
    mask = df['brand'].str.contains(pattern, case=False, na=False)
    ralph_df = df[mask].copy()
    
    return ralph_df

def clean_data(df):
    """Clean the Ralph Lauren dataset by handling missing values and formatting columns"""
    df = df.copy()
    
    # Price columns
    if 'prices.amountMax' in df.columns and 'prices.amountMin' in df.columns:
        df['price_avg'] = (df['prices.amountMax'] + df['prices.amountMin']) / 2
        df['prices.amountMax'] = pd.to_numeric(df['prices.amountMax'], errors='coerce')
        df['prices.amountMin'] = pd.to_numeric(df['prices.amountMin'], errors='coerce')
    
    # Date columns
    date_cols = ['dateAdded', 'dateUpdated', 'prices.dateAdded', 'prices.dateSeen']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Reviews
    if 'reviews' in df.columns:
        try:
            df['reviews'] = df['reviews'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
            df['review_count'] = df['reviews'].apply(len)
            df['avg_rating'] = df['reviews'].apply(lambda x: np.mean([rev.get('rating', 0) for rev in x]) if x else 0)
        except Exception: 
            print("Could not parse reviews column")
    
    # Sizes
    if 'sizes' in df.columns:
        try:
            df['sizes'] = df['sizes'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
        except Exception:
            print("Could not parse sizes column")
    
    return df

def analyse_prices(df):
    """Analyse price distribution of Ralph Lauren shoes"""
    plt.figure(figsize=(12, 6))
    sns.histplot(df['price_avg'], bins=20, kde=True)
    plt.title('Distribution of Ralph Lauren Shoe Prices')
    plt.xlabel('Average Price ($)')
    plt.ylabel('Count')
    plt.savefig('Results/price_distribution.png')
    
    price_stats = df['price_avg'].describe()
    print("\nPrice Statistics:")
    print(price_stats)
    
    return price_stats

def analyse_reviews(df):
    """Analyse product reviews if available"""
    plt.figure(figsize=(12, 6))
    sns.histplot(df['avg_rating'], bins=5, discrete=True)
    plt.title('Distribution of Product Ratings')
    plt.xlabel('Average Rating (1-5)')
    plt.ylabel('Count')
    plt.xticks(range(1, 6))
    plt.savefig('Results/reviews.png')
    
    rating_stats = df['avg_rating'].describe()
    print("\nRating Statistics:")
    print(rating_stats)
    
    return rating_stats

def analyse_colors(df):
    """Analyse color distribution of Ralph Lauren shoes"""
    all_colors = df['colors'].str.split(',').explode()
    all_colors = all_colors.str.strip().str.lower()
    color_counts = all_colors.value_counts().head(10)
    
    plt.figure(figsize=(12, 6))
    color_counts.plot(kind='bar')
    plt.title('Top 10 Colors for Ralph Lauren Shoes')
    plt.xlabel('Color')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.savefig('Results/colors.png')
    
    print("\nTop Colors:")
    print(color_counts)
    
    return color_counts

def price_vs_rating(df):
    """Analyse relationship between price and ratings"""    
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='price_avg', y='avg_rating', alpha=0.6)
    plt.title('Price vs. Rating for Ralph Lauren Shoes')
    plt.xlabel('Average Price ($)')
    plt.ylabel('Average Rating')
    plt.savefig('Results/price_rating.png')
    
    correlation = df[['price_avg', 'avg_rating']].corr().iloc[0, 1]
    print(f"\nCorrelation between price and rating: {correlation:.2f}")
    
    return correlation

def analyse_customer_journey(df):
    """Analyse digital customer journey metrics and save results to CSV"""
    results = {}
    customer_dir = os.path.join('Results', 'customer_journey')
    os.makedirs(customer_dir, exist_ok=True)
    
    # Product view to purchase conversion
    if 'prices.dateSeen' in df.columns and 'dateAdded' in df.columns:
        df['days_to_purchase'] = (pd.to_datetime(df['prices.dateSeen']) - 
                                 pd.to_datetime(df['dateAdded'])).dt.days
        results['avg_days_to_purchase'] = df['days_to_purchase'].mean()
        
        days_distribution = df['days_to_purchase'].describe().to_frame().T
        days_distribution.to_csv('Results/customer_journey/customer_journey_days_distribution.csv', index=False)
    
    # Price elasticity analysis
    if 'price_avg' in df.columns and 'review_count' in df.columns:
        price_elasticity = df[['price_avg', 'review_count']].corr().iloc[0,1]
        results['price_elasticity'] = price_elasticity
        
        price_bins = pd.qcut(df['price_avg'], q=5)
        elasticity_by_segment = df.groupby(price_bins)['review_count'].mean().reset_index()
        elasticity_by_segment.columns = ['price_range', 'avg_review_count']
        elasticity_by_segment.to_csv('Results/customer_journey/price_elasticity_by_segment.csv', index=False)
    
    summary_df = pd.DataFrame.from_dict(results, orient='index', columns=['value'])
    summary_df.reset_index(inplace=True)
    summary_df.columns = ['metric', 'value']
    summary_df.to_csv('Results/customer_journey/customer_journey_summary.csv', index=False)
    
    return results

def analyse_marketing_performance(df):
    """Analyse marketing campaign performance and save results to CSV"""
    analysis = {}
    marketing_dir = os.path.join('Results', 'marketing_performance')
    os.makedirs(marketing_dir, exist_ok=True)
    
    # Seasonal trends
    if 'dateAdded_year' in df.columns and 'dateAdded_month' in df.columns:
        seasonal_trends = df.groupby(['dateAdded_year', 'dateAdded_month'])['price_avg'].agg(['count', 'mean'])
        seasonal_trends.columns = ['products_added', 'avg_price']
        seasonal_trends.reset_index().to_csv(
            os.path.join(marketing_dir, 'seasonal_trends.csv'), 
            index=False
        )
        analysis['seasonal_trends'] = seasonal_trends
    
    # Promotional effectiveness
    if 'prices.isSale' in df.columns:
        promo_effect = df.groupby('prices.isSale').agg(
            avg_price=('price_avg', 'mean'),
            units_sold=('id', 'count'),
            avg_rating=('avg_rating', 'mean')
        )
        promo_effect.reset_index().to_csv(
            os.path.join(marketing_dir, 'promotional_effectiveness.csv'),
            index=False
        )
        analysis['promotional_effectiveness'] = promo_effect
    
    # Shipping policy impact
    if 'prices.shipping' in df.columns:
        shipping_impact = df.groupby('prices.shipping').agg(
            avg_price=('price_avg', 'mean'),
            count=('id', 'count')
        )
        shipping_impact.reset_index().to_csv(
            os.path.join(marketing_dir, 'shipping_impact.csv'),
            index=False
        )
        analysis['shipping_impact'] = shipping_impact
    
    # Save summary metrics
    if analysis:
        summary_data = []
        for metric, df_result in analysis.items():
            if isinstance(df_result, pd.DataFrame):
                summary_data.append({
                    'metric': metric,
                    'rows': len(df_result),
                    'columns': ', '.join(df_result.columns)
                })
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(
            os.path.join(marketing_dir, 'marketing_performance_summary.csv'),
            index=False
        )
    
    return analysis

def competitive_analysis(raw_df, cleaned_df):
    """Compare Ralph Lauren women's shoes against Skechers and Steve Madden"""
    os.makedirs('Results/competitive_analysis', exist_ok=True)
    
    # Define brand patterns (case-insensitive)
    brand_patterns = {
        'Ralph Lauren': r'\b(ralph\s*lauren|lauren\s*ralph|polo\s*ralph|rlx|lauren\s*by)\b',
        'Skechers': r'\b(skechers|sketchers|skx|skecher)\b',
        'Steve Madden': r'\b(steve\s*madden|madden\s*steve|stevemadden|stven\s*madn|mad\s*steve)\b'
    }
    
    # Clean competitor data
    competitors = {}
    for brand, pattern in brand_patterns.items():
        if brand == 'Ralph Lauren':
            competitors[brand] = cleaned_df
        else:
            mask = raw_df['brand'].str.contains(pattern, case=False, na=False)
            competitors[brand] = clean_data(raw_df[mask].copy())  # Clean competitor data
    
    # Price comparison table
    price_data = []
    for brand, df in competitors.items():
        if len(df) > 0:
            price_data.append({
                'Brand': brand,
                'Avg_Price': df['price_avg'].mean(),
                'Min_Price': df['price_avg'].min(),
                'Max_Price': df['price_avg'].max(),
                'Product_Count': len(df)
            })
    
    price_comparison = pd.DataFrame(price_data)
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=price_comparison, x='Brand', y='Avg_Price', 
                    order=price_comparison.sort_values('Avg_Price', ascending=False)['Brand'])
    
    # Add price labels
    for p in ax.patches:
        ax.annotate(f"${p.get_height():.0f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    
    plt.title('Average Price Comparison (Women\'s Shoes)')
    plt.ylabel('Price ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('Results/competitive_analysis/price_comparison.png')
    plt.close()
    
    price_comparison.to_csv('Results/competitive_analysis/price_comparison.csv', index=False)
    
    # Key insights
    print("\n Competitive Analysis Results")
    print(price_comparison[['Brand', 'Avg_Price', 'Product_Count']].to_markdown(index=False))
    
    rl_price = price_comparison[price_comparison['Brand'] == 'Ralph Lauren']['Avg_Price'].values[0]
    
    for _, row in price_comparison[price_comparison['Brand'] != 'Ralph Lauren'].iterrows():
        ratio = rl_price / row['Avg_Price']
        print(f"\n• Ralph Lauren is {ratio:.1f}x more expensive than {row['Brand']}")
        print(f"  ({row['Brand']} avg: ${row['Avg_Price']:.0f} vs RL: ${rl_price:.0f})")
    
    return price_comparison

def generate_digital_commerce_report(df, output_dir='Results'):
    """Generate focused digital commerce report using available data columns"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Summary Dashboard
    fig, ax = plt.subplots(2, 2, figsize=(18, 12))
    
    # Price distribution
    sns.histplot(df['price_avg'], bins=20, kde=True, ax=ax[0,0])
    ax[0,0].set_title('Price Distribution')
    
    # Color distribution
    if 'colors' in df.columns:
        top_colors = df['colors'].str.split(',').explode().str.strip().value_counts().head(10)
        top_colors.plot(kind='bar', ax=ax[0,1])
        ax[0,1].set_title('Top 10 Colors')
        ax[0,1].tick_params(axis='x', rotation=45)
    else:
        ax[0,1].axis('off')
    
    # Size availability
    if 'sizes' in df.columns:
        try:
            sizes = df['sizes'].explode().value_counts().head(15)
            sizes.plot(kind='bar', ax=ax[1,0])
            ax[1,0].set_title('Size Availability')
            ax[1,0].tick_params(axis='x', rotation=45)
        except Exception:
            ax[1,0].axis('off')
    else:
        ax[1,0].axis('off')
    
    # Price vs date added
    if 'dateAdded' in df.columns:
        df['dateAdded'] = pd.to_datetime(df['dateAdded'])
        monthly_avg = df.resample('M', on='dateAdded')['price_avg'].mean()
        monthly_avg.plot(ax=ax[1,1])
        ax[1,1].set_title('Monthly Price Trends')
    else:
        ax[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/executive_dashboard.png')
    plt.close()
    
    # Generate xlsx report with available metrics
    with pd.ExcelWriter(f'{output_dir}/digital_commerce_report.xlsx') as writer:
        # Basic product stats
        product_stats = df.groupby('name').agg({
            'price_avg': ['mean', 'min', 'max'],
            'prices.isSale': lambda x: (x).mean() if 'prices.isSale' in df.columns else None
        })
        product_stats.columns = ['_'.join(col).strip() for col in product_stats.columns.values]
        product_stats.sort_values('price_avg_mean', ascending=False).to_excel(
            writer, sheet_name='Product Price Ranges'
        )
        
        # Color analysis
        if 'colors' in df.columns:
            color_analysis = df['colors'].str.split(',').explode().str.strip().value_counts()
            color_analysis.to_excel(writer, sheet_name='Color Analysis')
        
        # Size analysis
        if 'sizes' in df.columns:
            try:
                size_analysis = df['sizes'].explode().value_counts()
                size_analysis.to_excel(writer, sheet_name='Size Analysis')
            except Exception:
                pass
        
        # Price trends by season
        if 'dateAdded' in df.columns:
            seasonal_prices = df.groupby(df['dateAdded'].dt.month)['price_avg'].agg(['mean', 'count'])
            seasonal_prices.index.name = 'Month'
            seasonal_prices.to_excel(writer, sheet_name='Seasonal Prices')
    
    print(f"Commerce report generated in {output_dir} using available data columns")

def save_analysis_results(df, output_path):
    """Save the cleaned and analysed data to a new CSV file"""
    try:
        df.to_csv(output_path, index=False)
        print(f"Analysis results saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving results: {e}")
        return False
    
if __name__ == "__main__":
    os.makedirs('Results', exist_ok=True)

    raw_df = pd.read_csv('raw_data.csv')  

    input_file = 'raw_data.csv'  
    output_file = 'ralph_lauren_shoe_analysis.csv'

    ralph_df = load_and_filter_data(input_file)
    cleaned_df = clean_data(ralph_df)

    price_stats = analyse_prices(cleaned_df)
    review_stats = analyse_reviews(cleaned_df)
    color_analysis = analyse_colors(cleaned_df)
    price_rating_corr = price_vs_rating(cleaned_df)

    costumer_journery = analyse_customer_journey(cleaned_df)
    marketing_performance = analyse_marketing_performance(cleaned_df)
    competitive_results = competitive_analysis(raw_df, cleaned_df)

    generate_digital_commerce_report(cleaned_df, output_dir='Results')

    save_success = save_analysis_results(cleaned_df, output_file)
    
    # Print summary
    print(f"Total Ralph Lauren products analysed: {len(cleaned_df)}")
    if price_stats is not None:
        print(f"\nPrice Statistics:\n{price_stats}")
    if review_stats is not None:
        print(f"\nReview Statistics:\n{review_stats}")
    if color_analysis is not None:
        print(f"\nTop Colors:\n{color_analysis}")
    if price_rating_corr is not None:
        print(f"\nPrice-Rating Correlation: {price_rating_corr:.2f}")
    if save_success:
        print(f"\nResults saved to: {output_file}")