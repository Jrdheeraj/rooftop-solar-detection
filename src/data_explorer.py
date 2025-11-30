import pandas as pd
import numpy as np
from pathlib import Path


class DataExplorer:
    def __init__(self, xlsx_path):
        self.df = pd.read_excel(xlsx_path)
        self.xlsx_path = xlsx_path
    
    def explore(self):
        """Print comprehensive data exploration"""
        print("=" * 60)
        print("TRAINING DATA EXPLORATION")
        print("=" * 60)
        
        # Basic info
        print(f"\n📊 DATASET SIZE:")
        print(f"   Total records: {len(self.df)}")
        print(f"   Columns: {list(self.df.columns)}")
        
        # Sample data
        print(f"\n📋 FIRST 5 ROWS:")
        print(self.df.head())
        
        # Data types
        print(f"\n📝 DATA TYPES:")
        print(self.df.dtypes)
        
        # Missing values
        print(f"\n❓ MISSING VALUES:")
        print(self.df.isnull().sum())
        
        # Solar panel distribution
        print(f"\n☀️ SOLAR PANEL DISTRIBUTION:")
        solar_counts = self.df['has_solar'].value_counts()
        for label, count in solar_counts.items():
            percentage = (count / len(self.df)) * 100
            status = "✅ HAS PANEL" if label == 1 else "❌ NO PANEL"
            print(f"   {status}: {count} ({percentage:.1f}%)")
        
        # Geographic distribution
        print(f"\n🌍 GEOGRAPHIC DISTRIBUTION:")
        print(f"   Latitude range: {self.df['latitude'].min():.4f} to {self.df['latitude'].max():.4f}")
        print(f"   Longitude range: {self.df['longitude'].min():.4f} to {self.df['longitude'].max():.4f}")
        print(f"   Mean latitude: {self.df['latitude'].mean():.4f}")
        print(f"   Mean longitude: {self.df['longitude'].mean():.4f}")
        
        # Sample IDs (FIXED - handles both 'sample_id' and 'sampleid')
        print(f"\n🔢 SAMPLE IDs:")
        
        # Detect column name (flexible for both formats)
        id_column = 'sample_id' if 'sample_id' in self.df.columns else 'sampleid'
        
        print(f"   Min: {self.df[id_column].min()}")
        print(f"   Max: {self.df[id_column].max()}")
        print(f"   Total unique: {self.df[id_column].nunique()}")
        
        print("\n" + "=" * 60)


# Run exploration
if __name__ == '__main__':
    explorer = DataExplorer('data/raw/EI_train_data.xlsx')
    explorer.explore()