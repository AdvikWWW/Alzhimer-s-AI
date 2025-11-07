#!/usr/bin/env python3
"""
Phase 2: Data Validation
Validates the dataset and features before training
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
METADATA_DIR = DATA_DIR / "metadata"


class DataValidator:
    """
    Validates dataset quality and feature extraction
    """
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.stats = {}
    
    def validate_metadata(self, metadata_path: Path) -> bool:
        """
        Validate metadata file
        """
        print("="*70)
        print("VALIDATING METADATA")
        print("="*70)
        
        if not metadata_path.exists():
            self.issues.append(f"Metadata file not found: {metadata_path}")
            return False
        
        df = pd.read_csv(metadata_path)
        
        # Check required columns
        required_cols = ['file_id', 'filename', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            self.issues.append(f"Missing required columns: {missing_cols}")
            return False
        
        # Check for duplicates
        duplicates = df['file_id'].duplicated().sum()
        if duplicates > 0:
            self.warnings.append(f"Found {duplicates} duplicate file_ids")
        
        # Check label distribution
        label_counts = df['label'].value_counts()
        print(f"\n📊 Label Distribution:")
        for label, count in label_counts.items():
            print(f"   {label}: {count}")
        
        # Check for class imbalance
        if len(label_counts) == 2:
            ratio = max(label_counts) / min(label_counts)
            if ratio > 2.0:
                self.warnings.append(f"Class imbalance detected (ratio: {ratio:.2f})")
        
        self.stats['total_samples'] = len(df)
        self.stats['label_distribution'] = label_counts.to_dict()
        
        print(f"\n✅ Metadata validation passed")
        return True
    
    def validate_features(self, features_path: Path) -> bool:
        """
        Validate feature file
        """
        print("\n" + "="*70)
        print("VALIDATING FEATURES")
        print("="*70)
        
        if not features_path.exists():
            self.issues.append(f"Features file not found: {features_path}")
            return False
        
        df = pd.read_csv(features_path)
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]
        
        if len(cols_with_missing) > 0:
            self.warnings.append(f"Found {len(cols_with_missing)} columns with missing values")
            print(f"\n⚠️  Columns with missing values:")
            for col, count in cols_with_missing.items():
                print(f"   {col}: {count} missing ({count/len(df)*100:.1f}%)")
        
        # Check for infinite values
        feature_cols = [col for col in df.columns if col not in ['label', 'file_id', 'filename', 'file_path']]
        inf_counts = np.isinf(df[feature_cols]).sum()
        cols_with_inf = inf_counts[inf_counts > 0]
        
        if len(cols_with_inf) > 0:
            self.warnings.append(f"Found {len(cols_with_inf)} columns with infinite values")
        
        # Check for zero variance features
        variances = df[feature_cols].var()
        zero_var_cols = variances[variances == 0].index.tolist()
        
        if len(zero_var_cols) > 0:
            self.warnings.append(f"Found {len(zero_var_cols)} features with zero variance")
            print(f"\n⚠️  Zero variance features (will be removed during training):")
            for col in zero_var_cols[:10]:  # Show first 10
                print(f"   {col}")
        
        # Feature statistics
        print(f"\n📊 Feature Statistics:")
        print(f"   Total features: {len(feature_cols)}")
        print(f"   Total samples: {len(df)}")
        print(f"   Missing values: {df[feature_cols].isnull().sum().sum()}")
        print(f"   Infinite values: {np.isinf(df[feature_cols]).sum().sum()}")
        print(f"   Zero variance features: {len(zero_var_cols)}")
        
        self.stats['total_features'] = len(feature_cols)
        self.stats['zero_variance_features'] = len(zero_var_cols)
        
        print(f"\n✅ Feature validation passed")
        return True
    
    def check_data_quality(self, features_path: Path):
        """
        Check overall data quality
        """
        print("\n" + "="*70)
        print("CHECKING DATA QUALITY")
        print("="*70)
        
        df = pd.read_csv(features_path)
        feature_cols = [col for col in df.columns if col not in ['label', 'file_id', 'filename', 'file_path']]
        
        # Check feature ranges
        print("\n📊 Feature Ranges (first 10 features):")
        for col in feature_cols[:10]:
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            print(f"   {col}: [{min_val:.2f}, {max_val:.2f}], mean={mean_val:.2f}")
        
        # Check for outliers (simple method: > 3 std from mean)
        outlier_counts = {}
        for col in feature_cols:
            mean = df[col].mean()
            std = df[col].std()
            outliers = ((df[col] < mean - 3*std) | (df[col] > mean + 3*std)).sum()
            if outliers > 0:
                outlier_counts[col] = outliers
        
        if len(outlier_counts) > 0:
            print(f"\n⚠️  Features with outliers (>3 std from mean):")
            for col, count in list(outlier_counts.items())[:10]:
                print(f"   {col}: {count} outliers ({count/len(df)*100:.1f}%)")
        
        self.stats['features_with_outliers'] = len(outlier_counts)
    
    def visualize_features(self, features_path: Path, output_dir: Path = None):
        """
        Create visualizations of features
        """
        print("\n" + "="*70)
        print("CREATING VISUALIZATIONS")
        print("="*70)
        
        if output_dir is None:
            output_dir = FEATURES_DIR / "visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.read_csv(features_path)
        feature_cols = [col for col in df.columns if col not in ['label', 'file_id', 'filename', 'file_path']]
        
        # 1. Label distribution
        plt.figure(figsize=(8, 6))
        df['label'].value_counts().plot(kind='bar')
        plt.title('Label Distribution')
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(output_dir / 'label_distribution.png')
        plt.close()
        print(f"✅ Saved: label_distribution.png")
        
        # 2. Feature correlation heatmap (first 20 features)
        plt.figure(figsize=(12, 10))
        corr = df[feature_cols[:20]].corr()
        sns.heatmap(corr, cmap='coolwarm', center=0, square=True, linewidths=0.5)
        plt.title('Feature Correlation Heatmap (First 20 Features)')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_correlation.png')
        plt.close()
        print(f"✅ Saved: feature_correlation.png")
        
        # 3. Feature distributions (first 9 features)
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, col in enumerate(feature_cols[:9]):
            df.boxplot(column=col, by='label', ax=axes[i])
            axes[i].set_title(col)
            axes[i].set_xlabel('')
        
        plt.suptitle('Feature Distributions by Label')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_distributions.png')
        plt.close()
        print(f"✅ Saved: feature_distributions.png")
        
        print(f"\n📁 Visualizations saved to: {output_dir}")
    
    def print_summary(self):
        """
        Print validation summary
        """
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        
        if len(self.issues) == 0:
            print("\n✅ No critical issues found!")
        else:
            print(f"\n❌ Found {len(self.issues)} critical issues:")
            for issue in self.issues:
                print(f"   - {issue}")
        
        if len(self.warnings) > 0:
            print(f"\n⚠️  Found {len(self.warnings)} warnings:")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        print(f"\n📊 Dataset Statistics:")
        for key, value in self.stats.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for k, v in value.items():
                    print(f"      {k}: {v}")
            else:
                print(f"   {key}: {value}")
        
        print("\n" + "="*70)
        
        if len(self.issues) == 0:
            print("✅ DATASET IS READY FOR TRAINING!")
        else:
            print("❌ PLEASE FIX ISSUES BEFORE TRAINING")
        
        print("="*70)


def main():
    """
    Main validation function
    """
    print("="*70)
    print("PHASE 2: DATA VALIDATION")
    print("="*70)
    
    validator = DataValidator()
    
    # Validate metadata
    metadata_path = METADATA_DIR / 'dataset_info.csv'
    metadata_valid = validator.validate_metadata(metadata_path)
    
    # Validate features
    features_path = FEATURES_DIR / 'features.csv'
    features_valid = validator.validate_features(features_path)
    
    if features_valid:
        # Check data quality
        validator.check_data_quality(features_path)
        
        # Create visualizations
        try:
            validator.visualize_features(features_path)
        except Exception as e:
            print(f"\n⚠️  Could not create visualizations: {e}")
    
    # Print summary
    validator.print_summary()
    
    if metadata_valid and features_valid and len(validator.issues) == 0:
        print("\n🎉 Ready to proceed to Phase 3: SVM Model Training!")


if __name__ == "__main__":
    main()
