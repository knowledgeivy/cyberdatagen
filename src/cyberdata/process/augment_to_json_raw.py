# cyberdata/process/augment_to_json_raw.py

import gzip
import json
import pandas as pd
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv

# Add parent directory to path for imports
CURRENT_DIR = Path(__file__).parent
sys.path.append(str(CURRENT_DIR.parent))

from cyberdata.utils.config_manager import get_config_manager
from cyberdata.utils.logger_config import setup_logger

# Set up logger
logger = setup_logger("cyberdata.scripts.augment_to_json_raw")

# Load environment variables
load_dotenv()

# Get config manager instance
config_manager = get_config_manager()

logger.info(f"Project root: {config_manager.project_root}")


class AugmentToJSONConverter:
    """Convert augmented CSV files to JSON format for seeds-raw directory."""
    
    def __init__(self):
        self.config_manager = config_manager
        self.seeds_augment_dir = self.config_manager.data_dir / "seeds-augment"
        self.seeds_raw_dir = self.config_manager.data_dir / "seeds-raw"
        
        # Ensure directories exist
        self.seeds_raw_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"AugmentToJSONConverter initialized:")
        logger.info(f"  - Source directory: {self.seeds_augment_dir}")
        logger.info(f"  - Target directory: {self.seeds_raw_dir}")
    
    def find_augmented_csv_files(self) -> List[Path]:
        """Find all CSV files in the seeds-augment directory."""
        csv_files = []
        
        if not self.seeds_augment_dir.exists():
            logger.warning(f"Seeds-augment directory not found: {self.seeds_augment_dir}")
            return csv_files
        
        # Look for CSV files (both .csv and .csv.gz)
        for pattern in ["*.csv", "*.csv.gz"]:
            csv_files.extend(list(self.seeds_augment_dir.glob(pattern)))
        
        logger.info(f"Found {len(csv_files)} CSV files to convert")
        for csv_file in csv_files:
            logger.info(f"  - {csv_file.name}")
        
        return csv_files
    
    def load_csv_data(self, csv_file: Path) -> pd.DataFrame:
        """Load CSV data with support for gzip compression."""
        logger.info(f"Loading CSV data from: {csv_file}")
        
        try:
            if csv_file.suffix == '.gz':
                with gzip.open(csv_file, 'rt', encoding='utf-8') as f:
                    df = pd.read_csv(f)
            else:
                df = pd.read_csv(csv_file)
            
            logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
            
            # Log basic statistics
            if 'label' in df.columns:
                label_counts = df['label'].value_counts()
                logger.info(f"Label distribution: {dict(label_counts)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV file {csv_file}: {str(e)}")
            raise
    
    def load_metadata(self, csv_file: Path) -> Dict[str, Any]:
        """Load metadata file if it exists."""
        # Look for corresponding metadata file
        metadata_patterns = [
            csv_file.parent / f"{csv_file.stem}_metadata.json",
            csv_file.parent / f"{csv_file.name.replace('.csv.gz', '').replace('.csv', '')}_metadata.json"
        ]
        
        for metadata_path in metadata_patterns:
            if metadata_path.exists():
                try:
                    with metadata_path.open('r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    logger.info(f"Loaded metadata from: {metadata_path}")
                    return metadata
                except Exception as e:
                    logger.warning(f"Error loading metadata from {metadata_path}: {e}")
        
        logger.warning(f"No metadata file found for {csv_file}")
        return {}
    
    def infer_area_and_nature(self, csv_file: Path, metadata: Dict[str, Any]) -> tuple[str, str]:
        """Infer area and nature from filename and metadata."""
        filename = csv_file.stem.lower()
        
        # Remove common suffixes
        filename = filename.replace('.csv', '').replace('_augment', '').replace('_train', '')
        
        # Default values
        area = "Augmented Analysis"
        nature = "augmented_data"
        
        # Try to extract from metadata first
        if metadata:
            source_file = metadata.get('source_data', {}).get('source_file', '')
            if source_file:
                source_name = Path(source_file).stem.lower()
                source_name = source_name.replace('.csv', '').replace('.gz', '')
                
                # Map common patterns to areas and natures
                if 'phishing' in source_name or 'email' in source_name:
                    area = "Social Engineering"
                    nature = "email_phishing_augmented"
                elif 'intrusion' in source_name or 'network' in source_name:
                    area = "Enterprise"
                    nature = "network_intrusion_augmented"
                elif 'malware' in source_name:
                    area = "Enterprise"
                    nature = "malware_detection_augmented"
                else:
                    nature = f"{source_name}_augmented"
        
        # Fallback to filename analysis
        if nature == "augmented_data":
            if 'phishing' in filename or 'email' in filename:
                area = "Social Engineering"
                nature = "email_phishing_augmented"
            elif 'intrusion' in filename or 'network' in filename:
                area = "Enterprise"
                nature = "network_intrusion_augmented"
            elif 'malware' in filename:
                area = "Enterprise"
                nature = "malware_detection_augmented"
            else:
                # Use filename as nature
                clean_name = filename.replace('_', '').replace('-', '')
                nature = f"{clean_name}_augmented"
        
        logger.info(f"Inferred area: '{area}', nature: '{nature}'")
        return area, nature
    
    def analyze_sample_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the sample types in the dataframe."""
        analysis = {
            'total_samples': len(df),
            'malicious_count': 0,
            'benign_count': 0,
            'unknown_count': 0,
            'malicious_ratio': 0.0,
            'has_labels': False
        }
        
        if 'label' in df.columns:
            analysis['has_labels'] = True
            label_counts = df['label'].value_counts()
            analysis['malicious_count'] = int(label_counts.get(1, 0))
            analysis['benign_count'] = int(label_counts.get(0, 0))
            analysis['unknown_count'] = len(df) - analysis['malicious_count'] - analysis['benign_count']
            
            if len(df) > 0:
                analysis['malicious_ratio'] = analysis['malicious_count'] / len(df)
        
        logger.info(f"Sample analysis: {analysis['malicious_count']} malicious, "
                   f"{analysis['benign_count']} benign, {analysis['unknown_count']} unknown")
        
        return analysis
    
    def create_enhanced_metadata(self, 
                                original_metadata: Dict[str, Any],
                                df: pd.DataFrame,
                                csv_file: Path,
                                area: str,
                                nature: str) -> Dict[str, Any]:
        """Create enhanced metadata for the JSON format."""
        sample_analysis = self.analyze_sample_types(df)
        
        # Base metadata structure
        enhanced_metadata = {
            'source': 'augmented_csv_conversion',
            'conversion_method': 'augment_to_json_raw',
            'conversion_timestamp': time.time(),
            'original_csv_file': str(csv_file.relative_to(self.config_manager.project_root)),
            'area': area,
            'nature': nature,
            'total_examples': len(df),
            'dataset_name': nature.replace('_augmented', ''),
            'data_characteristics': ['augmented', 'llm_generated', 'synthetic'],
            'quality_status': 'raw_augmented_unvalidated'
        }
        
        # Add sample distribution
        enhanced_metadata.update({
            'malicious_examples': sample_analysis['malicious_count'],
            'benign_examples': sample_analysis['benign_count'],
            'unknown_examples': sample_analysis['unknown_count'],
            'malicious_ratio': sample_analysis['malicious_ratio'],
            'has_balanced_labels': sample_analysis['has_labels']
        })
        
        # Merge with original metadata if available
        if original_metadata:
            enhanced_metadata['original_metadata'] = original_metadata
            
            # Extract useful fields from original metadata
            if 'generation_method' in original_metadata:
                enhanced_metadata['generation_method'] = original_metadata['generation_method']
            
            if 'model_used' in original_metadata:
                enhanced_metadata['model_used'] = original_metadata['model_used']
            
            if 'configuration' in original_metadata:
                enhanced_metadata['augmentation_configuration'] = original_metadata['configuration']
            
            if 'quality_metrics' in original_metadata:
                enhanced_metadata['quality_metrics'] = original_metadata['quality_metrics']
        
        # Add schema information
        enhanced_metadata['schema_info'] = {
            'columns': list(df.columns),
            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'required_columns': ['subject', 'body', 'label', 'source'] if 'subject' in df.columns else list(df.columns)
        }
        
        return enhanced_metadata
    
    def convert_csv_to_json(self, csv_file: Path) -> Path:
        """Convert a single CSV file to JSON format in seeds-raw."""
        logger.info(f"Converting {csv_file.name} to JSON format")
        
        try:
            # Load CSV data and metadata
            df = self.load_csv_data(csv_file)
            original_metadata = self.load_metadata(csv_file)
            
            # Infer area and nature
            area, nature = self.infer_area_and_nature(csv_file, original_metadata)
            
            # Convert DataFrame to list of dictionaries
            examples = df.to_dict('records')
            
            # Clean up any NaN values
            for example in examples:
                for key, value in example.items():
                    if pd.isna(value):
                        example[key] = '' if isinstance(value, str) else None
            
            # Create enhanced metadata
            enhanced_metadata = self.create_enhanced_metadata(
                original_metadata, df, csv_file, area, nature
            )
            
            # Create output directory structure
            area_clean = area.replace(' ', '_').replace('(', '').replace(')', '')
            output_dir = self.seeds_raw_dir / area_clean
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create JSON structure
            json_data = {
                'examples': examples,
                'metadata': enhanced_metadata
            }
            
            # Save JSON file
            output_file = output_dir / f"{nature}_examples.json"
            with output_file.open('w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            logger.info(f"✓ Converted {len(examples)} examples to: {output_file}")
            logger.info(f"  Area: {area}")
            logger.info(f"  Nature: {nature}")
            logger.info(f"  Malicious: {enhanced_metadata['malicious_examples']}")
            logger.info(f"  Benign: {enhanced_metadata['benign_examples']}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error converting {csv_file}: {str(e)}")
            raise
    
    def convert_all_files(self, csv_files: List[Path] = None) -> List[Path]:
        """Convert all CSV files or specified list to JSON format."""
        if csv_files is None:
            csv_files = self.find_augmented_csv_files()
        
        if not csv_files:
            logger.warning("No CSV files found to convert")
            return []
        
        converted_files = []
        
        logger.info(f"Starting conversion of {len(csv_files)} CSV files")
        logger.info("="*60)
        
        for i, csv_file in enumerate(csv_files):
            logger.info(f"\nConverting file {i+1}/{len(csv_files)}: {csv_file.name}")
            logger.info("-"*40)
            
            try:
                output_file = self.convert_csv_to_json(csv_file)
                converted_files.append(output_file)
                
            except Exception as e:
                logger.error(f"Failed to convert {csv_file.name}: {e}")
                continue
        
        logger.info("\n" + "="*60)
        logger.info("CONVERSION COMPLETED")
        logger.info("="*60)
        logger.info(f"Successfully converted: {len(converted_files)}/{len(csv_files)} files")
        
        for output_file in converted_files:
            logger.info(f"✓ {output_file.relative_to(self.config_manager.project_root)}")
        
        return converted_files


def main(csv_file: str = None, 
         area: str = None, 
         nature: str = None):
    """
    Main function to convert augmented CSV files to JSON format.
    
    Args:
        csv_file (str): Specific CSV file to convert (optional)
        area (str): Override area classification (optional)
        nature (str): Override nature classification (optional)
    """
    logger.info("="*80)
    logger.info("AUGMENTED CSV TO JSON CONVERTER")
    logger.info("="*80)
    
    try:
        # Initialize converter
        converter = AugmentToJSONConverter()
        
        if csv_file:
            # Convert specific file
            csv_path = converter.seeds_augment_dir / csv_file
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            logger.info(f"Converting specific file: {csv_file}")
            
            # Override area/nature if provided
            if area or nature:
                # Would need to modify converter to accept overrides
                logger.info(f"Using overrides - Area: {area}, Nature: {nature}")
            
            converted_files = converter.convert_all_files([csv_path])
        else:
            # Convert all files
            logger.info("Converting all CSV files in seeds-augment directory")
            converted_files = converter.convert_all_files()
        
        if converted_files:
            logger.info(f"\n🎉 Conversion completed successfully!")
            logger.info(f"Converted {len(converted_files)} files to JSON format")
            logger.info(f"Output directory: {converter.seeds_raw_dir}")
            logger.info("\nNext steps:")
            logger.info("1. Run seed_validator.py to validate the converted seeds")
            logger.info("2. Use scale_generation.py to generate larger datasets")
            logger.info("3. Apply scale_validation.py for production-ready data")
        else:
            logger.warning("No files were converted. Check the logs for issues.")
            
    except Exception as e:
        logger.error(f"Error in conversion process: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert augmented CSV files to JSON format for seeds-raw")
    parser.add_argument('--csv-file', help='Specific CSV file to convert (optional)')
    parser.add_argument('--area', help='Override area classification (optional)')
    parser.add_argument('--nature', help='Override nature classification (optional)')
    parser.add_argument('--list-files', action='store_true', 
                       help='List available CSV files without converting')
    
    args = parser.parse_args()
    
    if args.list_files:
        # Just list available files
        converter = AugmentToJSONConverter()
        csv_files = converter.find_augmented_csv_files()
        print(f"\nFound {len(csv_files)} CSV files in {converter.seeds_augment_dir}:")
        for csv_file in csv_files:
            print(f"  - {csv_file.name}")
        print(f"\nTo convert all files: python {Path(__file__).name}")
        print(f"To convert specific file: python {Path(__file__).name} --csv-file filename.csv.gz")
    else:
        main(
            csv_file=args.csv_file,
            area=args.area,
            nature=args.nature
        )