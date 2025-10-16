#!/usr/bin/env python3
"""
Movement Mapping Processor

This script processes an Excel file containing movement metadata and compares
pre-annotated movement types with human annotations to generate comparison results.

Input: Excel file with movement_metadata column and Movement1-5 columns
Output: New Excel file with comparison results (case_1, case_2, case_3) for each movement type
"""

import pandas as pd
import json
import ast
from typing import Dict, List, Any, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MovementMappingProcessor:
    def __init__(self):
        """Initialize the processor with the movement mapping dictionary."""
        self.mapping_dict = {
            "arc_left": "arc_shot",
            "arc_right": "arc_shot",
            "dolly_in": "push_in",
            "dolly_out": "pull_back",
            "pan_left": ["pan", "whip_pan"],
            "pan_right": ["pan", "whip_pan"],
            "pedestal_down": "pedestal_down",
            "pedestal_up": "pedestal_up",
            "roll_left": "roll",
            "roll_right": "roll",
            "static": "static_camera",
            "tilt_down": ["Pedestal", "Boom", "Crane Shot"],
            "tilt_up": "tilt_up",
            "truck_left": "truck_left",
            "truck_right": "truck_right",
            "undefined": None,
            "zoom_in": "dolly_zoom",
            "zoom_out": "dolly_zoom",
        }
    
    def parse_movement_metadata(self, metadata_str: Union[str, float, dict]) -> Dict[str, Any]:
        """
        Parse the movement_metadata string into a dictionary.
        
        Args:
            metadata_str: String representation of the movement metadata dictionary, or float/NaN
            
        Returns:
            Dictionary containing movement metadata
        """
        # Handle NaN or None values
        if pd.isna(metadata_str) or metadata_str is None:
            return {}
        
        # If it's already a dictionary, return it
        if isinstance(metadata_str, dict):
            return metadata_str
        
        # If it's a float (likely NaN), return empty dict
        if isinstance(metadata_str, float):
            return {}
        
        try:
            # Try to parse as JSON first
            if isinstance(metadata_str, str):
                return json.loads(metadata_str)
            else:
                return {}
        except json.JSONDecodeError:
            try:
                # Try to parse as Python literal
                return ast.literal_eval(metadata_str)
            except (ValueError, SyntaxError) as e:
                logger.warning(f"Could not parse movement_metadata: {metadata_str}. Error: {e}")
                return {}
    
    def normalize_movement_name(self, movement: str) -> str:
        """
        Normalize movement name for comparison.
        
        Args:
            movement: Movement name to normalize
            
        Returns:
            Normalized movement name
        """
        if not movement or pd.isna(movement):
            return ""
        return str(movement).strip().lower()
    
    def check_movement_match(self, pre_annotation: str, human_annotation: str) -> bool:
        """
        Check if pre-annotation matches human annotation using the mapping dictionary.
        
        Args:
            pre_annotation: Movement type from pre-annotation
            human_annotation: Movement type from human annotation
            
        Returns:
            True if they match, False otherwise
        """
        if not pre_annotation or not human_annotation or pd.isna(pre_annotation) or pd.isna(human_annotation):
            return False
        
        pre_annotation = self.normalize_movement_name(pre_annotation)
        human_annotation = self.normalize_movement_name(human_annotation)
        
        # Check if pre_annotation exists in mapping
        if pre_annotation in self.mapping_dict:
            mapped_value = self.mapping_dict[pre_annotation]
            
            # Handle different types of mapped values
            if mapped_value is None:
                return False
            elif isinstance(mapped_value, list):
                # Check if human annotation matches any value in the list
                return any(self.normalize_movement_name(val) == human_annotation for val in mapped_value)
            else:
                # Direct string comparison
                return self.normalize_movement_name(mapped_value) == human_annotation
        else:
            # Direct comparison if not in mapping
            return pre_annotation == human_annotation
    
    def process_movement_comparison(self, row: pd.Series) -> Dict[str, int]:
        """
        Process movement comparison for a single row.
        
        Args:
            row: Pandas Series representing a row from the DataFrame
            
        Returns:
            Dictionary with comparison results for each movement type
        """
        results = {}
        
        # Parse movement metadata
        movement_metadata = self.parse_movement_metadata(row.get('movement_metadata', '{}'))
        
        # Process each movement type (1-5)
        for i in range(1, 6):
            movement_type_key = f'movement_type_{i}'
            confidence_key = f'confidence_{i}'
            movement_col = f'Movement{i}'
            
            # Get pre-annotation and human annotation
            pre_annotation = movement_metadata.get(movement_type_key)
            human_annotation = row.get(movement_col)
            
            # Determine the case
            has_pre_annotation = pre_annotation is not None and pre_annotation != "" and not pd.isna(pre_annotation)
            has_human_annotation = human_annotation is not None and human_annotation != "" and not pd.isna(human_annotation)
            
            if has_pre_annotation and has_human_annotation:
                # Both have annotations - check if they match
                if self.check_movement_match(pre_annotation, human_annotation):
                    # Case 1: Both correct (Human and pre-annotation match)
                    results[f'movement_{i}_case_1'] = 1
                    results[f'movement_{i}_case_2'] = 0
                    results[f'movement_{i}_case_3'] = 0
                else:
                    # Case 2: Human is correct (pre-annotation is not matching)
                    results[f'movement_{i}_case_1'] = 0
                    results[f'movement_{i}_case_2'] = 1
                    results[f'movement_{i}_case_3'] = 0
            elif not has_pre_annotation and has_human_annotation:
                # Case 3: No pre-annotation (only Human input)
                results[f'movement_{i}_case_1'] = 0
                results[f'movement_{i}_case_2'] = 0
                results[f'movement_{i}_case_3'] = 1
            else:
                # No annotations or only pre-annotation (no human input)
                results[f'movement_{i}_case_1'] = 0
                results[f'movement_{i}_case_2'] = 0
                results[f'movement_{i}_case_3'] = 0
        
        return results
    
    def process_excel_file(self, input_file_path: str, output_file_path: str):
        """
        Process the Excel file and generate comparison results.
        
        Args:
            input_file_path: Path to the input Excel file
            output_file_path: Path to save the output Excel file
        """
        logger.info(f"Reading Excel file: {input_file_path}")
        
        try:
            # Read the Excel file
            df = pd.read_excel(input_file_path)
            logger.info(f"Loaded {len(df)} rows from Excel file")
            
            # Check if required columns exist
            required_columns = ['movement_metadata'] + [f'Movement{i}' for i in range(1, 6)]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                logger.info(f"Available columns: {list(df.columns)}")
                return
            
            # Process each row
            logger.info("Processing movement comparisons...")
            comparison_results = []
            
            for idx, row in df.iterrows():
                if idx % 100 == 0:
                    logger.info(f"Processing row {idx + 1}/{len(df)}")
                
                results = self.process_movement_comparison(row)
                comparison_results.append(results)
            
            # Convert results to DataFrame
            comparison_df = pd.DataFrame(comparison_results)
            
            # Combine with original data
            result_df = pd.concat([df, comparison_df], axis=1)
            
            # Save to Excel
            logger.info(f"Saving results to: {output_file_path}")
            result_df.to_excel(output_file_path, index=False)
            
            logger.info("Processing completed successfully!")
            
            # Print summary statistics
            self.print_summary_statistics(comparison_df)
            
        except Exception as e:
            logger.error(f"Error processing Excel file: {e}")
            raise
    
    def print_summary_statistics(self, comparison_df: pd.DataFrame):
        """
        Print summary statistics of the comparison results.
        
        Args:
            comparison_df: DataFrame containing comparison results
        """
        logger.info("\n=== SUMMARY STATISTICS ===")
        
        for i in range(1, 6):
            case_1_col = f'movement_{i}_case_1'
            case_2_col = f'movement_{i}_case_2'
            case_3_col = f'movement_{i}_case_3'
            
            if case_1_col in comparison_df.columns:
                case_1_count = comparison_df[case_1_col].sum()
                case_2_count = comparison_df[case_2_col].sum()
                case_3_count = comparison_df[case_3_col].sum()
                total = case_1_count + case_2_count + case_3_count
                
                logger.info(f"Movement {i}:")
                logger.info(f"  Case 1 (Both correct): {case_1_count} ({case_1_count/total*100:.1f}%)" if total > 0 else "  Case 1 (Both correct): 0 (0.0%)")
                logger.info(f"  Case 2 (Human correct): {case_2_count} ({case_2_count/total*100:.1f}%)" if total > 0 else "  Case 2 (Human correct): 0 (0.0%)")
                logger.info(f"  Case 3 (No pre-annotation): {case_3_count} ({case_3_count/total*100:.1f}%)" if total > 0 else "  Case 3 (No pre-annotation): 0 (0.0%)")


def main():
    """Main function to run the movement mapping processor."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # File paths - using relative paths from script location
    input_file = os.path.join(script_dir, "merged_annotated_outputs.xlsx")
    output_file = os.path.join(script_dir, "movement_comparison_results.xlsx")
    
    # Create processor instance
    processor = MovementMappingProcessor()
    
    # Process the Excel file
    processor.process_excel_file(input_file, output_file)


if __name__ == "__main__":
    main()
