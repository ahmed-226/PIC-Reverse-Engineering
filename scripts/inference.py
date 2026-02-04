"""
Inference script for PIC Assembly-to-C Decompiler
Loads fine-tuned model and decompiles .lst files to C code
"""

import argparse
import json
import torch
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unsloth import FastLanguageModel
from src.data_loader import LSTFileParser, create_inference_prompt
from src.config import InferenceConfig


class PICDecompiler:
    """Main decompiler class for inference"""
    
    def __init__(self, model_path: str, config: Optional[InferenceConfig] = None):
        """
        Initialize decompiler with trained model
        
        Args:
            model_path: Path to fine-tuned model directory
            config: Optional inference configuration
        """
        self.config = config or InferenceConfig()
        self.model_path = model_path
        
        print(f"Loading model from: {model_path}")
        self.model, self.tokenizer = self._load_model()
        print("✓ Model loaded successfully")
    
    def _load_model(self):
        """Load fine-tuned model for inference"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            self.model_path,
            load_in_4bit=True,
        )
        
        # Prepare for inference
        FastLanguageModel.for_inference(model)
        
        return model, tokenizer
    
    def decompile(self, 
                  assembly_code: str, 
                  function_name: str = "", 
                  context: str = "") -> str:
        """
        Decompile assembly code to C
        
        Args:
            assembly_code: PIC assembly code
            function_name: Optional function name
            context: Optional context information
            
        Returns:
            Generated C code
        """
        # Create prompt
        prompt = create_inference_prompt(
            assembly_code, 
            function_name, 
            context,
            processor="PIC16F877A"
        )
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=self.config.max_new_tokens,
                use_cache=self.config.use_cache,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
            )
        
        # Decode and extract response
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response part
        if "### Response:" in full_output:
            c_code = full_output.split("### Response:")[-1].strip()
        else:
            c_code = full_output.strip()
        
        return c_code
    
    def decompile_file(self, 
                       input_file: str, 
                       output_file: Optional[str] = None,
                       function_name: str = "",
                       context: str = "") -> Dict:
        """
        Decompile a single .lst file
        
        Args:
            input_file: Path to input .lst file
            output_file: Optional output file path (default: input_file.c)
            function_name: Optional function name
            context: Optional context
            
        Returns:
            Dictionary with results
        """
        print(f"\nProcessing: {input_file}")
        
        # Parse LST file
        try:
            assembly_code = LSTFileParser.parse_lst_file(
                input_file, 
                strip_comments=self.config.strip_comments
            )
            print(f"✓ Parsed {len(assembly_code.splitlines())} lines of assembly")
        except Exception as e:
            print(f"✗ Failed to parse file: {e}")
            return {"status": "error", "error": str(e)}
        
        # Decompile
        print("Generating C code...")
        start_time = datetime.now()
        
        try:
            c_code = self.decompile(assembly_code, function_name, context)
            elapsed_time = (datetime.now() - start_time).total_seconds()
            print(f"✓ Generated C code in {elapsed_time:.2f}s")
        except Exception as e:
            print(f"✗ Decompilation failed: {e}")
            return {"status": "error", "error": str(e)}
        
        # Save output
        if output_file:
            output_path = Path(output_file)
        else:
            input_path = Path(input_file)
            output_path = input_path.with_suffix('.c')
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(c_code)
            
            print(f"✓ Saved to: {output_path}")
        except Exception as e:
            print(f"✗ Failed to save output: {e}")
            return {"status": "error", "error": str(e)}
        
        return {
            "status": "success",
            "input_file": str(input_file),
            "output_file": str(output_path),
            "assembly_lines": len(assembly_code.splitlines()),
            "c_code_lines": len(c_code.splitlines()),
            "processing_time_seconds": elapsed_time,
            "c_code": c_code
        }
    
    def decompile_directory(self, 
                           input_dir: str, 
                           output_dir: Optional[str] = None) -> Dict[str, Dict]:
        """
        Decompile all .lst files in a directory
        
        Args:
            input_dir: Directory containing .lst files
            output_dir: Output directory (default: input_dir/output)
            
        Returns:
            Dictionary mapping filenames to results
        """
        directory = Path(input_dir)
        
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Invalid directory: {input_dir}")
        
        lst_files = list(directory.glob("*.lst"))
        
        if not lst_files:
            print(f"⚠️  No .lst files found in: {input_dir}")
            return {}
        
        print(f"\nFound {len(lst_files)} .lst files")
        print("="*60)
        
        # Setup output directory
        if output_dir:
            out_path = Path(output_dir)
        else:
            out_path = directory / "output"
        
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Process each file
        results = {}
        
        for i, lst_file in enumerate(lst_files, 1):
            print(f"\n[{i}/{len(lst_files)}] Processing: {lst_file.name}")
            
            output_file = out_path / lst_file.with_suffix('.c').name
            
            result = self.decompile_file(
                str(lst_file), 
                str(output_file)
            )
            
            results[lst_file.name] = result
        
        # Save summary
        summary_path = out_path / "decompilation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*60)
        print("BATCH PROCESSING COMPLETE")
        print("="*60)
        print(f"✓ Processed: {len(results)} files")
        print(f"✓ Summary saved to: {summary_path}")
        
        # Print statistics
        successful = sum(1 for r in results.values() if r.get('status') == 'success')
        failed = len(results) - successful
        
        print(f"\n📊 Results:")
        print(f"  • Successful: {successful}")
        print(f"  • Failed: {failed}")
        
        if successful > 0:
            total_time = sum(r.get('processing_time_seconds', 0) 
                           for r in results.values() 
                           if r.get('status') == 'success')
            avg_time = total_time / successful
            print(f"  • Average processing time: {avg_time:.2f}s")
        
        return results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="PIC Assembly-to-C Decompiler - Inference"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to fine-tuned model directory"
    )
    
    # Input arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-file",
        type=str,
        help="Path to input .lst file"
    )
    input_group.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing .lst files"
    )
    
    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        help="Output file/directory path (default: input_file.c or input_dir/output)"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["c", "json", "markdown"],
        default="c",
        help="Output format (default: c)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling (default: greedy decoding)"
    )
    
    # Optional metadata
    parser.add_argument(
        "--function-name",
        type=str,
        default="",
        help="Optional function name hint"
    )
    parser.add_argument(
        "--context",
        type=str,
        default="",
        help="Optional context information"
    )
    
    # Parsing options
    parser.add_argument(
        "--keep-comments",
        action="store_true",
        help="Keep comments from .lst file (default: strip)"
    )
    
    return parser.parse_args()


def format_output(result: Dict, output_format: str) -> str:
    """Format output based on requested format"""
    c_code = result.get('c_code', '')
    
    if output_format == "c":
        return c_code
    
    elif output_format == "json":
        return json.dumps(result, indent=2)
    
    elif output_format == "markdown":
        md = f"""# Decompilation Result

## Metadata
- **Input File**: {result.get('input_file', 'N/A')}
- **Processing Time**: {result.get('processing_time_seconds', 0):.2f}s
- **Assembly Lines**: {result.get('assembly_lines', 0)}
- **C Code Lines**: {result.get('c_code_lines', 0)}

## Generated C Code

```c
{c_code}
```
"""
        return md
    
    return c_code


def main():
    """Main inference pipeline"""
    args = parse_args()
    
    print("="*60)
    print("PIC Assembly-to-C Decompiler - Inference")
    print("="*60)
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  Running on CPU (slower)")
    
    # Create config
    config = InferenceConfig(
        model_path=args.model_path,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        strip_comments=not args.keep_comments,
    )
    
    # Initialize decompiler
    print("\n" + "="*60)
    print("INITIALIZING DECOMPILER")
    print("="*60)
    
    decompiler = PICDecompiler(args.model_path, config)
    
    # Process input
    print("\n" + "="*60)
    print("PROCESSING INPUT")
    print("="*60)
    
    try:
        if args.input_file:
            # Single file mode
            result = decompiler.decompile_file(
                args.input_file,
                args.output,
                args.function_name,
                args.context
            )
            
            if result.get('status') == 'success':
                print("\n✅ Decompilation completed successfully!")
                
                # Print preview
                if args.output_format == "c":
                    print("\n" + "="*60)
                    print("GENERATED C CODE (Preview)")
                    print("="*60)
                    c_code = result['c_code']
                    preview = c_code[:500] + ("..." if len(c_code) > 500 else "")
                    print(preview)
            else:
                print(f"\n❌ Decompilation failed: {result.get('error')}")
        
        else:
            # Directory mode
            results = decompiler.decompile_directory(
                args.input_dir,
                args.output
            )
            
            if results:
                print("\n✅ Batch decompilation completed!")
            else:
                print("\n⚠️  No files processed")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
