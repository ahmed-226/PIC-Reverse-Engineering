"""
Data loading and preprocessing module for PIC Assembly-to-C Decompiler
Handles JSON dataset loading, validation, prompt formatting, and .lst file parsing
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datasets import Dataset
from config import DataConfig


class DataLoader:
    """Handles dataset loading, validation, and formatting"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.valid_examples = []
        self.metadata = {}
        self.instruction_reference = {}
        self.register_map = {}
    
    def load_json_dataset(self, json_path: str) -> Tuple[Dataset, Dataset]:
        """
        Load and process the master dataset JSON file
        
        Args:
            json_path: Path to master_dataset.json
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        print(f"Loading dataset from: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract components
        training_examples = data.get("training_examples", [])
        self.instruction_reference = data.get("instruction_reference", {})
        self.register_map = data.get("register_map", {})
        self.metadata = data.get("metadata", {})
        
        print(f"Total examples in JSON: {len(training_examples)}")
        print(f"Instructions available: {self.instruction_reference.get('total_instructions', 0)}")
        print(f"Registers mapped: {self.register_map.get('total_registers', 0)}")
        print(f"Processor: {self.metadata.get('processor', 'Unknown')}")
        
        # Validate and clean examples
        self.valid_examples = self._validate_examples(training_examples)
        
        if len(self.valid_examples) == 0:
            raise ValueError("No valid examples found in dataset!")
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(self.valid_examples)
        print(f"✓ Dataset created with {len(dataset)} examples")
        
        # Format prompts
        formatted_dataset = dataset.map(self._format_prompt)
        
        # Split train/val
        train_dataset, val_dataset = self._split_dataset(formatted_dataset)
        
        print(f"Train Size: {len(train_dataset)}")
        print(f"Val Size: {len(val_dataset)}")
        print("-" * 60)
        
        return train_dataset, val_dataset
    
    def _validate_examples(self, training_examples: List[Dict]) -> List[Dict]:
        """Validate and clean training examples"""
        valid_examples = []
        invalid_count = 0
        
        for idx, example in enumerate(training_examples):
            try:
                # Check required fields
                if not all(field in example for field in ["instruction", "input", "output"]):
                    missing_fields = [f for f in ["instruction", "input", "output"] 
                                    if f not in example]
                    print(f"⚠️  Example {idx}: Missing fields - {', '.join(missing_fields)}")
                    invalid_count += 1
                    continue
                
                # Convert to strings and strip whitespace
                instruction = str(example["instruction"]).strip()
                input_code = str(example["input"]).strip()
                output_code = str(example["output"]).strip()
                
                # Check for empty strings
                if not instruction or not input_code or not output_code:
                    print(f"⚠️  Example {idx}: Empty field detected")
                    invalid_count += 1
                    continue
                
                # Extract optional fields
                function_name = example.get("function_name", "")
                context = example.get("context", {})
                
                # Convert context to string if it's a dict
                context_str = ""
                if isinstance(context, dict) and context:
                    context_parts = []
                    for key, value in context.items():
                        if value:
                            context_parts.append(f"{key}: {value}")
                    context_str = "\n".join(context_parts)
                
                # Create clean example
                clean_example = {
                    "instruction": instruction,
                    "input": input_code,
                    "output": output_code,
                    "function_name": str(function_name).strip() if function_name else "",
                    "context": context_str
                }
                
                valid_examples.append(clean_example)
                
            except Exception as e:
                print(f"⚠️  Example {idx}: Error during processing - {str(e)}")
                invalid_count += 1
                continue
        
        print(f"\n✓ Valid examples: {len(valid_examples)}")
        print(f"✗ Invalid examples: {invalid_count}")
        print("-" * 60)
        
        return valid_examples
    
    def _format_prompt(self, example: Dict) -> Dict:
        """Format example into Alpaca-style prompt"""
        # Start with instruction
        prompt = f"""### Instruction:
{example['instruction']}"""
        
        # Add function name if available
        if example.get('function_name'):
            prompt += f"\nFunction: {example['function_name']}"
        
        # Add context information if available
        if example.get('context'):
            prompt += f"""\n\n### Context:
{example['context']}"""
        
        # Add input and response
        prompt += f"""\n\n### Input:
{example['input']}\n\n### Response:
{example['output']}"""
        
        return {"text": prompt}
    
    def _split_dataset(self, dataset: Dataset) -> Tuple[Dataset, Dataset]:
        """Split dataset into train and validation sets"""
        split_ratio = getattr(self.config, 'train_val_split', 0.9)
        train_size = int(split_ratio * len(dataset))
        
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, len(dataset)))
        
        return train_dataset, val_dataset
    
    def get_system_context(self) -> str:
        """Build system context from reference materials"""
        system_context = f"""You are an expert PIC microcontroller assembly-to-C decompiler.
Processor: {self.metadata.get('processor', 'PIC16F877A')}

RULES:
"""
        # Add rules from config
        for i, rule in enumerate(self.config.system_rules, 1):
            system_context += f"{i}. {rule}\n"
        
        # Add important registers (limit to top 10)
        if "registers" in self.register_map:
            system_context += "\nKEY REGISTERS:\n"
            reg_items = list(self.register_map["registers"].items())[:10]
            for reg, desc in reg_items:
                system_context += f"- {reg}: {desc}\n"
        
        system_context += "\nYour task is to translate PIC assembly to readable, well-commented C code.\n"
        
        return system_context


class LSTFileParser:
    """Parse .lst (listing) files from PIC assemblers"""
    
    @staticmethod
    def parse_lst_file(file_path: str, strip_comments: bool = True) -> str:
        """
        Parse a .lst file and extract assembly code
        
        Args:
            file_path: Path to .lst file
            strip_comments: Whether to remove comments
            
        Returns:
            Cleaned assembly code
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        assembly_lines = []
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
            
            # Parse typical .lst format: address, opcode, label, instruction, operands, comment
            # Example: 0000 3001    MOVLW 0x01  ; Load 1
            
            # Try to extract the instruction part (usually after hex opcodes)
            # Pattern: optional address, optional opcode, then instruction
            match = re.match(r'^(?:[0-9A-Fa-f]{4}\s+)?(?:[0-9A-Fa-f]{4}\s+)?(.+)$', line)
            
            if match:
                instruction_part = match.group(1).strip()
                
                # Skip directive lines or empty results
                if not instruction_part or instruction_part.startswith(';'):
                    continue
                
                # Strip comments if requested
                if strip_comments and ';' in instruction_part:
                    instruction_part = instruction_part.split(';')[0].strip()
                
                # Skip common assembler directives
                if any(instruction_part.upper().startswith(directive) for directive in 
                      ['LIST', 'INCLUDE', '#INCLUDE', 'END', 'ORG', '__CONFIG', 
                       'PROCESSOR', 'RADIX', 'ERRORLEVEL']):
                    continue
                
                if instruction_part:
                    assembly_lines.append(instruction_part)
        
        return '\n'.join(assembly_lines)
    
    @staticmethod
    def parse_lst_directory(dir_path: str, strip_comments: bool = True) -> Dict[str, str]:
        """
        Parse all .lst files in a directory
        
        Args:
            dir_path: Path to directory containing .lst files
            strip_comments: Whether to remove comments
            
        Returns:
            Dictionary mapping filename to assembly code
        """
        directory = Path(dir_path)
        
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Invalid directory: {dir_path}")
        
        lst_files = list(directory.glob("*.lst"))
        
        if not lst_files:
            raise ValueError(f"No .lst files found in: {dir_path}")
        
        results = {}
        
        for lst_file in lst_files:
            try:
                assembly = LSTFileParser.parse_lst_file(str(lst_file), strip_comments)
                results[lst_file.name] = assembly
                print(f"✓ Parsed: {lst_file.name} ({len(assembly.splitlines())} lines)")
            except Exception as e:
                print(f"✗ Failed to parse {lst_file.name}: {e}")
        
        return results


def create_inference_prompt(assembly_code: str, 
                           function_name: str = "", 
                           context: str = "",
                           processor: str = "PIC16F877A") -> str:
    """
    Create inference prompt for assembly code
    
    Args:
        assembly_code: PIC assembly code
        function_name: Optional function name
        context: Optional context information
        processor: Processor type
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""### Instruction:
Decompile the following {processor} assembly to readable C code. Rename variables meaningfully and add comments."""
    
    if function_name:
        prompt += f"\nFunction: {function_name}"
    
    if context:
        prompt += f"\n\n### Context:\n{context}"
    
    prompt += f"""\n\n### Input:
{assembly_code}

### Response:
"""
    
    return prompt


if __name__ == "__main__":
    # Test data loader
    from config import DEFAULT_DATA_CONFIG
    
    print("Testing DataLoader...")
    loader = DataLoader(DEFAULT_DATA_CONFIG)
    
    # Test LST parser
    print("\nTesting LSTFileParser...")
    sample_lst = """
0000 3001    MOVLW 0x01  ; Load 1
0001 008F    MOVWF PORTA ; Output to PORTA
0002 3002    MOVLW 0x02
0003 008F    MOVWF PORTB
    """
    
    # Would parse actual file in production
    print("LST Parser module ready")
