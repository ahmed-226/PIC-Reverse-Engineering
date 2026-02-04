"""
Evaluation metrics for PIC Assembly-to-C Decompiler
Calculates various code quality and accuracy metrics
"""

import re
from typing import Dict, List
from difflib import SequenceMatcher
from collections import Counter


class CodeMetrics:
    """Calculate code quality metrics"""
    
    @staticmethod
    def calculate_token_accuracy(ground_truth: str, generated: str) -> float:
        """
        Calculate token-level accuracy
        
        Args:
            ground_truth: Expected output code
            generated: Generated output code
            
        Returns:
            Accuracy score (0-1)
        """
        gt_tokens = ground_truth.split()
        gen_tokens = generated.split()
        
        if len(gt_tokens) == 0:
            return 0.0
        
        matches = sum(1 for i in range(min(len(gt_tokens), len(gen_tokens)))
                     if gt_tokens[i] == gen_tokens[i])
        return matches / max(len(gt_tokens), len(gen_tokens))
    
    @staticmethod
    def calculate_code_similarity(ground_truth: str, generated: str) -> float:
        """
        Calculate sequence similarity using SequenceMatcher
        
        Args:
            ground_truth: Expected output code
            generated: Generated output code
            
        Returns:
            Similarity ratio (0-1)
        """
        return SequenceMatcher(None, ground_truth, generated).ratio()
    
    @staticmethod
    def extract_function_names(code: str) -> List[str]:
        """
        Extract function names from C code
        
        Args:
            code: C source code
            
        Returns:
            List of function names
        """
        pattern = r'\b(?:void|int|char|float|double|long|short|unsigned)\s+(\w+)\s*\('
        return re.findall(pattern, code)
    
    @staticmethod
    def calculate_function_name_accuracy(ground_truth: str, generated: str) -> float:
        """
        Check if function names match
        
        Args:
            ground_truth: Expected output code
            generated: Generated output code
            
        Returns:
            Accuracy score (0-1)
        """
        gt_funcs = set(CodeMetrics.extract_function_names(ground_truth))
        gen_funcs = set(CodeMetrics.extract_function_names(generated))
        
        if len(gt_funcs) == 0:
            return 1.0
        
        matches = len(gt_funcs & gen_funcs)
        return matches / len(gt_funcs)
    
    @staticmethod
    def calculate_keyword_accuracy(ground_truth: str, generated: str) -> float:
        """
        Calculate accuracy of C keywords
        
        Args:
            ground_truth: Expected output code
            generated: Generated output code
            
        Returns:
            Accuracy score (0-1)
        """
        keywords = [
            'if', 'else', 'while', 'for', 'return', 'void', 'int', 'char',
            'struct', 'switch', 'case', 'break', 'continue', 'unsigned',
            'volatile', 'static', 'const', 'typedef'
        ]
        
        gt_keywords = Counter([word for word in ground_truth.split() if word in keywords])
        gen_keywords = Counter([word for word in generated.split() if word in keywords])
        
        if sum(gt_keywords.values()) == 0:
            return 1.0
        
        matches = sum((gt_keywords & gen_keywords).values())
        total = sum(gt_keywords.values())
        return matches / total
    
    @staticmethod
    def calculate_structural_similarity(ground_truth: str, generated: str) -> float:
        """
        Calculate structural similarity based on braces and control flow
        
        Args:
            ground_truth: Expected output code
            generated: Generated output code
            
        Returns:
            Similarity score (0-1)
        """
        def extract_structure(code: str) -> str:
            # Remove string literals and comments
            code = re.sub(r'"[^"]*"', '""', code)
            code = re.sub(r'//.*?\n', '\n', code)
            code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
            
            # Extract only structural elements
            structure = re.sub(r'[^{}();\[\]]', '', code)
            return structure
        
        gt_structure = extract_structure(ground_truth)
        gen_structure = extract_structure(generated)
        
        return SequenceMatcher(None, gt_structure, gen_structure).ratio()


class ModelEvaluator:
    """Evaluate model performance on test set"""
    
    def __init__(self):
        self.metrics = CodeMetrics()
    
    def evaluate_single(self, ground_truth: str, generated: str) -> Dict[str, float]:
        """
        Evaluate a single example
        
        Args:
            ground_truth: Expected output
            generated: Generated output
            
        Returns:
            Dictionary of metric scores
        """
        return {
            'token_accuracy': self.metrics.calculate_token_accuracy(ground_truth, generated),
            'code_similarity': self.metrics.calculate_code_similarity(ground_truth, generated),
            'function_accuracy': self.metrics.calculate_function_name_accuracy(ground_truth, generated),
            'keyword_accuracy': self.metrics.calculate_keyword_accuracy(ground_truth, generated),
            'structural_similarity': self.metrics.calculate_structural_similarity(ground_truth, generated),
        }
    
    def evaluate_batch(self, 
                      ground_truths: List[str], 
                      generated_outputs: List[str]) -> Dict[str, List[float]]:
        """
        Evaluate multiple examples
        
        Args:
            ground_truths: List of expected outputs
            generated_outputs: List of generated outputs
            
        Returns:
            Dictionary mapping metric names to lists of scores
        """
        if len(ground_truths) != len(generated_outputs):
            raise ValueError("Number of ground truths and generated outputs must match")
        
        results = {
            'token_accuracy': [],
            'code_similarity': [],
            'function_accuracy': [],
            'keyword_accuracy': [],
            'structural_similarity': [],
        }
        
        for gt, gen in zip(ground_truths, generated_outputs):
            scores = self.evaluate_single(gt, gen)
            for metric, score in scores.items():
                results[metric].append(score)
        
        return results
    
    def calculate_aggregate_metrics(self, 
                                   batch_results: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate aggregate statistics from batch results
        
        Args:
            batch_results: Results from evaluate_batch
            
        Returns:
            Dictionary with mean, median, min, max for each metric
        """
        import numpy as np
        
        aggregate = {}
        
        for metric, scores in batch_results.items():
            aggregate[metric] = {
                'mean': float(np.mean(scores)),
                'median': float(np.median(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
            }
        
        return aggregate
    
    def compare_models(self, 
                      ground_truths: List[str],
                      model_a_outputs: List[str],
                      model_b_outputs: List[str],
                      model_a_name: str = "Model A",
                      model_b_name: str = "Model B") -> Dict:
        """
        Compare two models on the same test set
        
        Args:
            ground_truths: List of expected outputs
            model_a_outputs: Outputs from first model
            model_b_outputs: Outputs from second model
            model_a_name: Name for first model
            model_b_name: Name for second model
            
        Returns:
            Comparison results
        """
        import numpy as np
        
        # Evaluate both models
        results_a = self.evaluate_batch(ground_truths, model_a_outputs)
        results_b = self.evaluate_batch(ground_truths, model_b_outputs)
        
        # Calculate improvements
        improvements = {}
        for metric in results_a.keys():
            mean_a = np.mean(results_a[metric])
            mean_b = np.mean(results_b[metric])
            improvement = mean_b - mean_a
            improvement_pct = (improvement / mean_a * 100) if mean_a > 0 else 0
            
            improvements[metric] = {
                f'{model_a_name}_mean': mean_a,
                f'{model_b_name}_mean': mean_b,
                'improvement': improvement,
                'improvement_pct': improvement_pct,
            }
        
        return {
            'model_a_name': model_a_name,
            'model_b_name': model_b_name,
            'model_a_results': results_a,
            'model_b_results': results_b,
            'improvements': improvements,
        }
    
    def print_evaluation_report(self, 
                               aggregate_metrics: Dict[str, Dict[str, float]],
                               model_name: str = "Model"):
        """
        Print formatted evaluation report
        
        Args:
            aggregate_metrics: Results from calculate_aggregate_metrics
            model_name: Name of the model being evaluated
        """
        print("="*60)
        print(f"EVALUATION REPORT: {model_name}")
        print("="*60)
        
        for metric, stats in aggregate_metrics.items():
            metric_name = metric.replace('_', ' ').title()
            print(f"\n{metric_name}:")
            print(f"  Mean:   {stats['mean']:.4f} ({stats['mean']*100:.2f}%)")
            print(f"  Median: {stats['median']:.4f} ({stats['median']*100:.2f}%)")
            print(f"  Std:    {stats['std']:.4f}")
            print(f"  Range:  [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        print("\n" + "="*60)
    
    def print_comparison_report(self, comparison_results: Dict):
        """
        Print formatted comparison report
        
        Args:
            comparison_results: Results from compare_models
        """
        model_a = comparison_results['model_a_name']
        model_b = comparison_results['model_b_name']
        improvements = comparison_results['improvements']
        
        print("="*60)
        print(f"MODEL COMPARISON: {model_a} vs {model_b}")
        print("="*60)
        
        for metric, stats in improvements.items():
            metric_name = metric.replace('_', ' ').title()
            
            mean_a = stats[f'{model_a}_mean']
            mean_b = stats[f'{model_b}_mean']
            improvement = stats['improvement']
            improvement_pct = stats['improvement_pct']
            
            print(f"\n{metric_name}:")
            print(f"  {model_a}: {mean_a:.4f} ({mean_a*100:.2f}%)")
            print(f"  {model_b}: {mean_b:.4f} ({mean_b*100:.2f}%)")
            print(f"  Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)")
        
        print("\n" + "="*60)
        
        # Calculate overall improvement
        overall_improvement = sum(s['improvement_pct'] for s in improvements.values()) / len(improvements)
        print(f"\nOverall Average Improvement: {overall_improvement:+.2f}%")
        print("="*60)


if __name__ == "__main__":
    # Test metrics
    print("Testing evaluation metrics...")
    
    gt = """void setup() {
    PORTA = 0xFF;
    PORTB = 0x00;
    return;
}"""
    
    gen = """void setup() {
    PORTA = 0xFF;
    PORTB = 0x00;
    return;
}"""
    
    evaluator = ModelEvaluator()
    scores = evaluator.evaluate_single(gt, gen)
    
    print("\nSample evaluation:")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.4f}")
    
    print("\n✓ Metrics module ready")
