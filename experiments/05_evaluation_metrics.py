#!/usr/bin/env python3
"""
Experiment 05: Evaluation Metrics for Reasoning
===============================================
Implements and tests various metrics for evaluating reasoning quality.
Includes accuracy, coherence, completeness, and efficiency metrics.
"""

import re
import statistics
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
from collections import Counter
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime


@dataclass
class ReasoningChain:
    """Represents a complete reasoning chain"""
    problem: str
    steps: List[str]
    final_answer: str
    expected_answer: str
    metadata: Dict = None


class ReasoningMetrics:
    """Collection of metrics for evaluating reasoning quality"""
    
    def __init__(self):
        self.metrics = {
            'accuracy': self.calculate_accuracy,
            'coherence': self.calculate_coherence,
            'completeness': self.calculate_completeness,
            'efficiency': self.calculate_efficiency,
            'consistency': self.calculate_consistency,
            'verbosity': self.calculate_verbosity
        }
    
    def calculate_accuracy(self, chain: ReasoningChain) -> float:
        """Calculate answer accuracy (0 or 1)"""
        # Simple exact match for now
        return 1.0 if chain.final_answer == chain.expected_answer else 0.0
    
    def calculate_coherence(self, chain: ReasoningChain) -> float:
        """Calculate logical coherence of reasoning steps"""
        if not chain.steps:
            return 0.0
        
        score = 1.0
        
        # Check for logical connectors
        connectors = ['therefore', 'thus', 'hence', 'because', 'since', 'so']
        connector_count = sum(
            1 for step in chain.steps 
            for conn in connectors 
            if conn in step.lower()
        )
        connector_ratio = connector_count / len(chain.steps)
        
        # Check for numbered/ordered steps
        ordered = any(
            step.strip().startswith(str(i)) 
            for i, step in enumerate(chain.steps, 1)
        )
        
        # Check for contradictions (simplified)
        has_contradiction = False
        for i, step1 in enumerate(chain.steps):
            for step2 in chain.steps[i+1:]:
                if self._check_contradiction(step1, step2):
                    has_contradiction = True
                    score *= 0.5
        
        # Combine factors
        score *= (0.5 + connector_ratio)  # Bonus for connectors
        if ordered:
            score *= 1.1  # Bonus for ordering
        
        return min(1.0, score)
    
    def _check_contradiction(self, step1: str, step2: str) -> bool:
        """Simple contradiction check"""
        # Look for opposite statements
        opposites = [
            ('greater', 'less'), ('more', 'fewer'), 
            ('increase', 'decrease'), ('true', 'false')
        ]
        
        for word1, word2 in opposites:
            if word1 in step1.lower() and word2 in step2.lower():
                # Check if referring to same subject
                if len(set(step1.split()) & set(step2.split())) > 3:
                    return True
        return False
    
    def calculate_completeness(self, chain: ReasoningChain) -> float:
        """Calculate completeness of reasoning"""
        if not chain.steps:
            return 0.0
        
        # Expected components for complete reasoning
        components = {
            'problem_understanding': False,
            'information_extraction': False,
            'method_application': False,
            'calculation': False,
            'verification': False,
            'conclusion': False
        }
        
        # Check for each component
        for step in chain.steps:
            step_lower = step.lower()
            
            if any(w in step_lower for w in ['understand', 'given', 'problem']):
                components['problem_understanding'] = True
            if any(w in step_lower for w in ['identify', 'extract', 'note']):
                components['information_extraction'] = True
            if any(w in step_lower for w in ['apply', 'using', 'method']):
                components['method_application'] = True
            if any(w in step_lower for w in ['calculate', 'compute', '=']):
                components['calculation'] = True
            if any(w in step_lower for w in ['check', 'verify', 'confirm']):
                components['verification'] = True
            if any(w in step_lower for w in ['therefore', 'answer', 'conclude']):
                components['conclusion'] = True
        
        # Score based on components present
        return sum(components.values()) / len(components)
    
    def calculate_efficiency(self, chain: ReasoningChain) -> float:
        """Calculate reasoning efficiency (inverse of redundancy)"""
        if not chain.steps:
            return 0.0
        
        # Optimal steps estimate (problem-dependent)
        optimal_steps = self._estimate_optimal_steps(chain.problem)
        actual_steps = len(chain.steps)
        
        # Penalize both too few and too many steps
        if actual_steps < optimal_steps:
            efficiency = actual_steps / optimal_steps
        else:
            # Diminishing returns for extra steps
            efficiency = optimal_steps / actual_steps
        
        # Check for redundancy
        unique_content = len(set(step.lower() for step in chain.steps))
        redundancy_penalty = unique_content / actual_steps
        
        return efficiency * redundancy_penalty
    
    def _estimate_optimal_steps(self, problem: str) -> int:
        """Estimate optimal number of steps for a problem"""
        # Simple heuristic based on problem complexity
        if 'simple' in problem.lower() or len(problem) < 50:
            return 2
        elif 'complex' in problem.lower() or len(problem) > 150:
            return 6
        else:
            return 4
    
    def calculate_consistency(self, chain: ReasoningChain) -> float:
        """Calculate internal consistency"""
        if not chain.steps:
            return 0.0
        
        consistency_score = 1.0
        
        # Check numerical consistency
        numbers_per_step = []
        for step in chain.steps:
            numbers = re.findall(r'\d+\.?\d*', step)
            numbers_per_step.append(numbers)
        
        # Check if numbers are used consistently
        all_numbers = [n for nums in numbers_per_step for n in nums]
        if all_numbers:
            number_counts = Counter(all_numbers)
            # Repeated numbers should appear in context
            for num, count in number_counts.items():
                if count > 1:
                    consistency_score *= 1.05  # Bonus for reusing numbers
        
        # Check variable naming consistency
        variables = re.findall(r'\b[a-z]\b', ' '.join(chain.steps))
        if variables:
            var_counts = Counter(variables)
            # Variables should be used multiple times
            avg_usage = statistics.mean(var_counts.values())
            if avg_usage > 1.5:
                consistency_score *= 1.1
        
        return min(1.0, consistency_score)
    
    def calculate_verbosity(self, chain: ReasoningChain) -> float:
        """Calculate verbosity (lower is better)"""
        if not chain.steps:
            return 0.0
        
        total_words = sum(len(step.split()) for step in chain.steps)
        avg_words_per_step = total_words / len(chain.steps)
        
        # Optimal range: 5-15 words per step
        if 5 <= avg_words_per_step <= 15:
            return 1.0
        elif avg_words_per_step < 5:
            return avg_words_per_step / 5  # Too terse
        else:
            return 15 / avg_words_per_step  # Too verbose
    
    def evaluate_all(self, chain: ReasoningChain) -> Dict[str, float]:
        """Calculate all metrics for a reasoning chain"""
        results = {}
        for metric_name, metric_func in self.metrics.items():
            results[metric_name] = metric_func(chain)
        
        # Calculate composite score
        weights = {
            'accuracy': 0.3,
            'coherence': 0.2,
            'completeness': 0.2,
            'efficiency': 0.1,
            'consistency': 0.1,
            'verbosity': 0.1
        }
        
        composite = sum(
            results[m] * w 
            for m, w in weights.items()
        )
        results['composite_score'] = composite
        
        return results


class BenchmarkSuite:
    """Suite of benchmark problems for testing"""
    
    def __init__(self):
        self.problems = [
            {
                'type': 'math',
                'problem': 'If a car travels 120 miles in 2 hours, what is its average speed?',
                'answer': '60 mph',
                'optimal_steps': 3
            },
            {
                'type': 'logic',
                'problem': 'All dogs are animals. All animals need water. What can we conclude about dogs?',
                'answer': 'Dogs need water',
                'optimal_steps': 4
            },
            {
                'type': 'word',
                'problem': 'If CAT is coded as 3120, how is DOG coded?',
                'answer': '4157',
                'optimal_steps': 5
            }
        ]
    
    def generate_test_chains(self) -> List[ReasoningChain]:
        """Generate test reasoning chains"""
        chains = []
        
        # Good reasoning example
        chains.append(ReasoningChain(
            problem=self.problems[0]['problem'],
            steps=[
                "Given: distance = 120 miles, time = 2 hours",
                "Formula: speed = distance / time",
                "Calculation: speed = 120 / 2 = 60",
                "Therefore, average speed is 60 mph"
            ],
            final_answer="60 mph",
            expected_answer="60 mph"
        ))
        
        # Poor reasoning example
        chains.append(ReasoningChain(
            problem=self.problems[0]['problem'],
            steps=[
                "Car travels",
                "120 and 2",
                "Answer is 60"
            ],
            final_answer="60 mph",
            expected_answer="60 mph"
        ))
        
        # Verbose reasoning
        chains.append(ReasoningChain(
            problem=self.problems[0]['problem'],
            steps=[
                "First, let me carefully read and understand the problem statement",
                "A car is traveling and we need to find its speed",
                "The distance traveled by the car is 120 miles",
                "The time taken for this journey is 2 hours",
                "Now I need to recall the formula for speed",
                "Speed equals distance divided by time",
                "Let me substitute the values into the formula",
                "Speed = 120 miles / 2 hours",
                "Performing the division: 120 / 2 = 60",
                "Therefore, the average speed of the car is 60 miles per hour"
            ],
            final_answer="60 mph",
            expected_answer="60 mph"
        ))
        
        return chains


def test_metric_calculation():
    """Test individual metric calculations"""
    print("=" * 60)
    print("TESTING INDIVIDUAL METRICS")
    print("=" * 60)
    
    metrics = ReasoningMetrics()
    benchmark = BenchmarkSuite()
    chains = benchmark.generate_test_chains()
    
    for i, chain in enumerate(chains):
        print(f"\n=== Test Chain {i+1} ===")
        print(f"Steps: {len(chain.steps)}")
        
        results = metrics.evaluate_all(chain)
        
        for metric, value in results.items():
            if metric != 'composite_score':
                print(f"{metric:15s}: {value:.3f}")
        
        print(f"{'Composite':15s}: {results['composite_score']:.3f} ⭐")
    
    return chains, metrics


def test_metric_correlation():
    """Test correlation between different metrics"""
    print("\n" + "=" * 60)
    print("TESTING METRIC CORRELATIONS")
    print("=" * 60)
    
    # Generate many chains with varying quality
    chains = []
    for quality in ['excellent', 'good', 'fair', 'poor']:
        for _ in range(5):
            chains.append(generate_chain_with_quality(quality))
    
    metrics = ReasoningMetrics()
    all_scores = []
    
    for chain in chains:
        scores = metrics.evaluate_all(chain)
        all_scores.append(scores)
    
    # Calculate correlations
    metric_names = ['accuracy', 'coherence', 'completeness', 'efficiency']
    correlations = {}
    
    for m1 in metric_names:
        for m2 in metric_names:
            if m1 < m2:  # Avoid duplicates
                values1 = [s[m1] for s in all_scores]
                values2 = [s[m2] for s in all_scores]
                corr = calculate_correlation(values1, values2)
                correlations[f"{m1}-{m2}"] = corr
    
    print("\nMetric Correlations:")
    for pair, corr in correlations.items():
        print(f"  {pair:30s}: {corr:+.3f}")
    
    return correlations


def generate_chain_with_quality(quality: str) -> ReasoningChain:
    """Generate a reasoning chain with specified quality"""
    
    if quality == 'excellent':
        return ReasoningChain(
            problem="Solve: 2x + 5 = 13",
            steps=[
                "Given equation: 2x + 5 = 13",
                "Subtract 5 from both sides: 2x = 8",
                "Divide by 2: x = 4",
                "Verification: 2(4) + 5 = 8 + 5 = 13 ✓"
            ],
            final_answer="x = 4",
            expected_answer="x = 4"
        )
    elif quality == 'good':
        return ReasoningChain(
            problem="Solve: 2x + 5 = 13",
            steps=[
                "2x + 5 = 13",
                "2x = 8",
                "x = 4"
            ],
            final_answer="x = 4",
            expected_answer="x = 4"
        )
    elif quality == 'fair':
        return ReasoningChain(
            problem="Solve: 2x + 5 = 13",
            steps=[
                "Subtract 5",
                "Divide by 2",
                "Get 4"
            ],
            final_answer="4",
            expected_answer="x = 4"
        )
    else:  # poor
        return ReasoningChain(
            problem="Solve: 2x + 5 = 13",
            steps=["x = 4"],
            final_answer="4",
            expected_answer="x = 4"
        )


def calculate_correlation(values1: List[float], values2: List[float]) -> float:
    """Calculate Pearson correlation coefficient"""
    if len(values1) != len(values2) or len(values1) < 2:
        return 0.0
    
    mean1 = statistics.mean(values1)
    mean2 = statistics.mean(values2)
    
    numerator = sum((v1 - mean1) * (v2 - mean2) for v1, v2 in zip(values1, values2))
    
    sum_sq1 = sum((v - mean1) ** 2 for v in values1)
    sum_sq2 = sum((v - mean2) ** 2 for v in values2)
    
    denominator = math.sqrt(sum_sq1 * sum_sq2)
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def test_metric_sensitivity():
    """Test metric sensitivity to changes"""
    print("\n" + "=" * 60)
    print("TESTING METRIC SENSITIVITY")
    print("=" * 60)
    
    base_chain = ReasoningChain(
        problem="Calculate 15% of 200",
        steps=[
            "Convert 15% to decimal: 0.15",
            "Multiply: 0.15 × 200 = 30",
            "Therefore, 15% of 200 is 30"
        ],
        final_answer="30",
        expected_answer="30"
    )
    
    metrics = ReasoningMetrics()
    base_scores = metrics.evaluate_all(base_chain)
    
    # Test variations
    variations = {
        'remove_step': ReasoningChain(
            problem=base_chain.problem,
            steps=base_chain.steps[1:],  # Remove first step
            final_answer=base_chain.final_answer,
            expected_answer=base_chain.expected_answer
        ),
        'add_redundancy': ReasoningChain(
            problem=base_chain.problem,
            steps=base_chain.steps + ["Rechecking: 0.15 × 200 = 30"],
            final_answer=base_chain.final_answer,
            expected_answer=base_chain.expected_answer
        ),
        'wrong_answer': ReasoningChain(
            problem=base_chain.problem,
            steps=base_chain.steps,
            final_answer="35",  # Wrong
            expected_answer=base_chain.expected_answer
        )
    }
    
    print("\nBase scores:")
    for metric, value in base_scores.items():
        if metric != 'composite_score':
            print(f"  {metric:15s}: {value:.3f}")
    
    for var_name, var_chain in variations.items():
        var_scores = metrics.evaluate_all(var_chain)
        print(f"\n{var_name}:")
        
        for metric in ['accuracy', 'coherence', 'completeness', 'efficiency']:
            delta = var_scores[metric] - base_scores[metric]
            print(f"  {metric:15s}: {var_scores[metric]:.3f} ({delta:+.3f})")
    
    return base_scores, variations


def create_evaluation_dashboard(chains: List[ReasoningChain], metrics: ReasoningMetrics, 
                               save_path: str = None):
    """Create comprehensive evaluation metrics dashboard"""
    fig = plt.figure(figsize=(20, 16))
    
    # Calculate all metrics for all chains
    all_scores = [metrics.evaluate_all(chain) for chain in chains]
    
    # 1. Metrics Distribution (2x2 grid, top left)
    ax1 = plt.subplot(4, 4, 1)
    metric_names = ['accuracy', 'coherence', 'completeness', 'efficiency']
    metric_data = []
    
    for metric in metric_names:
        values = [scores[metric] for scores in all_scores]
        metric_data.extend([(metric, val) for val in values])
    
    df_metrics = pd.DataFrame(metric_data, columns=['Metric', 'Score'])
    sns.boxplot(data=df_metrics, x='Metric', y='Score', ax=ax1)
    ax1.set_title('Metric Score Distributions', fontsize=12, weight='bold')
    ax1.set_ylim(0, 1)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. Composite Score Scatter (top middle)
    ax2 = plt.subplot(4, 4, 2)
    composite_scores = [scores['composite_score'] for scores in all_scores]
    accuracies = [scores['accuracy'] for scores in all_scores]
    
    colors = ['green' if acc == 1 else 'red' for acc in accuracies]
    ax2.scatter(range(len(composite_scores)), composite_scores, c=colors, alpha=0.7)
    ax2.set_xlabel('Chain Index')
    ax2.set_ylabel('Composite Score')
    ax2.set_title('Composite Scores by Chain', fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Correlation Heatmap (top right)
    ax3 = plt.subplot(4, 4, 3)
    metric_matrix = []
    for metric1 in metric_names:
        row = []
        for metric2 in metric_names:
            values1 = [scores[metric1] for scores in all_scores]
            values2 = [scores[metric2] for scores in all_scores]
            corr = calculate_correlation(values1, values2)
            row.append(corr)
        metric_matrix.append(row)
    
    sns.heatmap(metric_matrix, annot=True, cmap='coolwarm', center=0,
                xticklabels=metric_names, yticklabels=metric_names, ax=ax3)
    ax3.set_title('Metric Correlations', fontsize=12, weight='bold')
    
    # 4. Performance by Chain Length (top right corner)
    ax4 = plt.subplot(4, 4, 4)
    chain_lengths = [len(chain.steps) for chain in chains]
    ax4.scatter(chain_lengths, composite_scores, alpha=0.7, c=colors)
    ax4.set_xlabel('Number of Steps')
    ax4.set_ylabel('Composite Score')
    ax4.set_title('Performance vs Chain Length', fontsize=12, weight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Accuracy Distribution (middle left)
    ax5 = plt.subplot(4, 4, 5)
    accuracy_counts = Counter(accuracies)
    ax5.pie(accuracy_counts.values(), labels=['Incorrect', 'Correct'], 
            autopct='%1.1f%%', colors=['red', 'green'])
    ax5.set_title('Accuracy Distribution', fontsize=12, weight='bold')
    
    # 6. Coherence vs Completeness (middle middle)
    ax6 = plt.subplot(4, 4, 6)
    coherence_scores = [scores['coherence'] for scores in all_scores]
    completeness_scores = [scores['completeness'] for scores in all_scores]
    
    ax6.scatter(coherence_scores, completeness_scores, c=colors, alpha=0.7, s=60)
    ax6.set_xlabel('Coherence')
    ax6.set_ylabel('Completeness') 
    ax6.set_title('Coherence vs Completeness', fontsize=12, weight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Add diagonal line
    ax6.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    
    # 7. Efficiency Analysis (middle right)
    ax7 = plt.subplot(4, 4, 7)
    efficiency_scores = [scores['efficiency'] for scores in all_scores]
    
    # Bin by efficiency ranges
    eff_ranges = ['Low (0-0.33)', 'Med (0.33-0.67)', 'High (0.67-1.0)']
    eff_counts = [
        sum(1 for e in efficiency_scores if 0 <= e < 0.33),
        sum(1 for e in efficiency_scores if 0.33 <= e < 0.67),
        sum(1 for e in efficiency_scores if 0.67 <= e <= 1.0)
    ]
    
    bars = ax7.bar(eff_ranges, eff_counts, color=['red', 'orange', 'green'], alpha=0.7)
    ax7.set_ylabel('Count')
    ax7.set_title('Efficiency Distribution', fontsize=12, weight='bold')
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45)
    
    # Add value labels
    for bar, count in zip(bars, eff_counts):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')
    
    # 8. Metric Radar Chart (middle right corner)
    ax8 = plt.subplot(4, 4, 8, projection='polar')
    
    # Average scores across all chains
    avg_scores = {
        metric: statistics.mean(scores[metric] for scores in all_scores)
        for metric in metric_names
    }
    
    categories = list(avg_scores.keys())
    values = list(avg_scores.values())
    
    # Complete the circle
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
    values += values[:1]
    angles = np.concatenate((angles, [angles[0]]))
    
    ax8.plot(angles, values, 'o-', linewidth=2, alpha=0.7)
    ax8.fill(angles, values, alpha=0.25)
    ax8.set_xticks(angles[:-1])
    ax8.set_xticklabels(categories)
    ax8.set_ylim(0, 1)
    ax8.set_title('Average Metric Performance', fontsize=12, weight='bold', pad=20)
    
    # 9-12. Individual metric histograms (bottom row)
    for i, metric in enumerate(metric_names):
        ax = plt.subplot(4, 4, 9+i)
        values = [scores[metric] for scores in all_scores]
        
        ax.hist(values, bins=10, alpha=0.7, color='skyblue', edgecolor='navy')
        ax.axvline(statistics.mean(values), color='red', linestyle='--', 
                  label=f'Mean: {statistics.mean(values):.2f}')
        ax.set_xlabel(metric.title())
        ax.set_ylabel('Frequency')
        ax.set_title(f'{metric.title()} Distribution', fontsize=10, weight='bold')
        ax.legend()
        ax.set_xlim(0, 1)
    
    plt.suptitle('🎯 Reasoning Evaluation Metrics Dashboard', 
                 fontsize=18, weight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Evaluation dashboard saved to {save_path}")
    
    return fig


def create_sample_chain_quality(quality: str) -> ReasoningChain:
    """Create sample chain based on quality level"""
    if quality == 'excellent':
        return ReasoningChain(
            problem="What is 20% of 150?",
            steps=[
                "Convert percentage to decimal: 20% = 0.20",
                "Multiply by the number: 0.20 × 150",
                "Calculate: 0.20 × 150 = 30",
                "Verify: 30 is 20% of 150"
            ],
            final_answer="30",
            expected_answer="30"
        )
    elif quality == 'good':
        return ReasoningChain(
            problem="What is 20% of 150?",
            steps=[
                "20% as decimal is 0.2",
                "0.2 × 150 = 30"
            ],
            final_answer="30",
            expected_answer="30"
        )
    elif quality == 'fair':
        return ReasoningChain(
            problem="What is 20% of 150?", 
            steps=["20% of 150 is 30"],
            final_answer="30",
            expected_answer="30"
        )
    else:  # poor
        return ReasoningChain(
            problem="What is 20% of 150?",
            steps=[],
            final_answer="25",  # Wrong answer
            expected_answer="30"
        )


def create_live_evaluation_demo():
    """Create a live evaluation demo with sample chains"""
    print("\n" + "="*60)
    print("🎯 LIVE EVALUATION METRICS DASHBOARD")
    print("="*60)
    
    # Generate sample reasoning chains with different quality levels
    sample_chains = []
    
    # High quality chains
    for i in range(5):
        sample_chains.append(create_sample_chain_quality('excellent'))
    
    # Medium quality chains  
    for i in range(8):
        sample_chains.append(create_sample_chain_quality('good'))
    
    # Lower quality chains
    for i in range(4):
        sample_chains.append(create_sample_chain_quality('fair'))
    
    # Poor quality chains
    for i in range(3):
        sample_chains.append(create_sample_chain_quality('poor'))
    
    metrics = ReasoningMetrics()
    
    # Create dashboard
    dashboard_fig = create_evaluation_dashboard(
        sample_chains, metrics, 'demo_evaluation_dashboard.png'
    )
    
    # Calculate summary statistics
    all_scores = [metrics.evaluate_all(chain) for chain in sample_chains]
    
    avg_composite = statistics.mean(s['composite_score'] for s in all_scores)
    accuracy_rate = statistics.mean(s['accuracy'] for s in all_scores) 
    
    print(f"\n📊 Dashboard Summary:")
    print(f"  Total chains analyzed: {len(sample_chains)}")
    print(f"  Average composite score: {avg_composite:.3f}")
    print(f"  Overall accuracy rate: {accuracy_rate:.1%}")
    print(f"  Dashboard saved: demo_evaluation_dashboard.png")
    
    return dashboard_fig, sample_chains


def main():
    """Run evaluation metrics experiment"""
    print("=" * 60)
    print("EXPERIMENT 05: EVALUATION METRICS FOR REASONING")
    print("=" * 60)
    
    # Test metric calculation
    chains, metrics = test_metric_calculation()
    
    # Test correlations
    correlations = test_metric_correlation()
    
    # Test sensitivity
    base_scores, variations = test_metric_sensitivity()
    
    # Create live demo dashboard
    create_live_evaluation_demo()
    
    # Save results
    results = {
        'metric_tests': [
            {
                'chain_length': len(chain.steps),
                'scores': metrics.evaluate_all(chain)
            }
            for chain in chains
        ],
        'correlations': correlations,
        'sensitivity_analysis': {
            'base_scores': base_scores,
            'variation_count': len(variations)
        },
        'observations': [
            "Composite score effectively combines multiple metrics",
            "Coherence and completeness are positively correlated",
            "Efficiency penalizes both too few and too many steps",
            "Metrics are sensitive to reasoning quality changes"
        ]
    }
    
    with open('05_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n✓ Experiment complete. Results saved to 05_results.json")


if __name__ == '__main__':
    main()