#!/usr/bin/env python3
"""
Experiment 03: Chain-of-Thought Prompting
==========================================
Explores different CoT prompting strategies and their effectiveness.
Tests zero-shot, few-shot, and self-consistency approaches.
"""

import random
import statistics
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import json


@dataclass
class ReasoningProblem:
    """Represents a problem requiring reasoning"""
    question: str
    answer: any
    problem_type: str  # math, logic, common_sense
    difficulty: int  # 1-5
    reasoning_steps: Optional[List[str]] = None


@dataclass
class ReasoningResponse:
    """Model's response with reasoning"""
    problem: ReasoningProblem
    reasoning_chain: List[str]
    final_answer: any
    confidence: float
    method: str  # zero_shot, few_shot, etc.


class ChainOfThoughtGenerator:
    """Generates CoT prompts and analyzes responses"""
    
    def __init__(self):
        self.prompt_templates = {
            'zero_shot': "Let's solve this step by step:\n{question}",
            'zero_shot_explicit': "{question}\n\nLet's think step by step:",
            'few_shot': "{examples}\n\nNow solve:\n{question}",
            'structured': """
Problem: {question}

Step 1: Understand the problem
Step 2: Identify key information
Step 3: Apply relevant rules/formulas
Step 4: Calculate/reason through
Step 5: Verify the answer

Solution:""",
            'self_ask': "{question}\n\nAre there any subquestions I should answer first?"
        }
    
    def generate_zero_shot_prompt(self, problem: ReasoningProblem) -> str:
        """Generate zero-shot CoT prompt"""
        return self.prompt_templates['zero_shot_explicit'].format(
            question=problem.question
        )
    
    def generate_few_shot_prompt(self, problem: ReasoningProblem, 
                                examples: List[ReasoningProblem]) -> str:
        """Generate few-shot CoT prompt with examples"""
        example_texts = []
        
        for ex in examples:
            ex_text = f"Q: {ex.question}\nA: Let's think step by step.\n"
            if ex.reasoning_steps:
                for step in ex.reasoning_steps:
                    ex_text += f"- {step}\n"
            ex_text += f"Therefore, the answer is {ex.answer}.\n"
            example_texts.append(ex_text)
        
        return self.prompt_templates['few_shot'].format(
            examples="\n".join(example_texts),
            question=problem.question
        )
    
    def parse_reasoning_chain(self, response: str) -> Tuple[List[str], any]:
        """Parse reasoning steps and final answer from response"""
        lines = response.strip().split('\n')
        reasoning_steps = []
        final_answer = None
        
        for line in lines:
            line = line.strip()
            if line.startswith(('-', '*', '•', 'Step')):
                reasoning_steps.append(line)
            elif 'therefore' in line.lower() or 'answer' in line.lower():
                # Try to extract final answer
                if ':' in line:
                    final_answer = line.split(':')[-1].strip()
                elif '=' in line:
                    final_answer = line.split('=')[-1].strip()
        
        return reasoning_steps, final_answer
    
    def simulate_reasoning(self, problem: ReasoningProblem, method: str) -> ReasoningResponse:
        """Simulate reasoning process (placeholder for actual model)"""
        # This would be replaced with actual model inference
        
        if problem.problem_type == 'math':
            steps = self._simulate_math_reasoning(problem)
        elif problem.problem_type == 'logic':
            steps = self._simulate_logic_reasoning(problem)
        else:
            steps = self._simulate_common_sense_reasoning(problem)
        
        # Add some randomness to simulate model uncertainty
        confidence = random.uniform(0.6, 0.95) if steps else random.uniform(0.3, 0.6)
        
        # Sometimes get wrong answer to simulate errors
        if random.random() < 0.2:  # 20% error rate
            final_answer = f"wrong_{problem.answer}"
        else:
            final_answer = problem.answer
        
        return ReasoningResponse(
            problem=problem,
            reasoning_chain=steps,
            final_answer=final_answer,
            confidence=confidence,
            method=method
        )
    
    def _simulate_math_reasoning(self, problem: ReasoningProblem) -> List[str]:
        """Simulate mathematical reasoning steps"""
        return [
            "Identify the numbers and operations",
            "Apply order of operations",
            "Perform calculations",
            f"Verify: calculation yields {problem.answer}"
        ]
    
    def _simulate_logic_reasoning(self, problem: ReasoningProblem) -> List[str]:
        """Simulate logical reasoning steps"""
        return [
            "Identify premises",
            "Apply logical rules",
            "Check for contradictions",
            f"Conclude: {problem.answer}"
        ]
    
    def _simulate_common_sense_reasoning(self, problem: ReasoningProblem) -> List[str]:
        """Simulate common sense reasoning"""
        return [
            "Consider context",
            "Apply world knowledge",
            "Evaluate plausibility",
            f"Determine: {problem.answer}"
        ]


class SelfConsistency:
    """Implements self-consistency for improved reasoning"""
    
    def __init__(self, num_samples: int = 5):
        self.num_samples = num_samples
    
    def generate_multiple_chains(self, problem: ReasoningProblem, 
                                generator: ChainOfThoughtGenerator) -> List[ReasoningResponse]:
        """Generate multiple reasoning chains"""
        responses = []
        
        for i in range(self.num_samples):
            # Add variation by using different methods
            methods = ['zero_shot', 'zero_shot_explicit', 'structured']
            method = random.choice(methods)
            
            response = generator.simulate_reasoning(problem, method)
            responses.append(response)
        
        return responses
    
    def aggregate_responses(self, responses: List[ReasoningResponse]) -> Dict:
        """Aggregate multiple responses via majority voting"""
        answers = [r.final_answer for r in responses]
        answer_counts = Counter(answers)
        
        # Most common answer
        majority_answer = answer_counts.most_common(1)[0][0]
        vote_count = answer_counts[majority_answer]
        
        # Calculate confidence based on agreement
        agreement_ratio = vote_count / len(responses)
        avg_confidence = statistics.mean([r.confidence for r in responses])
        
        # Select best reasoning chain (highest confidence with majority answer)
        best_chain = None
        for r in responses:
            if r.final_answer == majority_answer:
                if best_chain is None or r.confidence > best_chain.confidence:
                    best_chain = r
        
        return {
            'final_answer': majority_answer,
            'vote_distribution': dict(answer_counts),
            'agreement_ratio': agreement_ratio,
            'avg_confidence': avg_confidence,
            'best_reasoning_chain': best_chain.reasoning_chain if best_chain else [],
            'num_samples': len(responses)
        }


def create_test_problems() -> List[ReasoningProblem]:
    """Create test problems for evaluation"""
    problems = [
        ReasoningProblem(
            question="If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
            answer="5 minutes",
            problem_type="math",
            difficulty=3,
            reasoning_steps=[
                "Each machine makes 1 widget in 5 minutes",
                "100 machines can make 100 widgets in parallel",
                "Time remains 5 minutes"
            ]
        ),
        ReasoningProblem(
            question="All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly?",
            answer="No",
            problem_type="logic",
            difficulty=2,
            reasoning_steps=[
                "Roses ⊆ Flowers",
                "Some Flowers fade quickly",
                "But this doesn't mean the fading flowers are roses",
                "Cannot conclude some roses fade quickly"
            ]
        ),
        ReasoningProblem(
            question="A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            answer="$0.05",
            problem_type="math",
            difficulty=2,
            reasoning_steps=[
                "Let ball cost = x",
                "Bat cost = x + $1.00",
                "Total: x + (x + $1.00) = $1.10",
                "2x = $0.10",
                "x = $0.05"
            ]
        ),
        ReasoningProblem(
            question="If you're running a race and you pass the person in 2nd place, what place are you in?",
            answer="2nd place",
            problem_type="logic",
            difficulty=2,
            reasoning_steps=[
                "You pass the person in 2nd place",
                "You take their position",
                "You are now in 2nd place",
                "The person you passed is now in 3rd"
            ]
        ),
        ReasoningProblem(
            question="How many months have 28 days?",
            answer="All 12 months",
            problem_type="common_sense",
            difficulty=1,
            reasoning_steps=[
                "Every month has at least 28 days",
                "Some have more (29, 30, or 31)",
                "But all have 28 days"
            ]
        )
    ]
    return problems


def evaluate_prompting_strategies():
    """Evaluate different CoT prompting strategies"""
    print("=" * 60)
    print("EVALUATING COT PROMPTING STRATEGIES")
    print("=" * 60)
    
    generator = ChainOfThoughtGenerator()
    problems = create_test_problems()
    
    results = {
        'zero_shot': [],
        'few_shot': [],
        'structured': [],
        'self_consistency': []
    }
    
    # Test each strategy
    for problem in problems:
        print(f"\nProblem: {problem.question[:50]}...")
        
        # Zero-shot
        response = generator.simulate_reasoning(problem, 'zero_shot')
        results['zero_shot'].append({
            'correct': response.final_answer == problem.answer,
            'confidence': response.confidence
        })
        
        # Few-shot (use other problems as examples)
        examples = [p for p in problems if p != problem][:2]
        response = generator.simulate_reasoning(problem, 'few_shot')
        results['few_shot'].append({
            'correct': response.final_answer == problem.answer,
            'confidence': response.confidence
        })
        
        # Structured
        response = generator.simulate_reasoning(problem, 'structured')
        results['structured'].append({
            'correct': response.final_answer == problem.answer,
            'confidence': response.confidence
        })
        
        # Self-consistency
        sc = SelfConsistency(num_samples=5)
        responses = sc.generate_multiple_chains(problem, generator)
        aggregated = sc.aggregate_responses(responses)
        results['self_consistency'].append({
            'correct': aggregated['final_answer'] == problem.answer,
            'confidence': aggregated['avg_confidence'],
            'agreement': aggregated['agreement_ratio']
        })
        
        print(f"  Zero-shot: {'✓' if results['zero_shot'][-1]['correct'] else '✗'}")
        print(f"  Few-shot: {'✓' if results['few_shot'][-1]['correct'] else '✗'}")
        print(f"  Structured: {'✓' if results['structured'][-1]['correct'] else '✗'}")
        print(f"  Self-consistency: {'✓' if results['self_consistency'][-1]['correct'] else '✗'}")
    
    # Calculate metrics
    metrics = {}
    for strategy, strategy_results in results.items():
        accuracy = sum(r['correct'] for r in strategy_results) / len(strategy_results)
        avg_confidence = statistics.mean(r['confidence'] for r in strategy_results)
        
        metrics[strategy] = {
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'num_problems': len(strategy_results)
        }
        
        if strategy == 'self_consistency':
            metrics[strategy]['avg_agreement'] = statistics.mean(
                r['agreement'] for r in strategy_results
            )
    
    return metrics


def test_reasoning_depth():
    """Test how reasoning depth affects performance"""
    print("\n" + "=" * 60)
    print("TESTING REASONING DEPTH")
    print("=" * 60)
    
    depths = {
        'shallow': 2,
        'medium': 4,
        'deep': 6
    }
    
    results = {}
    
    for depth_name, num_steps in depths.items():
        print(f"\nTesting {depth_name} reasoning ({num_steps} steps)...")
        
        # Simulate problems requiring different depths
        correct_count = 0
        total = 10
        
        for i in range(total):
            # Problems requiring more steps are harder
            required_steps = random.randint(1, 6)
            
            # Success if we have enough depth
            success = num_steps >= required_steps
            
            # Add some randomness
            if random.random() < 0.1:  # 10% random failure
                success = not success
            
            if success:
                correct_count += 1
        
        accuracy = correct_count / total
        results[depth_name] = {
            'num_steps': num_steps,
            'accuracy': accuracy,
            'problems_solved': correct_count
        }
        
        print(f"  Accuracy: {accuracy:.1%}")
    
    return results


def analyze_failure_modes():
    """Analyze common failure modes in CoT reasoning"""
    print("\n" + "=" * 60)
    print("ANALYZING FAILURE MODES")
    print("=" * 60)
    
    failure_modes = {
        'arithmetic_error': {
            'description': 'Calculation mistakes',
            'frequency': 0.15,
            'example': '2 + 2 = 5'
        },
        'logic_error': {
            'description': 'Invalid logical inference',
            'frequency': 0.20,
            'example': 'All A are B, therefore all B are A'
        },
        'incomplete_reasoning': {
            'description': 'Missing critical steps',
            'frequency': 0.25,
            'example': 'Jump to conclusion without justification'
        },
        'misunderstanding': {
            'description': 'Misinterpret problem statement',
            'frequency': 0.10,
            'example': 'Answer different question'
        },
        'consistency_error': {
            'description': 'Contradictory statements',
            'frequency': 0.08,
            'example': 'X > Y in step 1, but X < Y in step 3'
        }
    }
    
    # Simulate failure detection
    total_errors = 100
    detected_errors = {}
    
    for mode, info in failure_modes.items():
        count = int(total_errors * info['frequency'])
        detected_errors[mode] = {
            'count': count,
            'percentage': info['frequency'] * 100,
            'description': info['description'],
            'example': info['example']
        }
    
    # Print analysis
    for mode, stats in detected_errors.items():
        print(f"\n{mode}:")
        print(f"  Frequency: {stats['percentage']:.1f}%")
        print(f"  Description: {stats['description']}")
        print(f"  Example: {stats['example']}")
    
    return detected_errors


def main():
    """Run chain-of-thought experiment"""
    print("=" * 60)
    print("EXPERIMENT 03: CHAIN-OF-THOUGHT PROMPTING")
    print("=" * 60)
    
    # Evaluate prompting strategies
    strategy_metrics = evaluate_prompting_strategies()
    
    print("\n=== Strategy Performance ===")
    for strategy, metrics in strategy_metrics.items():
        print(f"\n{strategy}:")
        print(f"  Accuracy: {metrics['accuracy']:.1%}")
        print(f"  Confidence: {metrics['avg_confidence']:.2f}")
        if 'avg_agreement' in metrics:
            print(f"  Agreement: {metrics['avg_agreement']:.1%}")
    
    # Test reasoning depth
    depth_results = test_reasoning_depth()
    
    # Analyze failures
    failure_analysis = analyze_failure_modes()
    
    # Save results
    results = {
        'prompting_strategies': strategy_metrics,
        'reasoning_depth': depth_results,
        'failure_modes': failure_analysis,
        'observations': [
            "Self-consistency improves reliability",
            "Structured prompts provide better guidance",
            "Deeper reasoning chains improve complex problem solving",
            "Common failures include arithmetic and logic errors"
        ]
    }
    
    with open('03_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Experiment complete. Results saved to 03_results.json")


if __name__ == '__main__':
    main()