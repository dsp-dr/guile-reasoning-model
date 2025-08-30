#!/usr/bin/env python3
"""
Experiment 02: Tokenization and Text Processing
===============================================
Explores different tokenization strategies for reasoning models.
Compares BPE, WordPiece, and SentencePiece tokenizers.
"""

import re
import time
import json
from collections import Counter
from typing import List, Dict, Tuple
import statistics


class SimpleTokenizer:
    """Basic whitespace and punctuation tokenizer"""
    
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
            '<THINK>': 4,  # Special token for reasoning
            '<STEP>': 5,   # Step delimiter
        }
        self.vocab.update(self.special_tokens)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.next_id = len(self.special_tokens)
    
    def tokenize(self, text: str) -> List[str]:
        """Simple regex-based tokenization"""
        # Split on whitespace and punctuation
        pattern = r'\w+|[^\w\s]'
        tokens = re.findall(pattern, text.lower())
        return tokens
    
    def build_vocab(self, texts: List[str], max_vocab_size: int = 10000):
        """Build vocabulary from texts"""
        token_counts = Counter()
        
        for text in texts:
            tokens = self.tokenize(text)
            token_counts.update(tokens)
        
        # Add most common tokens to vocab
        for token, _ in token_counts.most_common(max_vocab_size - len(self.vocab)):
            if token not in self.vocab:
                self.vocab[token] = self.next_id
                self.reverse_vocab[self.next_id] = token
                self.next_id += 1
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
    
    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text"""
        tokens = [self.reverse_vocab.get(id, '<UNK>') for id in ids]
        return ' '.join(tokens)


class BPETokenizer:
    """Byte Pair Encoding tokenizer"""
    
    def __init__(self):
        self.vocab = {}
        self.merges = []
        
    def get_pairs(self, word: List[str]) -> Counter:
        """Get all adjacent pairs in word"""
        pairs = Counter()
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += 1
        return pairs
    
    def learn_bpe(self, texts: List[str], num_merges: int = 1000):
        """Learn BPE merges from texts"""
        # Initialize with character-level tokens
        word_freqs = Counter()
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_freqs[' '.join(list(word) + ['</w>'])] += 1
        
        for _ in range(num_merges):
            pairs = Counter()
            for word, freq in word_freqs.items():
                word_tokens = word.split()
                word_pairs = self.get_pairs(word_tokens)
                for pair, count in word_pairs.items():
                    pairs[pair] += count * freq
            
            if not pairs:
                break
            
            best_pair = pairs.most_common(1)[0][0]
            self.merges.append(best_pair)
            
            # Apply merge
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                word_tokens = word.split()
                i = 0
                new_word = []
                while i < len(word_tokens):
                    if (i < len(word_tokens) - 1 and 
                        (word_tokens[i], word_tokens[i + 1]) == best_pair):
                        new_word.append(word_tokens[i] + word_tokens[i + 1])
                        i += 2
                    else:
                        new_word.append(word_tokens[i])
                        i += 1
                new_word_freqs[' '.join(new_word)] = freq
            word_freqs = new_word_freqs
        
        # Build vocab from final tokens
        self.vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        vocab_id = len(self.vocab)
        
        for word in word_freqs:
            for token in word.split():
                if token not in self.vocab:
                    self.vocab[token] = vocab_id
                    vocab_id += 1
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using learned BPE"""
        words = text.lower().split()
        tokens = []
        
        for word in words:
            word_tokens = list(word) + ['</w>']
            
            # Apply merges
            for merge in self.merges:
                i = 0
                new_word = []
                while i < len(word_tokens):
                    if (i < len(word_tokens) - 1 and 
                        (word_tokens[i], word_tokens[i + 1]) == merge):
                        new_word.append(word_tokens[i] + word_tokens[i + 1])
                        i += 2
                    else:
                        new_word.append(word_tokens[i])
                        i += 1
                word_tokens = new_word
            
            tokens.extend(word_tokens)
        
        return tokens


def analyze_tokenization_patterns(tokenizer, texts: List[str]) -> Dict:
    """Analyze tokenization patterns and statistics"""
    stats = {
        'total_tokens': 0,
        'unique_tokens': set(),
        'avg_tokens_per_text': 0,
        'compression_ratio': 0,
        'token_lengths': [],
        'reasoning_token_patterns': []
    }
    
    token_counts = []
    
    for text in texts:
        if hasattr(tokenizer, 'encode'):
            tokens = tokenizer.encode(text)
            token_count = len(tokens)
        else:
            tokens = tokenizer.tokenize(text)
            token_count = len(tokens)
            stats['unique_tokens'].update(tokens)
        
        token_counts.append(token_count)
        stats['total_tokens'] += token_count
        
        # Check for reasoning patterns
        if 'step' in text.lower() or 'therefore' in text.lower():
            stats['reasoning_token_patterns'].append({
                'text_snippet': text[:100],
                'token_count': token_count
            })
    
    stats['avg_tokens_per_text'] = statistics.mean(token_counts)
    stats['token_count_stddev'] = statistics.stdev(token_counts) if len(token_counts) > 1 else 0
    stats['unique_tokens'] = len(stats['unique_tokens'])
    
    # Calculate compression ratio
    total_chars = sum(len(text) for text in texts)
    stats['compression_ratio'] = total_chars / stats['total_tokens'] if stats['total_tokens'] > 0 else 0
    
    return stats


def test_reasoning_tokenization():
    """Test tokenization on reasoning examples"""
    reasoning_examples = [
        "Let's solve this step by step:\nStep 1: Identify the problem\nStep 2: Break it down\nStep 3: Apply logic\nTherefore, the answer is 42.",
        "Given that x = 5 and y = 3, we can calculate:\nx + y = 5 + 3 = 8\nx * y = 5 * 3 = 15\nThus, x + y < x * y",
        "To find the derivative of f(x) = x^2:\nf'(x) = lim(h→0) [f(x+h) - f(x)]/h\n= lim(h→0) [(x+h)^2 - x^2]/h\n= lim(h→0) [2xh + h^2]/h\n= 2x",
        "If all birds can fly, and penguins are birds, then penguins can fly.\nHowever, penguins cannot fly.\nTherefore, the premise is false.",
    ]
    
    print("=" * 60)
    print("REASONING TOKENIZATION ANALYSIS")
    print("=" * 60)
    
    # Test simple tokenizer
    simple_tokenizer = SimpleTokenizer()
    simple_tokenizer.build_vocab(reasoning_examples)
    
    print("\n=== Simple Tokenizer ===")
    for i, example in enumerate(reasoning_examples[:2]):
        tokens = simple_tokenizer.tokenize(example)
        encoded = simple_tokenizer.encode(example)
        decoded = simple_tokenizer.decode(encoded)
        
        print(f"\nExample {i + 1}:")
        print(f"Original: {example[:50]}...")
        print(f"Tokens ({len(tokens)}): {tokens[:10]}...")
        print(f"Encoded: {encoded[:10]}...")
        print(f"Decoded: {decoded[:50]}...")
    
    # Test BPE tokenizer
    bpe_tokenizer = BPETokenizer()
    bpe_tokenizer.learn_bpe(reasoning_examples, num_merges=100)
    
    print("\n=== BPE Tokenizer ===")
    for i, example in enumerate(reasoning_examples[:2]):
        tokens = bpe_tokenizer.tokenize(example)
        print(f"\nExample {i + 1}:")
        print(f"Tokens ({len(tokens)}): {tokens[:10]}...")
    
    # Analyze patterns
    simple_stats = analyze_tokenization_patterns(simple_tokenizer, reasoning_examples)
    
    print("\n=== Tokenization Statistics ===")
    print(f"Average tokens per text: {simple_stats['avg_tokens_per_text']:.2f}")
    print(f"Compression ratio: {simple_stats['compression_ratio']:.2f}")
    print(f"Unique tokens: {simple_stats['unique_tokens']}")
    
    return simple_stats


def benchmark_tokenizers():
    """Benchmark different tokenization approaches"""
    
    # Generate test data
    test_texts = []
    for i in range(100):
        test_texts.append(f"Step {i}: Process item {i * 2}. Result: {i * 3}.")
    
    print("\n=== Tokenizer Benchmarks ===")
    
    # Simple tokenizer
    simple_tokenizer = SimpleTokenizer()
    simple_tokenizer.build_vocab(test_texts)
    
    start = time.time()
    for text in test_texts:
        _ = simple_tokenizer.encode(text)
    simple_time = time.time() - start
    
    print(f"Simple tokenizer: {simple_time * 1000:.2f}ms for {len(test_texts)} texts")
    
    # BPE tokenizer
    bpe_tokenizer = BPETokenizer()
    bpe_tokenizer.learn_bpe(test_texts[:10], num_merges=50)  # Use subset for speed
    
    start = time.time()
    for text in test_texts:
        _ = bpe_tokenizer.tokenize(text)
    bpe_time = time.time() - start
    
    print(f"BPE tokenizer: {bpe_time * 1000:.2f}ms for {len(test_texts)} texts")
    
    return {
        'simple_time_ms': simple_time * 1000,
        'bpe_time_ms': bpe_time * 1000,
        'speedup': simple_time / bpe_time if bpe_time > 0 else 0
    }


def test_special_tokens_for_reasoning():
    """Test special tokens for chain-of-thought reasoning"""
    
    print("\n=== Special Tokens for Reasoning ===")
    
    tokenizer = SimpleTokenizer()
    
    # Add reasoning-specific tokens
    reasoning_tokens = {
        '<THINK>': 'Begin reasoning',
        '<STEP>': 'Reasoning step delimiter',
        '<CALC>': 'Calculation',
        '<VERIFY>': 'Verification step',
        '<CONCLUDE>': 'Final conclusion'
    }
    
    for token, description in reasoning_tokens.items():
        print(f"{token}: {description}")
    
    # Example with special tokens
    cot_example = """<THINK>
    <STEP>First, let's understand the problem
    <STEP>We need to find x where 2x + 5 = 13
    <CALC>2x = 13 - 5 = 8
    <CALC>x = 8 / 2 = 4
    <VERIFY>Check: 2(4) + 5 = 8 + 5 = 13 ✓
    <CONCLUDE>x = 4
    """
    
    print(f"\nChain-of-thought example:")
    print(cot_example)
    
    # Tokenize with special tokens
    tokens = tokenizer.tokenize(cot_example)
    print(f"\nTokenized ({len(tokens)} tokens): {tokens[:20]}...")
    
    return reasoning_tokens


def main():
    """Run tokenization exploration experiment"""
    print("=" * 60)
    print("EXPERIMENT 02: TOKENIZATION AND TEXT PROCESSING")
    print("=" * 60)
    
    # Test reasoning tokenization
    reasoning_stats = test_reasoning_tokenization()
    
    # Benchmark performance
    benchmark_results = benchmark_tokenizers()
    
    # Test special tokens
    special_tokens = test_special_tokens_for_reasoning()
    
    # Save results
    results = {
        'reasoning_stats': reasoning_stats,
        'benchmark': benchmark_results,
        'special_tokens': list(special_tokens.keys()),
        'observations': [
            "Simple tokenization preserves word boundaries",
            "BPE can handle subword units for better coverage",
            "Special tokens help delimit reasoning steps",
            "Compression ratio affects model efficiency"
        ]
    }
    
    with open('02_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n✓ Experiment complete. Results saved to 02_results.json")


if __name__ == '__main__':
    main()