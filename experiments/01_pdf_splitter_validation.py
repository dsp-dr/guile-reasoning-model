#!/usr/bin/env python3
"""
Experiment 01: PDF Splitter Validation
======================================
Validates PDF splitting tool with org-mode generated content.
Tests TOC extraction, page alignment, and chapter detection.
"""

import os
import subprocess
import json
from pathlib import Path
import tempfile
import sys

sys.path.append('..')
from pdf_splitter import PDFAnalyzer, PDFSplitter, SplitConfig


def generate_test_pdf():
    """Convert org-mode to PDF using pandoc"""
    print("Converting org-mode to PDF...")
    
    # Check if pandoc is available
    try:
        subprocess.run(['pandoc', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Pandoc not found. Installing...")
        subprocess.run(['pkg', 'install', '-y', 'pandoc'], check=True)
    
    # Convert org to PDF
    cmd = [
        'pandoc',
        'test-book.org',
        '-o', 'test-book.pdf',
        '--pdf-engine=xelatex',
        '--toc',
        '--number-sections',
        '-V', 'geometry:margin=1in',
        '-V', 'fontsize=12pt'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Fallback to simpler conversion
            cmd = ['pandoc', 'test-book.org', '-o', 'test-book.pdf']
            subprocess.run(cmd, check=True)
        print("PDF generated successfully")
        return 'test-book.pdf'
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return None


def analyze_pdf_structure(pdf_path):
    """Analyze PDF structure and validate detection"""
    print(f"\nAnalyzing PDF: {pdf_path}")
    
    analyzer = PDFAnalyzer(pdf_path)
    results = analyzer.analyze_structure()
    
    print("\n=== Analysis Results ===")
    print(f"Total Pages: {results['num_pages']}")
    print(f"Page Offset: {results['page_offset']}")
    print(f"TOC Entries: {results['toc_entries']}")
    print(f"Chapters Detected: {results['chapters_detected']}")
    
    # Validate TOC extraction
    if analyzer.toc_entries:
        print("\n=== TOC Validation ===")
        print(f"First entry: {analyzer.toc_entries[0].title}")
        print(f"Last entry: {analyzer.toc_entries[-1].title}")
        
        # Check for expected chapters
        expected_chapters = [
            "Introduction to Reasoning Models",
            "Mathematical Foundations",
            "Neural Network Basics",
            "Building the Core Architecture"
        ]
        
        found_chapters = [e.title for e in analyzer.toc_entries]
        for expected in expected_chapters:
            if any(expected in title for title in found_chapters):
                print(f"✓ Found: {expected}")
            else:
                print(f"✗ Missing: {expected}")
    
    return analyzer


def test_splitting_strategies(analyzer, pdf_path):
    """Test different splitting configurations"""
    print("\n=== Testing Splitting Strategies ===")
    
    configs = [
        ("By Chapters", SplitConfig(max_pages_per_split=30, split_by_chapters=True)),
        ("By Pages", SplitConfig(max_pages_per_split=20, split_by_chapters=False)),
        ("Small Chunks", SplitConfig(max_pages_per_split=10, min_pages_per_split=5))
    ]
    
    results = {}
    
    for name, config in configs:
        print(f"\nTesting: {name}")
        output_dir = Path(f"output_{name.lower().replace(' ', '_')}")
        
        splitter = PDFSplitter(analyzer, config)
        splits = splitter.split(output_dir)
        
        results[name] = {
            'num_splits': len(splits),
            'splits': splits,
            'output_dir': str(output_dir)
        }
        
        print(f"  Created {len(splits)} splits")
        
        # Validate splits
        total_pages = sum(s['page_count'] for s in splits)
        print(f"  Total pages: {total_pages}")
        
        # Check file sizes
        for split in splits[:3]:  # Check first 3
            file_path = output_dir / split['filename']
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  {split['filename']}: {size_mb:.2f} MB")
    
    return results


def validate_page_alignment(analyzer):
    """Validate page number detection and alignment"""
    print("\n=== Page Alignment Validation ===")
    
    # Check if page numbers were detected
    if analyzer.page_offset != 0:
        print(f"Page offset detected: {analyzer.page_offset}")
        
        # Validate some pages
        for i in [0, 10, 20]:
            if i < len(analyzer.pages_info):
                page = analyzer.pages_info[i]
                print(f"Physical page {page.physical_page + 1} -> Book page {page.book_page}")
    else:
        print("No page offset detected (may be correct for generated PDF)")
    
    # Check chapter detection
    chapter_pages = [p for p in analyzer.pages_info if p.is_chapter_start]
    print(f"\nChapter start pages: {len(chapter_pages)}")
    
    for page in chapter_pages[:5]:  # Show first 5
        print(f"  Page {page.physical_page + 1}: {page.chapter_title}")


def performance_metrics(analyzer, splits_results):
    """Calculate performance metrics"""
    print("\n=== Performance Metrics ===")
    
    metrics = {
        'toc_extraction_rate': len(analyzer.toc_entries) / 10,  # Expected ~10 chapters
        'chapter_detection_rate': sum(1 for p in analyzer.pages_info if p.is_chapter_start) / 10,
        'splitting_efficiency': {}
    }
    
    for strategy, results in splits_results.items():
        if results['splits']:
            avg_pages = sum(s['page_count'] for s in results['splits']) / len(results['splits'])
            variance = sum((s['page_count'] - avg_pages) ** 2 for s in results['splits']) / len(results['splits'])
            
            metrics['splitting_efficiency'][strategy] = {
                'num_splits': len(results['splits']),
                'avg_pages_per_split': avg_pages,
                'variance': variance,
                'balance_score': 1 / (1 + variance)  # Higher is better
            }
    
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    """Run PDF splitter validation experiment"""
    print("=" * 60)
    print("EXPERIMENT 01: PDF SPLITTER VALIDATION")
    print("=" * 60)
    
    # Generate test PDF
    pdf_path = generate_test_pdf()
    if not pdf_path:
        # Use existing PDF if generation fails
        pdf_path = '../tmp/Build_a_Reasoning_Model_(From_Scratch)_v1_MEAP.pdf'
        if not Path(pdf_path).exists():
            print("No PDF available for testing")
            return
    
    # Analyze structure
    analyzer = analyze_pdf_structure(pdf_path)
    
    # Test splitting
    splits_results = test_splitting_strategies(analyzer, pdf_path)
    
    # Validate alignment
    validate_page_alignment(analyzer)
    
    # Calculate metrics
    metrics = performance_metrics(analyzer, splits_results)
    
    # Save results
    with open('01_results.json', 'w') as f:
        json.dump({
            'pdf_path': pdf_path,
            'analysis': {
                'num_pages': analyzer.num_pages,
                'toc_entries': len(analyzer.toc_entries),
                'page_offset': analyzer.page_offset
            },
            'splits': splits_results,
            'metrics': metrics
        }, f, indent=2)
    
    print("\n✓ Experiment complete. Results saved to 01_results.json")


if __name__ == '__main__':
    main()