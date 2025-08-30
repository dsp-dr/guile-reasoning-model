#!/usr/bin/env python3
"""
PDF Book Processing Tool
========================
A comprehensive tool for splitting PDFs based on working areas with TOC extraction
and page number alignment capabilities.
"""

import os
import re
import json
import click
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
from tabulate import tabulate

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

from PyPDF2 import PdfReader, PdfWriter


@dataclass
class PageInfo:
    """Information about a PDF page"""
    physical_page: int  # 0-based index in PDF
    book_page: Optional[int]  # Actual page number in book
    content_preview: str
    is_toc: bool = False
    is_chapter_start: bool = False
    chapter_title: Optional[str] = None


@dataclass
class TocEntry:
    """Table of Contents entry"""
    title: str
    level: int
    book_page: Optional[int]
    physical_page: int
    children: List['TocEntry'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


@dataclass 
class SplitConfig:
    """Configuration for PDF splitting"""
    max_pages_per_split: int = 50
    split_by_chapters: bool = True
    include_toc_in_splits: bool = True
    preserve_chapter_boundaries: bool = True
    min_pages_per_split: int = 10
    output_format: str = "{base}_part_{num:03d}_{title}.pdf"


class PDFAnalyzer:
    """Analyzes PDF structure and content"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.reader = PdfReader(pdf_path)
        self.num_pages = len(self.reader.pages)
        self.pages_info: List[PageInfo] = []
        self.toc_entries: List[TocEntry] = []
        self.page_offset = 0  # Difference between physical and book page numbers
        
    def analyze_structure(self) -> Dict[str, Any]:
        """Analyze overall PDF structure"""
        click.echo(f"Analyzing PDF: {self.pdf_path}")
        
        metadata = self._extract_metadata()
        self._detect_page_numbering()
        self._extract_toc()
        self._detect_chapters()
        
        return {
            'metadata': metadata,
            'num_pages': self.num_pages,
            'page_offset': self.page_offset,
            'toc_entries': len(self.toc_entries),
            'chapters_detected': sum(1 for p in self.pages_info if p.is_chapter_start)
        }
    
    def _extract_metadata(self) -> Dict[str, Any]:
        """Extract PDF metadata"""
        meta = self.reader.metadata
        if meta:
            return {
                'title': meta.get('/Title', 'Unknown'),
                'author': meta.get('/Author', 'Unknown'),
                'subject': meta.get('/Subject', ''),
                'creator': meta.get('/Creator', ''),
            }
        return {'title': 'Unknown', 'author': 'Unknown'}
    
    def _detect_page_numbering(self):
        """Detect page numbering scheme and offset"""
        click.echo("Detecting page numbering...")
        
        for i in range(min(20, self.num_pages)):
            page = self.reader.pages[i]
            text = page.extract_text()
            
            # Look for page numbers at bottom
            lines = text.strip().split('\n')
            if lines:
                last_line = lines[-1].strip()
                # Check for standalone numbers
                if re.match(r'^\d+$', last_line):
                    detected_page = int(last_line)
                    if i == 0 and detected_page > 1:
                        # Likely frontmatter
                        continue
                    self.page_offset = detected_page - (i + 1)
                    click.echo(f"Page offset detected: {self.page_offset}")
                    break
                    
        # Initialize page info
        for i in range(self.num_pages):
            page = self.reader.pages[i]
            text = page.extract_text()
            preview = text[:200].replace('\n', ' ') if text else ''
            
            book_page = i + 1 + self.page_offset if self.page_offset else None
            
            self.pages_info.append(PageInfo(
                physical_page=i,
                book_page=book_page,
                content_preview=preview
            ))
    
    def _extract_toc(self):
        """Extract Table of Contents"""
        click.echo("Extracting Table of Contents...")
        
        # Try to get bookmarks/outlines
        if HAS_PYMUPDF:
            self._extract_toc_pymupdf()
        else:
            self._extract_toc_pypdf2()
            
        # Also try to detect TOC pages in content
        self._detect_toc_pages()
    
    def _extract_toc_pymupdf(self):
        """Extract TOC using PyMuPDF"""
        try:
            doc = fitz.open(self.pdf_path)
            toc = doc.get_toc()
            
            for level, title, page in toc:
                entry = TocEntry(
                    title=title,
                    level=level,
                    physical_page=page - 1,  # Convert to 0-based
                    book_page=self.pages_info[page - 1].book_page if page <= len(self.pages_info) else None
                )
                self.toc_entries.append(entry)
                
            doc.close()
        except Exception as e:
            click.echo(f"Warning: Could not extract TOC with PyMuPDF: {e}")
    
    def _extract_toc_pypdf2(self):
        """Extract TOC using PyPDF2"""
        try:
            outlines = self.reader.outline
            if outlines:
                self._process_outline(outlines, level=1)
        except Exception as e:
            click.echo(f"Warning: Could not extract TOC with PyPDF2: {e}")
    
    def _process_outline(self, outline, level=1):
        """Process PDF outline recursively"""
        for item in outline:
            if isinstance(item, dict):
                page_num = self.reader.get_destination_page_number(item)
                entry = TocEntry(
                    title=item.get('/Title', 'Unknown'),
                    level=level,
                    physical_page=page_num,
                    book_page=self.pages_info[page_num].book_page if page_num < len(self.pages_info) else None
                )
                self.toc_entries.append(entry)
            elif isinstance(item, list):
                self._process_outline(item, level + 1)
    
    def _detect_toc_pages(self):
        """Detect TOC pages in content"""
        toc_keywords = ['table of contents', 'contents', 'toc']
        
        for i in range(min(30, self.num_pages)):
            page = self.reader.pages[i]
            text = page.extract_text().lower()
            
            if any(keyword in text for keyword in toc_keywords):
                self.pages_info[i].is_toc = True
                # Parse TOC content
                self._parse_toc_content(i, text)
    
    def _parse_toc_content(self, page_num: int, text: str):
        """Parse TOC content from page text"""
        lines = text.split('\n')
        
        chapter_pattern = re.compile(r'(chapter\s+\d+|part\s+\w+|section\s+\d+)[:\s]+(.+?)[\s.]*(\d+)', re.IGNORECASE)
        
        for line in lines:
            match = chapter_pattern.search(line)
            if match:
                title = match.group(2).strip()
                page = int(match.group(3))
                
                # Find corresponding physical page
                physical_page = self._book_to_physical_page(page)
                
                entry = TocEntry(
                    title=title,
                    level=1 if 'part' in match.group(1).lower() else 2,
                    book_page=page,
                    physical_page=physical_page
                )
                self.toc_entries.append(entry)
    
    def _book_to_physical_page(self, book_page: int) -> int:
        """Convert book page number to physical page index"""
        if self.page_offset:
            return book_page - self.page_offset - 1
        return book_page - 1
    
    def _detect_chapters(self):
        """Detect chapter starts in content"""
        chapter_patterns = [
            re.compile(r'^(CHAPTER|Chapter)\s+(\d+|[IVX]+)', re.MULTILINE),
            re.compile(r'^(PART|Part)\s+(\d+|[IVX]+)', re.MULTILINE),
            re.compile(r'^(\d+)\.\s+[A-Z]', re.MULTILINE),  # Numbered sections
        ]
        
        for i, page_info in enumerate(self.pages_info):
            page = self.reader.pages[i]
            text = page.extract_text()
            
            for pattern in chapter_patterns:
                match = pattern.search(text[:500])  # Check beginning of page
                if match:
                    page_info.is_chapter_start = True
                    page_info.chapter_title = match.group(0)
                    break


class PDFSplitter:
    """Splits PDF into manageable chunks"""
    
    def __init__(self, analyzer: PDFAnalyzer, config: SplitConfig):
        self.analyzer = analyzer
        self.config = config
        self.reader = PdfReader(analyzer.pdf_path)
        
    def split(self, output_dir: Path) -> List[Dict[str, Any]]:
        """Split PDF according to configuration"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config.split_by_chapters and self.analyzer.toc_entries:
            return self._split_by_toc(output_dir)
        else:
            return self._split_by_pages(output_dir)
    
    def _split_by_toc(self, output_dir: Path) -> List[Dict[str, Any]]:
        """Split PDF based on TOC entries"""
        splits = []
        
        # Group TOC entries into splits
        current_split = []
        current_pages = 0
        
        for i, entry in enumerate(self.analyzer.toc_entries):
            # Determine pages for this section
            start_page = entry.physical_page
            end_page = (self.analyzer.toc_entries[i + 1].physical_page 
                       if i + 1 < len(self.analyzer.toc_entries) 
                       else self.analyzer.num_pages)
            
            section_pages = end_page - start_page
            
            # Check if we should start a new split
            if (current_pages + section_pages > self.config.max_pages_per_split and 
                current_pages >= self.config.min_pages_per_split):
                # Save current split
                if current_split:
                    splits.append(self._create_split(current_split, output_dir, len(splits) + 1))
                current_split = [entry]
                current_pages = section_pages
            else:
                current_split.append(entry)
                current_pages += section_pages
        
        # Save final split
        if current_split:
            splits.append(self._create_split(current_split, output_dir, len(splits) + 1))
        
        return splits
    
    def _split_by_pages(self, output_dir: Path) -> List[Dict[str, Any]]:
        """Split PDF by page count"""
        splits = []
        
        for i in range(0, self.analyzer.num_pages, self.config.max_pages_per_split):
            start = i
            end = min(i + self.config.max_pages_per_split, self.analyzer.num_pages)
            
            writer = PdfWriter()
            for j in range(start, end):
                writer.add_page(self.reader.pages[j])
            
            # Generate filename
            part_num = len(splits) + 1
            filename = self.config.output_format.format(
                base=self.analyzer.pdf_path.stem,
                num=part_num,
                title=f"pages_{start+1}-{end}"
            )
            
            output_path = output_dir / filename
            with open(output_path, 'wb') as f:
                writer.write(f)
            
            splits.append({
                'part': part_num,
                'filename': filename,
                'start_page': start + 1,
                'end_page': end,
                'page_count': end - start
            })
        
        return splits
    
    def _create_split(self, toc_entries: List[TocEntry], output_dir: Path, part_num: int) -> Dict[str, Any]:
        """Create a split from TOC entries"""
        if not toc_entries:
            return None
        
        start_page = toc_entries[0].physical_page
        # Find end page
        last_entry = toc_entries[-1]
        next_entry_idx = self.analyzer.toc_entries.index(last_entry) + 1
        
        if next_entry_idx < len(self.analyzer.toc_entries):
            end_page = self.analyzer.toc_entries[next_entry_idx].physical_page
        else:
            end_page = self.analyzer.num_pages
        
        writer = PdfWriter()
        for i in range(start_page, end_page):
            writer.add_page(self.reader.pages[i])
        
        # Generate descriptive filename
        title = toc_entries[0].title.replace('/', '-').replace(' ', '_')[:50]
        filename = self.config.output_format.format(
            base=self.analyzer.pdf_path.stem,
            num=part_num,
            title=title
        )
        
        output_path = output_dir / filename
        with open(output_path, 'wb') as f:
            writer.write(f)
        
        return {
            'part': part_num,
            'filename': filename,
            'chapters': [e.title for e in toc_entries],
            'start_page': start_page + 1,
            'end_page': end_page,
            'page_count': end_page - start_page
        }


@click.group()
def cli():
    """PDF Book Processing Tool"""
    pass


@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True))
def analyze(pdf_path):
    """Analyze PDF structure and metadata"""
    analyzer = PDFAnalyzer(pdf_path)
    results = analyzer.analyze_structure()
    
    click.echo("\n=== PDF Analysis Results ===")
    click.echo(f"File: {pdf_path}")
    click.echo(f"Total Pages: {results['num_pages']}")
    click.echo(f"Page Offset: {results['page_offset']}")
    click.echo(f"TOC Entries: {results['toc_entries']}")
    click.echo(f"Chapters Detected: {results['chapters_detected']}")
    
    if results['metadata']:
        click.echo("\n=== Metadata ===")
        for key, value in results['metadata'].items():
            if value:
                click.echo(f"{key}: {value}")


@cli.command('extract-toc')
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output file for TOC (JSON or YAML)')
def extract_toc(pdf_path, output):
    """Extract table of contents from PDF"""
    analyzer = PDFAnalyzer(pdf_path)
    analyzer.analyze_structure()
    
    if not analyzer.toc_entries:
        click.echo("No TOC entries found")
        return
    
    # Display TOC
    click.echo("\n=== Table of Contents ===")
    table_data = []
    for entry in analyzer.toc_entries:
        indent = "  " * (entry.level - 1)
        table_data.append([
            f"{indent}{entry.title}",
            entry.book_page or '-',
            entry.physical_page + 1
        ])
    
    click.echo(tabulate(table_data, headers=['Title', 'Book Page', 'PDF Page'], tablefmt='grid'))
    
    # Save to file if requested
    if output:
        toc_data = [asdict(e) for e in analyzer.toc_entries]
        
        if output.endswith('.yaml') or output.endswith('.yml'):
            with open(output, 'w') as f:
                yaml.dump(toc_data, f, default_flow_style=False)
        else:
            with open(output, 'w') as f:
                json.dump(toc_data, f, indent=2)
        
        click.echo(f"\nTOC saved to: {output}")


@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='output', help='Output directory for split PDFs')
@click.option('--max-pages', '-m', default=50, help='Maximum pages per split')
@click.option('--by-chapters/--by-pages', default=True, help='Split by chapters or by page count')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file (YAML)')
def split(pdf_path, output_dir, max_pages, by_chapters, config):
    """Split PDF into smaller chunks"""
    
    # Load configuration
    if config:
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
        split_config = SplitConfig(**config_data)
    else:
        split_config = SplitConfig(
            max_pages_per_split=max_pages,
            split_by_chapters=by_chapters
        )
    
    # Analyze PDF
    click.echo(f"Analyzing PDF: {pdf_path}")
    analyzer = PDFAnalyzer(pdf_path)
    analyzer.analyze_structure()
    
    # Split PDF
    click.echo(f"\nSplitting PDF...")
    output_path = Path(output_dir)
    splitter = PDFSplitter(analyzer, split_config)
    splits = splitter.split(output_path)
    
    # Display results
    click.echo(f"\n=== Split Results ===")
    click.echo(f"Created {len(splits)} splits in {output_path}")
    
    table_data = []
    for split in splits:
        chapters = split.get('chapters', [])
        chapter_str = chapters[0] if chapters else f"Pages {split['start_page']}-{split['end_page']}"
        if len(chapters) > 1:
            chapter_str += f" (+{len(chapters)-1} more)"
        
        table_data.append([
            split['part'],
            split['filename'],
            split['page_count'],
            chapter_str
        ])
    
    click.echo(tabulate(table_data, headers=['Part', 'Filename', 'Pages', 'Content'], tablefmt='grid'))


@cli.command('create-config')
@click.option('--output', '-o', default='split-config.yaml', help='Output configuration file')
def create_config(output):
    """Create a sample configuration file"""
    config = SplitConfig()
    
    config_dict = {
        'max_pages_per_split': config.max_pages_per_split,
        'split_by_chapters': config.split_by_chapters,
        'include_toc_in_splits': config.include_toc_in_splits,
        'preserve_chapter_boundaries': config.preserve_chapter_boundaries,
        'min_pages_per_split': config.min_pages_per_split,
        'output_format': config.output_format,
        '# Additional options': {
            'custom_split_points': [
                {'title': 'Part 1', 'start_page': 1, 'end_page': 100},
                {'title': 'Part 2', 'start_page': 101, 'end_page': 200}
            ]
        }
    }
    
    with open(output, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    click.echo(f"Configuration template created: {output}")
    click.echo("Edit this file to customize splitting behavior")


if __name__ == '__main__':
    cli()